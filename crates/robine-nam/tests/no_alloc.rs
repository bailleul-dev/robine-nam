//! The real-time contract: once a model is built, processing and resetting
//! never touch the allocator. Enforced with a counting global allocator.

use robine_nam::{Model, NamFile, Size};
use std::alloc::{GlobalAlloc, Layout, System};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::SeqCst);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::SeqCst);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn allocations<R>(f: impl FnOnce() -> R) -> (usize, R) {
    let before = ALLOCS.load(Ordering::SeqCst);
    let r = f();
    (ALLOCS.load(Ordering::SeqCst) - before, r)
}

fn fixture_model(size: Size, max_block: usize) -> Model {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/A2.nam");
    NamFile::read(path)
        .unwrap()
        .build_with(size, max_block)
        .unwrap()
}

#[test]
fn processing_never_allocates() {
    for size in [Size::Small, Size::Large] {
        let mut model = fixture_model(size, 64);
        let input: Vec<f32> = (0..1_000).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        let mut output = vec![0.0f32; input.len()];
        let mut in_place = input.clone();

        // Block-aligned, unaligned, longer than max_block, in place, per sample, reset.
        let (n, _) = allocations(|| {
            model.process(&input[..64], &mut output[..64]);
            model.process(&input[..17], &mut output[..17]);
            model.process(&input, &mut output);
            model.process_in_place(&mut in_place);
            for &x in &input[..100] {
                model.process_sample(x);
            }
            model.reset();
            model.process(&input, &mut output);
        });
        assert_eq!(n, 0, "{size}: {n} allocation(s) on the audio path");
        assert!(output.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn building_does_allocate_so_keep_it_off_the_audio_thread() {
    let (n, model) = allocations(|| fixture_model(Size::Large, 64));
    assert!(n > 0);
    let (n, _clone) = allocations(|| model.clone());
    assert!(n > 0, "clone allocates: do it off the audio thread");
}
