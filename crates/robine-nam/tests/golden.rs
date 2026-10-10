//! Golden test against the upstream NeuralAmpModelerCore reference.
//!
//! The fixtures were produced by `nam-lab golden` (tools/) running the
//! upstream *generic* WaveNet on the upstream `A2.nam` example model, which
//! bundles both the small (3-channel) and large (8-channel) variants.
//! Regenerate with `make golden`.

use robine_nam::{NamFile, Size};
use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn read_f32(name: &str) -> Vec<f32> {
    let bytes = std::fs::read(fixture(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

/// Upstream and robine-nam only differ by floating-point evaluation order
/// (FMA grouping, Eigen GEMM blocking); measured deltas are ~4e-7 on
/// outputs of order 0.5, so 1e-5 is a comfortable but still tight gate.
const MAX_ABS: f32 = 1e-5;
const MAX_ESR: f64 = 1e-10;

fn check(size: Size, max_block: usize) {
    let file = NamFile::read(fixture("A2.nam")).unwrap();
    let mut model = file.build_with(size, max_block).unwrap();
    let input = read_f32("a2.input.f32");
    let expected = read_f32(&format!("a2-{size}.output.f32"));
    assert_eq!(input.len(), expected.len());
    let mut output = vec![0.0; input.len()];
    model.process(&input, &mut output);

    let mut max_abs = 0.0f32;
    let mut err2 = 0.0f64;
    let mut ref2 = 0.0f64;
    for (i, (&o, &e)) in output.iter().zip(&expected).enumerate() {
        let d = (o - e).abs();
        assert!(o.is_finite(), "non-finite output at {i}");
        max_abs = max_abs.max(d);
        err2 += (d as f64).powi(2);
        ref2 += (e as f64).powi(2);
    }
    let esr = err2 / ref2;
    assert!(
        max_abs <= MAX_ABS && esr <= MAX_ESR,
        "{size} max_block {max_block}: max abs {max_abs:e}, ESR {esr:e}"
    );
    assert!(ref2 > 0.0, "reference output is silent");
}

#[test]
fn small_matches_upstream_reference() {
    for max_block in [1, 64, 256] {
        check(Size::Small, max_block);
    }
}

#[test]
fn large_matches_upstream_reference() {
    for max_block in [1, 64, 256] {
        check(Size::Large, max_block);
    }
}

#[test]
fn reset_reproduces_golden_output() {
    let file = NamFile::read(fixture("A2.nam")).unwrap();
    let mut model = file.build(Size::Large).unwrap();
    let input = read_f32("a2.input.f32");
    let mut first = vec![0.0; input.len()];
    model.process(&input, &mut first);
    model.reset();
    let mut second = vec![0.0; input.len()];
    model.process(&input, &mut second);
    assert_eq!(first, second);
}
