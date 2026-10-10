use crate::arch::{Size, RECEPTIVE_FIELD};
use crate::engine::Engine;
use crate::error::Error;
use crate::file::{Metadata, NamFile, Variant};
use std::path::Path;

/// Lane count used to store the small model (3 channels + 1 zero pad).
const SMALL_LANES: usize = 4;
/// Lane count used to store the large model (8 channels).
const LARGE_LANES: usize = 8;
/// Register tiling per size: `TILE` frames x `SPLIT` independent
/// accumulator sets. Register budget (32 NEON registers):
/// small 8 x 2 acc + 8 hist + 1 col = 25, large 4 x 2 x 2 acc + 8 hist + 2 col = 26.
const SMALL_TILE: usize = 8;
const SMALL_SPLIT: usize = 2;
const LARGE_TILE: usize = 4;
const LARGE_SPLIT: usize = 2;

// Both variants are large by nature; boxing would add an indirection on
// the audio path for nothing.
#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
enum Kind {
    Small(
        Engine<
            SMALL_LANES,
            { SMALL_LANES / 4 },
            { Size::Small.channels() },
            SMALL_SPLIT,
            SMALL_TILE,
        >,
    ),
    Large(
        Engine<
            LARGE_LANES,
            { LARGE_LANES / 4 },
            { Size::Large.channels() },
            LARGE_SPLIT,
            LARGE_TILE,
        >,
    ),
}

/// A loaded, prewarmed, ready-to-run A2 model.
///
/// `process`, `process_in_place`, `process_sample` and `reset` never
/// allocate and never lock, so they are safe to call from a real-time
/// audio thread. Everything else (loading, building, cloning) is not.
///
/// `Model` is `Send` and `Sync`; cloning duplicates the whole state (useful
/// to run the same capture on several channels).
///
/// On x86 the host should enable flush-to-zero / denormals-are-zero on the
/// audio thread as it would for any DSP; the engine itself does not touch
/// the FPU control word.
#[derive(Clone)]
pub struct Model {
    kind: Kind,
    size: Size,
    version: String,
    sample_rate: f32,
    metadata: Metadata,
}

impl Model {
    /// Default `max_block` used by the convenience constructors.
    pub const DEFAULT_MAX_BLOCK: usize = 256;

    /// Build from a file variant with the given `max_block`.
    pub fn from_variant(variant: &Variant, max_block: usize) -> Result<Self, Error> {
        let kind = match variant.size {
            Size::Small => Kind::Small(Engine::new(&variant.weights, max_block)?),
            Size::Large => Kind::Large(Engine::new(&variant.weights, max_block)?),
        };
        Ok(Self {
            kind,
            size: variant.size,
            version: variant.version.clone(),
            sample_rate: variant.sample_rate,
            metadata: variant.metadata.clone(),
        })
    }

    /// Parse `.nam` text and build its largest variant.
    pub fn parse(text: &str) -> Result<Self, Error> {
        NamFile::parse(text)?.build_largest()
    }

    /// Parse `.nam` bytes (UTF-8 JSON) and build its largest variant.
    pub fn parse_bytes(bytes: &[u8]) -> Result<Self, Error> {
        NamFile::parse_bytes(bytes)?.build_largest()
    }

    /// Read a `.nam` document from any `Read` and build its largest variant.
    pub fn from_reader(reader: impl std::io::Read) -> Result<Self, Error> {
        NamFile::from_reader(reader)?.build_largest()
    }

    /// Read a `.nam` file and build its largest variant.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Error> {
        NamFile::read(path)?.build_largest()
    }

    /// Read a `.nam` file and build the requested size.
    pub fn load_size(path: impl AsRef<Path>, size: Size) -> Result<Self, Error> {
        NamFile::read(path)?.build(size)
    }

    /// Model size.
    pub fn size(&self) -> Size {
        self.size
    }

    /// Sample rate the model was trained at. Feeding audio at another rate
    /// changes the tone; resampling is the caller's job.
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// `.nam` format version string.
    pub fn version(&self) -> &str {
        &self.version
    }

    /// Embedded metadata.
    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    /// Largest slice handled in one pass by `process`.
    pub fn max_block(&self) -> usize {
        match &self.kind {
            Kind::Small(e) => e.max_block(),
            Kind::Large(e) => e.max_block(),
        }
    }

    /// Receptive field in samples (6347 for A2).
    pub const fn receptive_field(&self) -> usize {
        RECEPTIVE_FIELD
    }

    /// Restore the state right after loading (all histories at the prewarmed
    /// steady state). Allocation-free.
    pub fn reset(&mut self) {
        match &mut self.kind {
            Kind::Small(e) => e.reset(),
            Kind::Large(e) => e.reset(),
        }
    }

    /// Process `input` into `output`. Slices may be any length (processed
    /// in chunks of `max_block`) but must have equal lengths.
    ///
    /// # Panics
    /// If `input.len() != output.len()`.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len(), "input and output lengths differ");
        let max_block = self.max_block();
        for (i, o) in input.chunks(max_block).zip(output.chunks_mut(max_block)) {
            match &mut self.kind {
                Kind::Small(e) => e.process_block(i, o),
                Kind::Large(e) => e.process_block(i, o),
            }
        }
    }

    /// Process `buf` in place.
    pub fn process_in_place(&mut self, buf: &mut [f32]) {
        let max_block = self.max_block();
        for chunk in buf.chunks_mut(max_block) {
            match &mut self.kind {
                Kind::Small(e) => e.process_block_in_place(chunk),
                Kind::Large(e) => e.process_block_in_place(chunk),
            }
        }
    }

    /// Process a single sample. Block processing is cheaper per sample;
    /// prefer `process` when you have more than one.
    pub fn process_sample(&mut self, input: f32) -> f32 {
        let mut out = [0.0f32];
        match &mut self.kind {
            Kind::Small(e) => e.process_block(&[input], &mut out),
            Kind::Large(e) => e.process_block(&[input], &mut out),
        }
        out[0]
    }
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model")
            .field("size", &self.size)
            .field("version", &self.version)
            .field("sample_rate", &self.sample_rate)
            .field("name", &self.metadata.name)
            .field("max_block", &self.max_block())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::file::test_support::wavenet_json;

    fn pseudo_random_weights(count: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.4
            })
            .collect()
    }

    fn test_signal(len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let t = i as f32 / 48_000.0;
                0.3 * (2.0 * core::f32::consts::PI * 110.0 * t).sin()
                    + 0.1 * (2.0 * core::f32::consts::PI * 1_234.5 * t).sin()
            })
            .collect()
    }

    fn random_model(size: Size, max_block: usize) -> Model {
        let weights = pseudo_random_weights(size.weight_count(), 7);
        let file = NamFile::parse(&wavenet_json(size.channels(), weights)).unwrap();
        file.build_with(size, max_block).unwrap()
    }

    #[test]
    fn model_is_send_sync_and_clone() {
        fn assert_send_sync<T: Send + Sync + Clone>() {}
        assert_send_sync::<Model>();
        let input = test_signal(500);
        let mut a = random_model(Size::Large, 64);
        let mut warm = vec![0.0; 100];
        a.process(&input[..100], &mut warm);
        let mut b = a.clone();
        let mut out_a = vec![0.0; input.len()];
        let mut out_b = vec![0.0; input.len()];
        a.process(&input, &mut out_a);
        b.process(&input, &mut out_b);
        assert_eq!(out_a, out_b, "a clone carries the full state");
    }

    #[test]
    fn zero_weights_are_silent() {
        for size in [Size::Small, Size::Large] {
            let file = NamFile::parse(&wavenet_json(
                size.channels(),
                vec![0.0; size.weight_count()],
            ))
            .unwrap();
            let mut model = file.build(size).unwrap();
            let input = vec![0.5; 300];
            let mut output = vec![1.0; 300];
            model.process(&input, &mut output);
            assert!(output.iter().all(|&v| v == 0.0));
        }
    }

    #[test]
    fn chunking_is_bit_exact() {
        for size in [Size::Small, Size::Large] {
            let input = test_signal(1_000);
            let mut reference = vec![0.0; input.len()];
            random_model(size, 256).process(&input, &mut reference);
            assert!(reference.iter().all(|v| v.is_finite()));
            assert!(reference.iter().any(|&v| v != 0.0));

            for block in [1usize, 3, 4, 7, 64, 333, 1_000] {
                let mut model = random_model(size, block);
                let mut output = vec![0.0; input.len()];
                model.process(&input, &mut output);
                assert_eq!(output, reference, "size {size} max_block {block}");
            }

            let mut model = random_model(size, 64);
            let mut output = vec![0.0; input.len()];
            for (i, o) in input.chunks(37).zip(output.chunks_mut(37)) {
                model.process(i, o);
            }
            assert_eq!(output, reference, "size {size} chunked 37");

            let mut model = random_model(size, 64);
            let sample_by_sample: Vec<f32> =
                input.iter().map(|&x| model.process_sample(x)).collect();
            assert_eq!(sample_by_sample, reference, "size {size} per sample");

            let mut model = random_model(size, 50);
            let mut in_place = input.clone();
            model.process_in_place(&mut in_place);
            assert_eq!(in_place, reference, "size {size} in place");
        }
    }

    #[test]
    fn reset_restores_initial_state() {
        let input = test_signal(2_000);
        let mut model = random_model(Size::Large, 128);
        let mut first = vec![0.0; input.len()];
        model.process(&input, &mut first);
        let mut second = vec![0.0; input.len()];
        model.process(&input, &mut second);
        assert_ne!(first, second, "state must carry over between calls");
        model.reset();
        let mut third = vec![0.0; input.len()];
        model.process(&input, &mut third);
        assert_eq!(first, third);
    }

    #[test]
    fn prewarm_makes_silence_stationary() {
        // After prewarm, silence in must give a constant (DC) output.
        let mut model = random_model(Size::Small, 64);
        let input = vec![0.0; 500];
        let mut output = vec![0.0; 500];
        model.process(&input, &mut output);
        let first = output[0];
        assert!(
            output.iter().all(|&v| (v - first).abs() <= 1e-6),
            "{output:?}"
        );
    }
}
