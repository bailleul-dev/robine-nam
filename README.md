# robine-nam

![AI Declaration Format](https://img.shields.io/badge/AI_Declaration_Format-validated-blue)

A small, allocation-free, audio-driver-agnostic Rust runtime for
[Neural Amp Modeler](https://www.neuralampmodeler.com) **A2** ("nam2")
models, plus the measurement tools used to prove it matches the upstream
reference and to track its performance.

It deliberately supports only the two A2 variants:

| size    | channels | upstream name        |
|---------|----------|----------------------|
| `Small` | 3        | A2-Lite / "nano"     |
| `Large` | 8        | A2-Full / "standard" |

either as a bare `WaveNet` file or packed in a `SlimmableContainer` (the
layout TONE3000 and the trainer produce). LSTM, A1 WaveNets, FiLM, gated or
custom shapes are rejected at load time with a reason.

## Library

```toml
[dependencies]
robine-nam = { path = "crates/robine-nam" }   # one dependency: serde_json
```

```rust
use robine_nam::{Model, NamFile, Size};

// From a file on disk...
let file = NamFile::read("amp.nam")?;            // parse once, not on the audio thread
let mut model = file.build(Size::Large)?;         // or file.build_largest()

// ...or from bytes you already have (embedded, downloaded, read from anywhere).
let bytes: &[u8] = include_bytes!("amp.nam");
let mut embedded = Model::parse_bytes(bytes)?;    // largest variant
let small = NamFile::parse_bytes(bytes)?.build(Size::Small)?;
// `NamFile::from_reader(impl Read)` covers files, sockets and resources.

assert_eq!(model.sample_rate(), 48_000.0);        // no resampling is done for you

// Real-time safe from here: no allocation, no locks, any slice length.
let input = vec![0.0f32; 128];
let mut output = vec![0.0f32; 128];
model.process(&input, &mut output);
model.process_in_place(&mut output);
let y = model.process_sample(0.1);
model.reset();                                    // back to the prewarmed state
```

What the library does:

* parses `.nam` JSON (version, metadata, calibration levels, weights) and
  picks the small / large submodel of a container;
* runs the network with the upstream weight order, prewarms the receptive
  field (6347 samples) at load and caches that state so `reset` is free;
* processes blocks of any length, chunked by a `max_block` chosen at build
  time, with bit-identical results whatever the chunking, and **adds no
  latency in any mode**: the network is causal, output `i` depends on inputs
  `0..=i` only (enforced by a test), block processing just computes the
  samples the host already has; `process_sample` has its own dedicated
  path for hosts that run sample by sample;
* allocates nothing once built: `process`, `process_in_place`,
  `process_sample` and `reset` are enforced allocation-free by a test with a
  counting global allocator (`tests/no_alloc.rs`); parsing, building and
  `clone` allocate and belong off the audio thread.

What it does not do: resampling, gain staging, IRs, parameter smoothing,
audio I/O. Those belong to the host.

### Engine design

* `[f32; L]` frames: 8 lanes for the large model, 4 for the small one (3
  channels plus a zero pad so every column update is one vector FMA).
* Power-of-two history rings with a tail mirror: every tap read of a block
  is one contiguous slice, writes are O(block), no periodic `memmove`.
* Register-tiled micro-kernel (4 frames x independent accumulator sets)
  written on a tiny `V4` SIMD type with NEON, SSE/FMA and scalar backends
  (`--features scalar`); `unsafe` is confined to that one module.
* Fixed evaluation order per frame, so results do not depend on block size.

### Accuracy and speed

Measured with `tools/nam-lab` against the upstream
[NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore)
(generic WaveNet and its own A2 fast path), release builds, Apple M1,
48 kHz, upstream `A2.nam` example and real TONE3000 captures:

* delta vs upstream generic: max abs error about 4e-7 (output scale ~0.5),
  error-to-signal ratio ~1e-13, SNR > 120 dB, the same order as the delta
  between upstream's own two code paths;
* [`BENCH.md`](BENCH.md) has the current table (ns/sample, real-time
  factor, per-block latency percentiles) for robine-nam, the upstream fast
  path and the upstream generic path side by side; at block 64 robine-nam
  is roughly 1.3x (small) and 2.5x (large) faster than the upstream A2 fast
  path. Regenerate it with `make bench`.

Always use release builds for anything audio (`cargo build --release`,
`cargo test --release`); the profiles enable LTO and `codegen-units = 1`.

## Repository layout

```
crates/robine-nam/      the library (root Cargo workspace)
  tests/fixtures/       upstream A2.nam (MIT) + golden vectors from the reference
tools/                  separate workspace, extractable to its own repo
  nam-reference-sys/    cc build of the upstream C++ (fetched at a pinned commit into vendor/) + C shim
  nam-lab/              CLI: inspect, render, compare, compare-wav, bench, golden
scripts/download_assets.py   test assets (DI inputs, IRs, upstream models, TONE3000 captures)
assets/                 downloaded material, git-ignored
.env.example            variables read by the Makefile and the script (copy to .env, git-ignored)
```

## Getting started

```sh
git clone https://github.com/bailleul-dev/robine-nam
make test                      # library unit + golden tests (release)
make tools                     # fetches the pinned upstream reference, builds it and nam-lab
make assets                    # public DI inputs, IRs, upstream example models
make compare                   # robine vs upstream, exit code != 0 on mismatch
make bench                     # throughput / latency table (+ target/bench.json)
make listen                    # WAVs to hear: robine vs upstream, diff +60 dB, blind A/B (assets/listening)

# A2 captures from TONE3000 (first run opens a browser for approval):
cp .env.example .env            # then set TONE3000_CLIENT_ID=t3k_pub_... (tone3000.com/settings)
make tone TONE=6580
python3 -I scripts/download_assets.py --search "AC30"
```

`nam-lab` usage is documented in [`tools/README.md`](tools/README.md).

## AI usage

This project was built with intensive use of an AI coding agent (Claude
Code), directed and reviewed by the maintainer, and verified against the
upstream reference implementation. The details are recorded in
[`aidecl.yaml`](aidecl.yaml) following the
[AI Declaration Format](https://ai-declaration.org/); `make aidecl-check`
validates it.

## License

MIT. The upstream NeuralAmpModelerCore fetched by the tools is MIT (Steven
Atkinson); TONE3000 captures and demo audio keep their own licenses and are
never committed.
