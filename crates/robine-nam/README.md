# robine-nam

Minimal, allocation-free, audio-driver-agnostic runtime for
[Neural Amp Modeler](https://www.neuralampmodeler.com) **A2** ("nam2")
models: `Size::Small` (3 channels, A2-Lite / nano) and `Size::Large`
(8 channels, A2-Full / standard), as bare `WaveNet` files or inside a
`SlimmableContainer`. One dependency (`serde_json`), `unsafe` confined to
the SIMD wrapper (NEON, SSE/FMA, scalar fallback).

```rust
use robine_nam::{Model, NamFile, Size};

let file = NamFile::read("amp.nam")?;      // off the audio thread
let mut model = file.build(Size::Large)?;  // prewarmed, ready
let mut buf = vec![0.0f32; 256];
model.process_in_place(&mut buf);          // RT-safe: no allocation, no locks
# Ok::<(), robine_nam::Error>(())
```

* Output matches the upstream NeuralAmpModelerCore reference to ~1e-7
  (SNR > 120 dB), verified by golden tests and by the measurement tools in
  the repository.
* Any block length; results are bit-identical whatever the chunking.
* No resampling, no gain staging, no I/O: `Metadata` exposes loudness and
  calibration levels so the host can do the usual NAM gain staging.

Build and measure in release mode only. Full documentation, benchmarks and
tools: <https://github.com/bailleul-dev/robine-nam>.
