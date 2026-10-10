# Changelog

## 0.1.0 — 2026-10-10

Initial release.

* `robine-nam`: NAM A2 small (3 ch) / large (8 ch) runtime, bare `WaveNet`
  or `SlimmableContainer`, prewarm with cached reset, chunk-independent
  output, NEON / SSE-FMA / scalar backends, golden tests against the
  upstream NeuralAmpModelerCore reference.
* `tools/`: `nam-reference-sys` (upstream C++ via FFI) and `nam-lab`
  (inspect, render, compare, compare-wav, bench, golden).
* `scripts/download_assets.py`: public DI inputs and IRs, upstream example
  models, TONE3000 A2 captures.
