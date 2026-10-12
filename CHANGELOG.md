# Changelog

## Unreleased

* The measurement tools (`nam-reference-sys`, `nam-lab`, the listening-set
  script) moved to robine-wavelab, where they share their measures with the
  tools of robine-spice. This repository keeps the library, its tests and
  the asset downloader.

## 0.1.0 — 2026-10-10

Initial release.

* `robine-nam`: NAM A2 small (3 ch) / large (8 ch) runtime, bare `WaveNet`
  or `SlimmableContainer`, prewarm with cached reset, chunk-independent
  output, zero added latency, dedicated single-sample path, NEON / SSE-FMA /
  scalar backends, golden and zero-allocation tests, reference comparison
  against upstream NeuralAmpModelerCore.
* `tools/`: `nam-reference-sys` (upstream C++ via FFI) and `nam-lab`
  (inspect, render, compare, compare-wav, bench, golden).
* `scripts/download_assets.py`: public DI inputs and IRs, upstream example
  models, TONE3000 A2 captures.
