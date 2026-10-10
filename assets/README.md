# assets/

Local-only test material, populated by `scripts/download_assets.py`
(`make assets`). Nothing in here is committed except this file.

| directory          | content                                             | source / license |
|--------------------|-----------------------------------------------------|------------------|
| `inputs/`          | dry DI guitar/bass WAVs (48 kHz, 24-bit, mono)      | TONE3000 `neural-amp-modeler-wasm` demo files, local testing only |
| `irs/`             | cabinet / reverb impulse responses                  | same |
| `models/upstream/` | example `.nam` models from NeuralAmpModelerCore     | MIT |
| `models/<tone>-<id>/` | A2 captures downloaded from TONE3000            | per-tone license (mostly T3K: personal use, no redistribution) |
| `manifest.json`    | provenance, sizes and SHA-256 of every file         | generated |
