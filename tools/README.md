# NAM measurement tools

A standalone Cargo workspace (it is excluded from the root workspace so it can
move to its own repository). Its only link to the library is the path
dependency `robine-nam = { path = "../../crates/robine-nam" }` in
`nam-lab/Cargo.toml`.

* `nam-reference-sys/` compiles the upstream
  [NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore)
  with `NAM_ENABLE_A2_FAST` and exposes it through a tiny C shim. The
  sources are fetched at the commit pinned in `nam-reference-sys/UPSTREAM`
  by `nam-reference-sys/vendor.sh` (run by `make tools`) into a git-ignored
  `vendor/` directory: the root clone stays small and the library never
  needs them, so the upstream *generic* WaveNet and the upstream *A2 fast
  path* can both be driven in-process with the same buffers as robine-nam.
* `nam-lab/` is the CLI:

| command       | purpose |
|---------------|---------|
| `inspect`     | describe a `.nam` (variants, metadata) and, with `--build`, confirm every implementation accepts it |
| `render`      | render a WAV through `robine`, `ref-generic`, `ref-fast` or `ref-auto` |
| `compare`     | sample-exact delta robine vs upstream (max abs, RMS, ESR, SNR) for several block sizes; non-zero exit above `--tolerance` |
| `compare-wav` | quality metrics between any two renders: lag estimate, gain match, ESR, SNR, correlation, log-spectral distance |
| `bench`       | ns/sample, real-time factor, CPU % and per-block p50/p99/p99.9/max latency per implementation, size and block |
| `golden`      | produce the `tests/fixtures` golden vectors from the upstream generic path |

Always use release builds (`cargo build --release`); the CLI prints a warning
otherwise and the numbers are meaningless.

```sh
make tools        # fetches the pinned upstream, builds the C++ and the CLI
cd tools
./target/release/nam-lab compare -m ../assets/models/upstream/A2.nam
./target/release/nam-lab bench -m ../assets/models/upstream/A2.nam --blocks 16,64,256
./target/release/nam-lab render -m amp.nam -i "../assets/inputs/Brit - Guitar.wav" -o out.wav
```

## A/B of two builds

Benchmarks on a laptop drift by several percent over minutes (background
load, thermal state), so compare two builds interleaved, never one after the
other:

```sh
cp target/release/nam-lab /tmp/nam-lab-before   # build A
# ... change the library, rebuild ...
tools/scripts/ab.sh /tmp/nam-lab-before target/release/nam-lab
```

`bench --brief` prints one line per configuration with the best and median
ns/sample; `bench` without it prints the full latency table.
