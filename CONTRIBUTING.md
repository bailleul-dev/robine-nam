# Contributing

Thanks for looking at robine-nam. A few rules keep the project honest.

## Layout

* `crates/robine-nam` is the library: one dependency, `#![deny(unsafe_code)]`
  with `unsafe` confined to `src/simd.rs`, nothing that allocates on the
  audio path. Measurement code never goes in here.
* `tools/` is a separate workspace (reference FFI + `nam-lab`). It is meant
  to be extractable, so it depends on the library only through the path
  dependency in `tools/nam-lab/Cargo.toml`.
* `scripts/` are stdlib-only Python helpers (assets, listening sets).

## Before opening a pull request

```sh
make test                 # debug + release + scalar backend, clippy, fmt, doc
make tools && make compare # delta against the upstream reference must PASS
```

Anything touching `src/engine.rs` or `src/simd.rs` also needs numbers:
`make bench` before and after, or better `tools/scripts/ab.sh` with the two
binaries interleaved, and a line in `OPTIMIZATION.md` saying what was tried
and what it gave, including the things that did not work.

Release builds only for anything audio: debug numbers are meaningless.

## What never gets committed

* `.env`, TONE3000 tokens.
* Downloaded captures, DI audio, IRs, listening sets (`assets/` is ignored).
  Most TONE3000 models are licensed for personal use.
* The fetched upstream sources (`tools/nam-reference-sys/vendor/`).

## Scope

The library handles NAM A2 small and large, nothing else, on purpose. Other
architectures, resampling, gain staging and I/O belong to the host or to
other crates.
