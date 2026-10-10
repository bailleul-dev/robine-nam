# Optimization notes

What was tried on the A2 engine, what stayed, what was dropped and why, and
what has not been explored yet. Numbers are best-of-N ns/sample at block 64,
48 kHz, Apple M1, release build; quality is the sample-exact delta against
the upstream NeuralAmpModelerCore generic WaveNet (`nam-lab compare`).

## Where we stand

| | small (3 ch) | large (8 ch) |
|---|---:|---:|
| robine-nam | ~90 ns | ~320-350 ns |
| upstream A2 fast path | ~120 ns | ~860-940 ns |
| upstream generic WaveNet | ~720 ns | ~950 ns |
| vector FMAs per sample | ~560 | ~2 950 |
| achieved FMA rate | ~6.4 /ns | ~9.3 /ns |
| machine peak (16 independent chains, register only) | 11.5 /ns | 11.5 /ns |

Quality never moved through any kept step: max abs 3e-7 to 2e-6 (output
scale 0.5 to 1), ESR ~1e-13, SNR 124 to 133 dB, the same order as the gap
between upstream's own two code paths. Output is bit-identical whatever the
block size.

## Kept

| what | why |
|------|-----|
| `[f32; L]` frames, 4 lanes for the small model (3 channels + 1 zero pad), 8 for the large | one frame = 1 or 2 NEON registers; every column update is one vector FMA; pad lanes stay zero because all pad weights are zero |
| explicit `V4` SIMD type (NEON / SSE+FMA / scalar) instead of autovectorisation | LLVM only vectorised loops whose accumulators lived in memory; with loop-carried accumulators it emitted scalar `fmadd` (step 0: 262/540 ns) |
| register tiles: T frames x S accumulator sets, all taps and channels inside | weight columns loaded once per tile and tap; S independent chains per frame keep the FMA pipes busy (chains of length C/S per tap instead of C) |
| tile shapes small T=8/S=2, large T=4/S=2 | register budget (32 NEON): small 16 acc + 8 hist + 1 col, large 16 + 8 + 2; every other shape measured worse (see dropped) |
| channel loop unrolled by a `macro_rules!` with literal indices | with `for j in 0..C` or a closure the `j % S`, `j / 4`, `j % 4` indices are dynamic and the tiles are spilled to the stack (seen in the assembly: 8 FMAs per ~60 instructions) |
| tap loop left as a plain `for k in 0..K` | unrolling it textually cost +10 % (code size), no gain |
| post-conv stage tiled too (activations for T frames, then the 1x1 residual with each weight column loaded once) | small but consistent win, simpler code |
| lane-broadcast FMA (`fma_lane`) for the 1x1 residual instead of extracting a scalar | neutral in speed, fewer instructions |
| power-of-two history rings with a tail mirror, one `memcpy` push per layer and block | any tap read of a block is one contiguous slice; push cost is O(block) with no periodic `memmove` |
| scratch `layer_in` buffer in L1 + push, rather than writing into the next ring directly | per-frame mirror writes into rings that live in L2 lost to one grouped `memcpy` (+10 %) |
| prewarm at load, steady state cached, `reset` just refills the rings | `reset` is allocation-free and O(ring size) |
| dedicated single-sample path (`process_sample`: direct ring writes, no scratch copies, one-frame tiles, same accumulator sets as the block path) | block 1 went from 411 / 873 ns to 313 / 607 ns (upstream: 813 / 1 862) while staying bit-identical to block processing |

## Dropped (measured)

| what | result | why it loses |
|------|--------|--------------|
| tap-major loop with the accumulator in memory (`z[f]` load / FMA / store) | 188 / 676 ns | vectorises, but one load + one store per FMA group: load-bound |
| large T=2/S=4 | +13 % | weight loads amortised over 2 frames only |
| large T=6/S=1, T=8/S=1 | +26 % / noise | 12 to 16 chains of length 8: latency-bound or spilled |
| small T=8/S=1, T=12/S=1 | +5 % / +19 % | chains of length 3 per tap, not enough independent work |
| channel loop unrolled through a closure | +10 % | accumulators still spilled |
| tap loop unrolled textually (K = 6 and 15) | +10 %, larger code | no register benefit, worse i-cache |
| layers write straight into the next layer's ring, no push | +10 % (keeping the delay-0 frames live) / +6 % (reloading them) | 8 more live registers, or one more load per frame, plus per-frame conditional mirror stores into L2 |
| per-tap `(history, weights)` slices precomputed once per layer | +15 % | array of fat pointers loaded per tap is slower than the `bases: [usize; K]` array plus one slice |

## Not explored, with the trade-offs

Things that could still move the numbers, deliberately not done in this
round. Each one either changes the output, adds `unsafe`, adds a target or a
dependency, or needs a platform we have not measured on.

| idea | expected effect | trade-off |
|------|-----------------|-----------|
| unchecked indexing in the tap loop (`get_unchecked`) | small model maybe -5 to -10 % (range checks are a visible share of its non-FMA instructions); large ~0 | `unsafe` outside the SIMD module; the gain is bounded by the ~20 % of non-FMA work |
| fp16 / bf16 weights and histories (NEON `fmlal`, AVX-512 BF16) | up to 2x on the large model | changes the output (A2 captures are trained in fp32; expect ESR 1e-5 to 1e-4 instead of 1e-13), needs per-model validation and a quality gate |
| int8 quantisation | 2-4x | same, worse: the dilated stack amplifies quantisation noise over 23 layers |
| 256-bit kernels on x86 (AVX2 `__m256`, 8 lanes in one register) | large model on x86 could gain ~1.5x over the current 2x SSE path | x86 has only been compile-checked, never benchmarked; needs runtime feature detection or `-C target-cpu`, and a third backend to maintain |
| AVX-512 / SVE / SME (Apple M4+) | more lanes, outer-product units | platform-specific, not measurable here |
| fully generated kernels (one function per (size, K, T) with named registers, e.g. via a build script) | removes the remaining `mov.16b`, `movi`, address arithmetic (~10 % of instructions in the large loop) | code generation machinery for a few percent |
| tile-major order across layers (one tile through all 23 layers before the next tile) | keeps `layer_in` and `head_sum` in registers, no scratch traffic | needs T x NV x 2 more live registers; spills at the current tile shapes |
| rings sized to fit L1 (the three dilation-239 layers alone take 3 x 64 KB at `max_block` 256) | fewer L2 reads per tap | lower `max_block` or a split ring; untested whether the tap reads are a bottleneck at all (hardware prefetch handles them well) |
| head conv (16 taps x NV FMAs + horizontal sum) as a tile | ~1 % of the work | not worth it |
| skipping the last layer's 1x1 residual (its output is unused) | ~1/23 of the residual work, <1 % | trivial, but it is a special case in the kernel |
| multi-threading | none for one model | real-time audio is single-threaded per instance; parallelism belongs to the host (one model per channel) |
| fewer FMAs by algorithm (sparsity, pruning, Winograd-style tricks for dilated 1-D conv) | the only way past the FMA peak | changes the model or the output; out of scope for a runtime |
| `max_block` sweep | ring sizes and chunking only | processing cost per sample is flat from block 32 up; block 16 is ~10 % more expensive (tile tails); block 1 (`process_sample`) is 3.5x small / 1.9x large the block-64 cost because the per-layer fixed work (ring write, bases, one-frame tile, head) is paid per sample |
| shorter FMA chains on the single-sample path (one accumulator set per input channel) | measured -10 % on `process_sample` (313 -> 278 ns small, 607 -> 575 ns large) | changes the summation order, so block-1 output stops being bit-identical to block-64 output (1e-9 differences); dropped to keep the chunk-independence guarantee |

## What has not been measured

* Any x86-64 machine (SSE/FMA path compiles, including with `+fma`, and
  the scalar fallback compiles for wasm32, but no timings).
* Other ARM cores (Cortex-A7x/X), Windows, Linux schedulers.
* Denormal behaviour on x86 without FTZ/DAZ (the host is expected to set
  them, as for any DSP).
* Power consumption, which matters on pedals and embedded targets.

## Method

* `nam-lab bench --brief`: best-of-N and median ns/sample; best-of-N removes
  scheduler noise but not slow drift (a laptop under background load drifts
  by ~10 % over minutes).
* `tools/scripts/ab.sh`: two binaries interleaved in ABBA order. The first
  AB-only version had a ~5 % bias in favour of whichever ran second, which
  would have "confirmed" a false 7 % gain.
* Always read the generated assembly (`RUSTFLAGS="--emit asm"`, count
  `fmla`, `ldr/str` and stack references in the hottest block) when a change
  targets register allocation: twice the source looked right while the
  accumulators lived on the stack.
* A register-only micro-benchmark of `vfmaq_laneq_f32` chains gives the
  machine's real FMA ceiling; compare achieved FMA/ns against it before
  spending effort on a kernel.
