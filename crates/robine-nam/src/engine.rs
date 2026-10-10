//! Allocation-free A2 WaveNet engine, monomorphised over the channel count.
//!
//! Design notes (everything here runs on the audio thread after construction):
//!
//! * Frames are stored as `[f32; L]` where `L` is the SIMD lane count: 8 for
//!   the large model (two NEON / one AVX register) and 4 for the small one
//!   (its 3 channels are padded with a zero lane so that every column update
//!   is a single vector FMA). All pad weights are zero, so pad lanes stay
//!   zero through the whole network. Inner loops are broadcast-FMA column
//!   updates that LLVM vectorises without intrinsics.
//! * Every convolution history is a power-of-two ring with a `max_block`
//!   wide tail mirror: a block read of any tap is always one contiguous
//!   slice, and the write cost is O(block) with no periodic `memmove`.
//! * Frames are processed in register tiles (`T` frames x `S` independent
//!   accumulator sets, chosen per model size in `model.rs`) so that each
//!   weight column is loaded once per tile and tap, the accumulators stay in
//!   registers across all taps and input channels, and the FMA dependency
//!   chains are short enough to keep the pipes busy.
//! * Kernel sizes are const-generic (6 and 15) so the tap loop unrolls.
//! * The arithmetic is performed per frame in a fixed order, so chunking the
//!   input differently produces bit-identical output.

use crate::arch::{
    layer_lookback, DILATIONS, HEAD_KERNEL, KERNEL_SIZES, NUM_LAYERS, RECEPTIVE_FIELD,
};
use crate::error::Error;

use crate::simd::V4;

fn next_pow2(v: usize) -> usize {
    v.next_power_of_two().max(1)
}

/// Power-of-two ring of `[f32; C]` frames with a tail mirror of `max_block`
/// frames so that any block read is contiguous.
#[derive(Clone)]
struct Ring<const C: usize> {
    data: Vec<[f32; C]>,
    size: usize,
    mask: usize,
    write: usize,
    max_block: usize,
}

impl<const C: usize> Ring<C> {
    fn new(lookback: usize, max_block: usize) -> Self {
        let size = next_pow2(lookback + max_block);
        Self {
            data: vec![[0.0; C]; size + max_block],
            size,
            mask: size - 1,
            write: 0,
            max_block,
        }
    }

    /// Set every frame (and the mirror) to `frame`.
    fn fill(&mut self, frame: [f32; C]) {
        self.data.fill(frame);
    }

    /// Append a block of frames (`frames.len() <= max_block`).
    #[inline]
    fn push(&mut self, frames: &[[f32; C]]) {
        let n = frames.len();
        debug_assert!(n <= self.max_block);
        let wp = self.write;
        let first = n.min(self.size - wp);
        self.data[wp..wp + first].copy_from_slice(&frames[..first]);
        // Mirror the part of the first segment that lies in [0, max_block).
        let mirror_end = (wp + first).min(self.max_block);
        if wp < mirror_end {
            self.data.copy_within(wp..mirror_end, self.size + wp);
        }
        if first < n {
            let rest = n - first;
            self.data[..rest].copy_from_slice(&frames[first..]);
            self.data.copy_within(0..rest, self.size);
        }
        self.write = (wp + n) & self.mask;
    }

    /// Index of the first frame of the block of `n` frames that was just
    /// pushed, shifted `delay` samples into the past. The `n` frames starting
    /// there are contiguous in `data` thanks to the mirror.
    #[inline(always)]
    fn base(&self, n: usize, delay: usize) -> usize {
        self.write.wrapping_sub(n + delay) & self.mask
    }

    fn last(&self) -> [f32; C] {
        self.data[self.write.wrapping_sub(1) & self.mask]
    }
}

#[derive(Clone)]
struct Layer<const L: usize> {
    dilation: usize,
    /// `conv_w[k * L + j]` is the output column for input lane `j` at tap `k`.
    conv_w: Vec<[f32; L]>,
    conv_b: [f32; L],
    mixin_w: [f32; L],
    /// `l1x1_w[j]` is the output column for bottleneck lane `j`.
    l1x1_w: [[f32; L]; L],
    l1x1_b: [f32; L],
    hist: Ring<L>,
    /// Steady-state history frame after prewarm, used by `reset`.
    cached: [f32; L],
}

#[derive(Clone)]
pub(crate) struct Engine<
    const L: usize,
    const NV: usize,
    const C: usize,
    const S: usize,
    const T: usize,
> {
    rechannel_w: [f32; L],
    layers: Vec<Layer<L>>,
    head_w: [[f32; L]; HEAD_KERNEL],
    head_b: f32,
    head_scale: f32,
    head_hist: Ring<L>,
    head_cached: [f32; L],
    // Scratch, sized to max_block frames.
    cond: Vec<f32>,
    layer_in: Vec<[f32; L]>,
    head_sum: Vec<[f32; L]>,
    max_block: usize,
}

impl<const L: usize, const NV: usize, const C: usize, const S: usize, const T: usize>
    Engine<L, NV, C, S, T>
{
    /// Build an engine with `C` logical channels stored in `L >= C` lanes.
    // Index loops mirror the upstream weight order on purpose.
    #[allow(clippy::needless_range_loop)]
    pub(crate) fn new(weights: &[f32], max_block: usize) -> Result<Self, Error> {
        const { assert!(C <= L && NV * 4 == L, "channels must fit in the lane count") };
        let c = C;
        let expected = crate::arch::weight_count(c);
        if weights.len() != expected {
            return Err(Error::WeightCount {
                expected,
                found: weights.len(),
            });
        }
        let max_block = max_block.max(1);
        let mut it = weights.iter().copied();
        let mut take = || it.next().expect("weight count already validated");

        let mut rechannel_w = [0.0; L];
        for w in rechannel_w.iter_mut().take(c) {
            *w = take();
        }

        let mut layers = Vec::with_capacity(NUM_LAYERS);
        for li in 0..NUM_LAYERS {
            let k = KERNEL_SIZES[li];
            // Column `j` of tap `tap` lives at `tap * L + j`; pad columns stay zero.
            let mut conv_w = vec![[0.0f32; L]; k * L];
            for i in 0..c {
                for j in 0..c {
                    for tap in 0..k {
                        conv_w[tap * L + j][i] = take();
                    }
                }
            }
            let mut conv_b = [0.0; L];
            for b in conv_b.iter_mut().take(c) {
                *b = take();
            }
            let mut mixin_w = [0.0; L];
            for m in mixin_w.iter_mut().take(c) {
                *m = take();
            }
            let mut l1x1_w = [[0.0; L]; L];
            for i in 0..c {
                for j in 0..c {
                    l1x1_w[j][i] = take();
                }
            }
            let mut l1x1_b = [0.0; L];
            for b in l1x1_b.iter_mut().take(c) {
                *b = take();
            }
            layers.push(Layer {
                dilation: DILATIONS[li],
                conv_w,
                conv_b,
                mixin_w,
                l1x1_w,
                l1x1_b,
                hist: Ring::new(layer_lookback(li), max_block),
                cached: [0.0; L],
            });
        }

        let mut head_w = [[0.0; L]; HEAD_KERNEL];
        for j in 0..c {
            for k in 0..HEAD_KERNEL {
                head_w[k][j] = take();
            }
        }
        let head_b = take();
        let head_scale = take();
        debug_assert!(it.next().is_none());

        let mut engine = Self {
            rechannel_w,
            layers,
            head_w,
            head_b,
            head_scale,
            head_hist: Ring::new(HEAD_KERNEL - 1, max_block),
            head_cached: [0.0; L],
            cond: vec![0.0; max_block],
            layer_in: vec![[0.0; L]; max_block],
            head_sum: vec![[0.0; L]; max_block],
            max_block,
        };
        engine.prewarm();
        Ok(engine)
    }

    pub(crate) fn max_block(&self) -> usize {
        self.max_block
    }

    /// Run the receptive field worth of silence through the model and cache
    /// the resulting steady state so that `reset` is cheap.
    fn prewarm(&mut self) {
        let mut remaining = RECEPTIVE_FIELD;
        let mut scratch = vec![0.0f32; self.max_block];
        while remaining > 0 {
            let n = remaining.min(self.max_block);
            self.cond[..n].fill(0.0);
            self.run_block(n, &mut scratch[..n]);
            remaining -= n;
        }
        for layer in &mut self.layers {
            layer.cached = layer.hist.last();
        }
        self.head_cached = self.head_hist.last();
    }

    /// Restore the prewarmed steady state. Allocation-free.
    pub(crate) fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.hist.fill(layer.cached);
        }
        self.head_hist.fill(self.head_cached);
    }

    /// Process one block (`input.len() == output.len() <= max_block`).
    #[inline]
    pub(crate) fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        let n = input.len();
        debug_assert_eq!(n, output.len());
        debug_assert!(n <= self.max_block);
        self.cond[..n].copy_from_slice(input);
        self.run_block(n, output);
    }

    /// Process one block in place.
    #[inline]
    pub(crate) fn process_block_in_place(&mut self, buf: &mut [f32]) {
        let n = buf.len();
        debug_assert!(n <= self.max_block);
        self.cond[..n].copy_from_slice(buf);
        self.run_block(n, buf);
    }

    #[allow(clippy::needless_range_loop)]
    fn run_block(&mut self, n: usize, output: &mut [f32]) {
        let cond = &self.cond[..n];
        let layer_in = &mut self.layer_in[..n];
        let head_sum = &mut self.head_sum[..n];
        for f in 0..n {
            let x = cond[f];
            for c in 0..L {
                layer_in[f][c] = self.rechannel_w[c] * x;
            }
            head_sum[f] = [0.0; L];
        }
        for layer in &mut self.layers {
            match layer.conv_w.len() / L {
                6 => layer_forward::<L, NV, C, S, T, 6>(layer, cond, layer_in, head_sum),
                15 => layer_forward::<L, NV, C, S, T, 15>(layer, cond, layer_in, head_sum),
                _ => unreachable!("A2 kernel sizes are 6 and 15"),
            }
        }
        self.head_hist.push(head_sum);
        let mut bases = [0usize; HEAD_KERNEL];
        for (k, base) in bases.iter_mut().enumerate() {
            *base = self.head_hist.base(n, HEAD_KERNEL - 1 - k);
        }
        let data = &self.head_hist.data;
        for (f, out) in output.iter_mut().enumerate() {
            let mut acc = [V4::zero(); NV];
            for k in 0..HEAD_KERNEL {
                let h = &data[bases[k] + f];
                let w = &self.head_w[k];
                for v in 0..NV {
                    acc[v] = acc[v].fma(V4::load(&w[v * 4..]), V4::load(&h[v * 4..]));
                }
            }
            let mut y = self.head_b;
            for v in 0..NV {
                y += acc[v].hsum();
            }
            *out = y * self.head_scale;
        }
    }
}

/// One dilated layer over a block: push history, then dilated conv with
/// input mixin and bias, LeakyReLU, accumulate into the head sum, and the
/// 1x1 residual back into `layer_in`.
///
/// Frames are processed in register tiles of `T` frames (plus a
/// single-frame tail); the accumulators stay in registers across all taps
/// and input channels, every weight column is loaded once per tile and
/// tap, and each update is one lane-broadcast FMA.
#[inline(always)]
fn layer_forward<
    const L: usize,
    const NV: usize,
    const C: usize,
    const S: usize,
    const T: usize,
    const K: usize,
>(
    layer: &mut Layer<L>,
    cond: &[f32],
    layer_in: &mut [[f32; L]],
    head_sum: &mut [[f32; L]],
) {
    let n = cond.len();
    let layer_in = &mut layer_in[..n];
    let head_sum = &mut head_sum[..n];
    layer.hist.push(layer_in);

    let mut bases = [0usize; K];
    for (k, base) in bases.iter_mut().enumerate() {
        *base = layer.hist.base(n, (K - 1 - k) * layer.dilation);
    }

    let mut f = 0;
    while f + T <= n {
        layer_tile::<L, NV, C, S, K, T>(layer, &bases, f, cond, layer_in, head_sum);
        f += T;
    }
    while f < n {
        layer_tile::<L, NV, C, S, K, 1>(layer, &bases, f, cond, layer_in, head_sum);
        f += 1;
    }
}

/// `T` consecutive frames of one layer, starting at frame `f`.
///
/// `S` independent accumulator sets per frame (input channel `j` goes to
/// set `j % S`) shorten the FMA dependency chains so that the FMA pipes
/// stay busy; the sets are summed once at the end of the taps.
// Explicit index loops over tiny const-sized arrays: they fully unroll and
// keep everything in registers.
#[allow(clippy::needless_range_loop)]
#[inline(always)]
fn layer_tile<
    const L: usize,
    const NV: usize,
    const C: usize,
    const S: usize,
    const K: usize,
    const T: usize,
>(
    layer: &Layer<L>,
    bases: &[usize; K],
    f: usize,
    cond: &[f32],
    layer_in: &mut [[f32; L]],
    head_sum: &mut [[f32; L]],
) {
    const { assert!(NV * 4 == L && C <= L && C <= 8 && S >= 1 && S <= C) };
    let hist: &[[f32; L]] = &layer.hist.data;
    let conv_w: &[[f32; L]] = &layer.conv_w;

    // acc[t][s][v]: accumulator set `s`, lanes 4v..4v+4 of frame f+t.
    let mut acc = [[[V4::zero(); NV]; S]; T];
    for t in 0..T {
        let x = cond[f + t];
        for v in 0..NV {
            acc[t][0][v] =
                V4::load(&layer.conv_b[v * 4..]).fma_splat(V4::load(&layer.mixin_w[v * 4..]), x);
        }
    }

    for k in 0..K {
        let base = bases[k] + f;
        let h: &[[f32; L]] = &hist[base..base + T];
        let wk: &[[f32; L]] = &conv_w[k * L..k * L + C];
        let mut hv = [[V4::zero(); NV]; T];
        for t in 0..T {
            for v in 0..NV {
                hv[t][v] = V4::load(&h[t][v * 4..]);
            }
        }
        // Textually unrolled over the input channel `j` so that `j % S`,
        // `j / 4` and `j % 4` are compile-time constants: with a plain
        // `for j in 0..C` LLVM keeps the accumulators in memory.
        macro_rules! conv_step {
            ($($j:literal)*) => { $( if $j < C {
                let col = &wk[$j];
                for v in 0..NV {
                    let cv = V4::load(&col[v * 4..]);
                    for t in 0..T {
                        acc[t][$j % S][v] = acc[t][$j % S][v].fma_lane(cv, hv[t][$j / 4], $j % 4);
                    }
                }
            } )* };
        }
        conv_step!(0 1 2 3 4 5 6 7);
    }

    // Post-conv, tiled as well: activations for all T frames first, then
    // the 1x1 residual with every weight column loaded once per tile.
    let mut a = [[V4::zero(); NV]; T];
    for t in 0..T {
        let mut at = acc[t][0];
        for s in 1..S {
            for v in 0..NV {
                at[v] = at[v].add(acc[t][s][v]);
            }
        }
        for v in 0..NV {
            at[v] = at[v].leaky_relu();
        }
        a[t] = at;
        let hs = &mut head_sum[f + t];
        for v in 0..NV {
            V4::load(&hs[v * 4..]).add(at[v]).store(&mut hs[v * 4..]);
        }
    }
    let mut r = [[[V4::zero(); NV]; S]; T];
    for t in 0..T {
        for v in 0..NV {
            r[t][0][v] = V4::load(&layer.l1x1_b[v * 4..]);
        }
    }
    macro_rules! residual_step {
        ($($j:literal)*) => { $( if $j < C {
            let col = &layer.l1x1_w[$j];
            for v in 0..NV {
                let cv = V4::load(&col[v * 4..]);
                for t in 0..T {
                    r[t][$j % S][v] = r[t][$j % S][v].fma_lane(cv, a[t][$j / 4], $j % 4);
                }
            }
        } )* };
    }
    residual_step!(0 1 2 3 4 5 6 7);
    for t in 0..T {
        let li = &mut layer_in[f + t];
        for v in 0..NV {
            let mut rv = r[t][0][v];
            for s in 1..S {
                rv = rv.add(r[t][s][v]);
            }
            V4::load(&li[v * 4..]).add(rv).store(&mut li[v * 4..]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_reads_are_contiguous_and_consistent() {
        let max_block = 5;
        let mut ring = Ring::<2>::new(7, max_block);
        let mut expected: Vec<[f32; 2]> = vec![[0.0; 2]; 64];
        let mut t = 0usize;
        for block in [1usize, 5, 3, 5, 5, 2, 4, 5, 1, 5, 5, 5] {
            let frames: Vec<[f32; 2]> = (0..block)
                .map(|i| [(t + i) as f32, -((t + i) as f32)])
                .collect();
            expected.extend_from_slice(&frames);
            ring.push(&frames);
            t += block;
            for delay in 0..=7 {
                let base = ring.base(block, delay);
                let got = &ring.data[base..base + block];
                let start = expected.len() - block - delay;
                assert_eq!(
                    got,
                    &expected[start..start + block],
                    "block {block} delay {delay}"
                );
            }
        }
    }
}
