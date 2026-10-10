//! Fixed architecture description of NAM "A2" (nam2) WaveNet models.
//!
//! Both supported sizes share the exact same topology; only the channel
//! count differs:
//!
//! | size  | channels | upstream name |
//! |-------|----------|---------------|
//! | Small | 3        | A2-Lite / nano |
//! | Large | 8        | A2-Full / standard |
//!
//! The numbers here mirror `NAM/wavenet/a2_fast.h` in NeuralAmpModelerCore.

/// Number of dilated layers in the single A2 layer array.
pub const NUM_LAYERS: usize = 23;
/// Kernel size of the head rechannel convolution (channels -> 1).
pub const HEAD_KERNEL: usize = 16;
/// Negative slope of the LeakyReLU used by every layer.
pub const LEAKY_SLOPE: f32 = 0.01;
/// Channel count of the small (A2-Lite / nano) model.
pub const SMALL_CHANNELS: usize = 3;
/// Channel count of the large (A2-Full / standard) model.
pub const LARGE_CHANNELS: usize = 8;

/// Per-layer kernel sizes.
pub const KERNEL_SIZES: [usize; NUM_LAYERS] = [
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6,
];
/// Per-layer dilations.
pub const DILATIONS: [usize; NUM_LAYERS] = [
    1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239,
];

/// Look-back of a layer in samples: `(kernel_size - 1) * dilation`.
pub const fn layer_lookback(layer: usize) -> usize {
    (KERNEL_SIZES[layer] - 1) * DILATIONS[layer]
}

const fn total_lookback() -> usize {
    let mut sum = 0;
    let mut i = 0;
    while i < NUM_LAYERS {
        sum += layer_lookback(i);
        i += 1;
    }
    sum
}

/// Receptive field of the model in samples, which is also the number of
/// zero samples processed during prewarm. Matches the upstream count
/// (`1 + sum(lookbacks) + head_kernel - 1`).
pub const RECEPTIVE_FIELD: usize = 1 + total_lookback() + (HEAD_KERNEL - 1);

/// Number of weights an A2 model with `channels` channels must carry.
///
/// Layout (upstream order): rechannel (C), then per layer
/// `K*C*C + C` (conv + bias), `C` (input mixin), `C*C + C` (1x1 + bias),
/// then head conv `16*C + 1` and the trailing `head_scale`.
pub const fn weight_count(channels: usize) -> usize {
    let c = channels;
    let mut n = c;
    let mut i = 0;
    while i < NUM_LAYERS {
        let k = KERNEL_SIZES[i];
        n += k * c * c + c + c + c * c + c;
        i += 1;
    }
    n + HEAD_KERNEL * c + 1 + 1
}

/// Model size, i.e. which of the two A2 variants a model is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Size {
    /// 3 channels (A2-Lite, "nano").
    Small,
    /// 8 channels (A2-Full, "standard").
    Large,
}

impl Size {
    /// Channel count of this size.
    pub const fn channels(self) -> usize {
        match self {
            Size::Small => SMALL_CHANNELS,
            Size::Large => LARGE_CHANNELS,
        }
    }

    /// Size for a channel count, if it is one of the two supported ones.
    pub const fn from_channels(channels: usize) -> Option<Size> {
        match channels {
            SMALL_CHANNELS => Some(Size::Small),
            LARGE_CHANNELS => Some(Size::Large),
            _ => None,
        }
    }

    /// Number of weights of a model of this size.
    pub const fn weight_count(self) -> usize {
        weight_count(self.channels())
    }
}

impl core::fmt::Display for Size {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Size::Small => "small",
            Size::Large => "large",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn receptive_field_matches_upstream() {
        // 7 blocks of (5 * 409) + 14 + 182, plus 1, plus 15.
        assert_eq!(RECEPTIVE_FIELD, 1 + 3 * 5 * 409 + 14 + 14 * 13 + 15);
        assert_eq!(RECEPTIVE_FIELD, 6347);
    }

    #[test]
    fn weight_counts_match_example_models() {
        assert_eq!(Size::Small.weight_count(), 1871);
        assert_eq!(Size::Large.weight_count(), 12146);
    }
}
