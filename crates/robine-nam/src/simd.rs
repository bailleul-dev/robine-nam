//! Minimal 4-lane f32 SIMD abstraction with NEON (aarch64), SSE/FMA
//! (x86_64) and scalar backends. This is the only module allowed to use
//! `unsafe`, and only for intrinsics on data whose length is checked here.
//!
//! All backends perform the same per-lane operations in the same order, so
//! on hardware with fused multiply-add the scalar backend (`scalar` feature)
//! is bit-identical to the SIMD ones.

#![allow(unsafe_code)]

use crate::arch::LEAKY_SLOPE;

#[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
mod backend {
    use super::LEAKY_SLOPE;
    use core::arch::aarch64::*;

    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct V4(float32x4_t);

    impl V4 {
        #[inline(always)]
        pub fn zero() -> Self {
            unsafe { V4(vdupq_n_f32(0.0)) }
        }
        #[inline(always)]
        pub fn load(p: &[f32]) -> Self {
            assert!(p.len() >= 4);
            unsafe { V4(vld1q_f32(p.as_ptr())) }
        }
        #[inline(always)]
        pub fn store(self, p: &mut [f32]) {
            assert!(p.len() >= 4);
            unsafe { vst1q_f32(p.as_mut_ptr(), self.0) }
        }
        /// `self + col * x`
        #[inline(always)]
        pub fn fma_splat(self, col: V4, x: f32) -> Self {
            unsafe { V4(vfmaq_n_f32(self.0, col.0, x)) }
        }
        /// `self + col * h[lane]`
        #[inline(always)]
        pub fn fma_lane(self, col: V4, h: V4, lane: usize) -> Self {
            unsafe {
                V4(match lane {
                    0 => vfmaq_laneq_f32::<0>(self.0, col.0, h.0),
                    1 => vfmaq_laneq_f32::<1>(self.0, col.0, h.0),
                    2 => vfmaq_laneq_f32::<2>(self.0, col.0, h.0),
                    _ => vfmaq_laneq_f32::<3>(self.0, col.0, h.0),
                })
            }
        }
        /// `self + a * b` (element-wise)
        #[inline(always)]
        pub fn fma(self, a: V4, b: V4) -> Self {
            unsafe { V4(vfmaq_f32(self.0, a.0, b.0)) }
        }
        #[inline(always)]
        pub fn add(self, o: V4) -> Self {
            unsafe { V4(vaddq_f32(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn leaky_relu(self) -> Self {
            unsafe {
                let scaled = vmulq_n_f32(self.0, LEAKY_SLOPE);
                let neg = vcltq_f32(self.0, vdupq_n_f32(0.0));
                V4(vbslq_f32(neg, scaled, self.0))
            }
        }
        #[inline(always)]
        fn lane(self, lane: usize) -> f32 {
            unsafe {
                match lane {
                    0 => vgetq_lane_f32::<0>(self.0),
                    1 => vgetq_lane_f32::<1>(self.0),
                    2 => vgetq_lane_f32::<2>(self.0),
                    _ => vgetq_lane_f32::<3>(self.0),
                }
            }
        }
        /// `((l0 + l1) + l2) + l3`
        #[inline(always)]
        pub fn hsum(self) -> f32 {
            ((self.lane(0) + self.lane(1)) + self.lane(2)) + self.lane(3)
        }
    }
}

#[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
mod backend {
    use super::LEAKY_SLOPE;
    use core::arch::x86_64::*;

    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct V4(__m128);

    #[inline(always)]
    unsafe fn fmadd(a: __m128, b: __m128, c: __m128) -> __m128 {
        #[cfg(target_feature = "fma")]
        {
            _mm_fmadd_ps(a, b, c)
        }
        #[cfg(not(target_feature = "fma"))]
        {
            _mm_add_ps(_mm_mul_ps(a, b), c)
        }
    }

    impl V4 {
        #[inline(always)]
        pub fn zero() -> Self {
            unsafe { V4(_mm_setzero_ps()) }
        }
        #[inline(always)]
        pub fn load(p: &[f32]) -> Self {
            assert!(p.len() >= 4);
            unsafe { V4(_mm_loadu_ps(p.as_ptr())) }
        }
        #[inline(always)]
        pub fn store(self, p: &mut [f32]) {
            assert!(p.len() >= 4);
            unsafe { _mm_storeu_ps(p.as_mut_ptr(), self.0) }
        }
        #[inline(always)]
        pub fn fma_splat(self, col: V4, x: f32) -> Self {
            unsafe { V4(fmadd(col.0, _mm_set1_ps(x), self.0)) }
        }
        #[inline(always)]
        pub fn fma_lane(self, col: V4, h: V4, lane: usize) -> Self {
            unsafe {
                let b = match lane {
                    0 => _mm_shuffle_ps::<0x00>(h.0, h.0),
                    1 => _mm_shuffle_ps::<0x55>(h.0, h.0),
                    2 => _mm_shuffle_ps::<0xAA>(h.0, h.0),
                    _ => _mm_shuffle_ps::<0xFF>(h.0, h.0),
                };
                V4(fmadd(col.0, b, self.0))
            }
        }
        #[inline(always)]
        pub fn fma(self, a: V4, b: V4) -> Self {
            unsafe { V4(fmadd(a.0, b.0, self.0)) }
        }
        #[inline(always)]
        pub fn add(self, o: V4) -> Self {
            unsafe { V4(_mm_add_ps(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn leaky_relu(self) -> Self {
            unsafe {
                let scaled = _mm_mul_ps(self.0, _mm_set1_ps(LEAKY_SLOPE));
                let neg = _mm_cmplt_ps(self.0, _mm_setzero_ps());
                V4(_mm_or_ps(
                    _mm_and_ps(neg, scaled),
                    _mm_andnot_ps(neg, self.0),
                ))
            }
        }
        #[inline(always)]
        pub fn hsum(self) -> f32 {
            let mut tmp = [0.0f32; 4];
            self.store(&mut tmp);
            ((tmp[0] + tmp[1]) + tmp[2]) + tmp[3]
        }
    }
}

#[cfg(any(
    feature = "scalar",
    not(any(target_arch = "aarch64", target_arch = "x86_64"))
))]
mod backend {
    use super::LEAKY_SLOPE;

    /// Fused multiply-add where it is a single instruction, plain
    /// `a*b+c` elsewhere (the libm fallback of `mul_add` is very slow).
    #[inline(always)]
    fn fma(a: f32, b: f32, c: f32) -> f32 {
        #[cfg(any(target_arch = "aarch64", target_feature = "fma"))]
        {
            a.mul_add(b, c)
        }
        #[cfg(not(any(target_arch = "aarch64", target_feature = "fma")))]
        {
            a * b + c
        }
    }

    #[derive(Clone, Copy)]
    pub struct V4([f32; 4]);

    // Index loops mirror the lane-wise SIMD operations on purpose.
    #[allow(clippy::needless_range_loop)]
    impl V4 {
        #[inline(always)]
        pub fn zero() -> Self {
            V4([0.0; 4])
        }
        #[inline(always)]
        pub fn load(p: &[f32]) -> Self {
            V4([p[0], p[1], p[2], p[3]])
        }
        #[inline(always)]
        pub fn store(self, p: &mut [f32]) {
            p[..4].copy_from_slice(&self.0);
        }
        #[inline(always)]
        pub fn fma_splat(self, col: V4, x: f32) -> Self {
            let mut r = self.0;
            for i in 0..4 {
                r[i] = fma(col.0[i], x, r[i]);
            }
            V4(r)
        }
        #[inline(always)]
        pub fn fma_lane(self, col: V4, h: V4, lane: usize) -> Self {
            self.fma_splat(col, h.0[lane])
        }
        #[inline(always)]
        pub fn fma(self, a: V4, b: V4) -> Self {
            let mut r = self.0;
            for i in 0..4 {
                r[i] = fma(a.0[i], b.0[i], r[i]);
            }
            V4(r)
        }
        #[inline(always)]
        pub fn add(self, o: V4) -> Self {
            let mut r = self.0;
            for i in 0..4 {
                r[i] += o.0[i];
            }
            V4(r)
        }
        #[inline(always)]
        pub fn leaky_relu(self) -> Self {
            let mut r = self.0;
            for v in r.iter_mut() {
                if *v < 0.0 {
                    *v *= LEAKY_SLOPE;
                }
            }
            V4(r)
        }
        #[inline(always)]
        pub fn hsum(self) -> f32 {
            ((self.0[0] + self.0[1]) + self.0[2]) + self.0[3]
        }
    }
}

pub use backend::V4;

#[cfg(test)]
mod tests {
    use super::V4;

    #[test]
    fn lane_ops_behave() {
        let a = V4::load(&[1.0, 2.0, 3.0, 4.0]);
        let col = V4::load(&[10.0, 20.0, 30.0, 40.0]);
        let h = V4::load(&[0.5, -1.0, 2.0, 0.0]);
        let mut out = [0.0; 4];
        a.fma_lane(col, h, 1).store(&mut out);
        assert_eq!(out, [-9.0, -18.0, -27.0, -36.0]);
        a.fma_splat(col, 2.0).store(&mut out);
        assert_eq!(out, [21.0, 42.0, 63.0, 84.0]);
        h.leaky_relu().store(&mut out);
        assert_eq!(out, [0.5, -0.01, 2.0, 0.0]);
        assert_eq!(a.hsum(), 10.0);
        a.fma(col, h).store(&mut out);
        assert_eq!(out, [6.0, -18.0, 63.0, 4.0]);
    }
}
