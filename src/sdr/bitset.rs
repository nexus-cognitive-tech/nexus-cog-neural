//! Bit-level SDR helpers.

use super::{Sdr, SDR_WIDTH};

/// Count active bits in an SDR. Equivalent to
/// `sdr.active_count()` but provided as a free function so generic
/// algorithms that only have a `&dyn SdrLike` can use it.
#[must_use]
pub fn active_count(sdr: &Sdr) -> usize {
    sdr.active_count()
}

/// Density in `[0.0, 1.0]` — `active / width`.
#[must_use]
pub fn density(sdr: &Sdr) -> f32 {
    sdr.density()
}

/// Bit-vector statistics used by region-level diagnostics.
#[derive(Debug, Clone, Copy)]
pub struct SdrStats {
    /// Total bit width.
    pub width: usize,
    /// Number of active bits.
    pub active: usize,
    /// Density `active / width`.
    pub density: f32,
    /// `width - active`.
    pub inactive: usize,
}

impl SdrStats {
    /// Compute stats for the given SDR.
    #[must_use]
    pub fn from(sdr: &Sdr) -> Self {
        let active = sdr.active_count();
        Self {
            width: SDR_WIDTH,
            active,
            density: active as f32 / SDR_WIDTH as f32,
            inactive: SDR_WIDTH - active,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_match_active_count() {
        let s = Sdr::from_bits([0, 5, 10]);
        let st = SdrStats::from(&s);
        assert_eq!(st.active, 3);
        assert_eq!(st.inactive, SDR_WIDTH - 3);
        assert!((st.density - 3.0 / SDR_WIDTH as f32).abs() < 1e-9);
    }
}
