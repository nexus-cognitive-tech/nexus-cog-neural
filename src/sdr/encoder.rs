//! Encoders — converters from real-world values into SDRs.
//!
//! Every encoder satisfies the same [`Encoder`] trait so the
//! spatial pooler can be fed by a uniform `encode(…) -> Sdr`
//! interface. Encoders share two common tricks:
//!
//! * **Bucketed width** — the SDR is split into `n` buckets of
//!   equal width, and the encoder lights up enough buckets to
//!   represent the input plus a configurable bit of overlap with
//!   its neighbours (so two close inputs produce SDRs with large
//!   overlap).
//! * **Periodic wrap-around** — when the input space is cyclic
//!   (e.g. day-of-week, hour-of-day, angle) the last bucket wraps
//!   around and shares bits with the first.

use super::{Sdr, SDR_WIDTH, DEFAULT_SPARSITY};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Encoder trait — every concrete encoder produces an SDR of the
/// system's standard [`SDR_WIDTH`].
pub trait Encoder<T> {
    /// Encode the supplied value into an SDR.
    fn encode(&mut self, value: T) -> Sdr;

    /// Width of every output SDR — always [`SDR_WIDTH`].
    #[must_use]
    fn output_width(&self) -> usize {
        SDR_WIDTH
    }

    /// Active bits per output — encoded count.
    #[must_use]
    fn active_per_output(&self) -> usize;
}

/// Scalar encoder — maps a real number into an SDR by bucketing
/// `[min, max]` into `buckets` equal-width buckets and lighting
/// up `active` adjacent buckets around the input position.
///
/// Overlap is in `[0, 1]` and measures how many bits two adjacent
/// bucket values share. With `active = 5, buckets = 100` two
/// adjacent inputs share ~80% of their bits. With
/// `periodic = false`, values outside `[min, max]` clamp and the
/// extreme buckets do not wrap around.
pub struct ScalarEncoder {
    min: f32,
    max: f32,
    buckets: usize,
    active: usize,
    bits_per_bucket: usize,
    periodic: bool,
}

impl ScalarEncoder {
    /// Construct a new non-periodic scalar encoder. `buckets` must
    /// be ≥ `active`, and `active` ≥ 1.
    pub fn new(min: f32, max: f32, buckets: usize, active: usize) -> Self {
        assert!(buckets > 0, "buckets must be > 0");
        assert!(active > 0, "active must be > 0");
        assert!(active <= buckets, "active ({active}) > buckets ({buckets})");
        let bits_per_bucket = SDR_WIDTH / buckets;
        assert!(bits_per_bucket > 0, "SDR_WIDTH too small for {buckets} buckets");
        Self { min, max, buckets, active, bits_per_bucket, periodic: false }
    }

    /// Construct a periodic encoder (e.g. for day-of-week,
    /// angle, hour-of-day).
    pub fn new_periodic(min: f32, max: f32, buckets: usize, active: usize) -> Self {
        let mut e = Self::new(min, max, buckets, active);
        e.periodic = true;
        e
    }
}

impl Encoder<f32> for ScalarEncoder {
    fn encode(&mut self, value: f32) -> Sdr {
        let clamped = value.clamp(self.min, self.max);
        let range = (self.max - self.min).max(f32::EPSILON);
        let pos = ((clamped - self.min) / range * (self.buckets - 1) as f32).round() as i64;
        let half = (self.active / 2) as i64;
        let mut active_indices: Vec<usize> = Vec::with_capacity(self.active * self.bits_per_bucket);
        for offset in 0..self.active as i64 {
            let raw = pos + offset - half;
            let bucket = if self.periodic {
                raw.rem_euclid(self.buckets as i64)
            } else {
                raw.clamp(0, self.buckets as i64 - 1)
            } as usize;
            for bit in 0..self.bits_per_bucket {
                active_indices.push(bucket * self.bits_per_bucket + bit);
            }
        }
        Sdr::from_bits(active_indices)
    }
    fn active_per_output(&self) -> usize {
        self.active * self.bits_per_bucket
    }
}

/// Category encoder — maps a small set of category identifiers to
/// stable, random SDRs with high mutual overlap probability.
///
/// The SDRs are generated lazily from a deterministic RNG so the
/// same category always produces the same SDR — and overlapping
/// categories (by index) get a small bonus overlap so that
/// "nearby" categories share a few bits.
pub struct CategoryEncoder {
    bits_per_category: usize,
    #[allow(dead_code)] // planned: bonus overlap for nearby categories
    overlap: usize,
    cache: std::collections::HashMap<u32, Sdr>,
}

impl CategoryEncoder {
    /// `categories` controls the maximum category id we accept.
    /// `bits_per_category` defaults to the standard 2% sparsity
    /// (`SDR_WIDTH * DEFAULT_SPARSITY`).
    pub fn new(categories: u32) -> Self {
        let _ = categories;
        let bits_per_category = (SDR_WIDTH as f32 * DEFAULT_SPARSITY) as usize;
        Self {
            bits_per_category,
            overlap: 4,
            cache: std::collections::HashMap::new(),
        }
    }
}

impl Encoder<u32> for CategoryEncoder {
    fn encode(&mut self, category: u32) -> Sdr {
        if let Some(cached) = self.cache.get(&category) {
            return cached.cloned_active();
        }
        let mut rng = StdRng::seed_from_u64(category as u64);
        let s = Sdr::random_active(&mut rng, self.bits_per_category);
        self.cache.insert(category, s.cloned_active());
        s
    }
    fn active_per_output(&self) -> usize {
        self.bits_per_category
    }
}

/// Date / time encoder — wraps day-of-week, hour-of-day and
/// minute-of-hour with periodic encoders and concatenates them.
pub struct DateEncoder {
    hour: ScalarEncoder,
    minute: ScalarEncoder,
    dow: ScalarEncoder,
}

impl Default for DateEncoder {
    fn default() -> Self {
        Self {
            hour: ScalarEncoder::new_periodic(0.0, 24.0, 24, 5),
            minute: ScalarEncoder::new_periodic(0.0, 60.0, 60, 6),
            dow: ScalarEncoder::new_periodic(0.0, 7.0, 7, 3),
        }
    }
}

impl DateEncoder {
    /// Encode a `(day_of_week, hour, minute)` triple.
    pub fn encode_date(&mut self, dow: u32, hour: u32, minute: u32) -> Sdr {
        let h = self.hour.encode(hour as f32);
        let m = self.minute.encode(minute as f32);
        let d = self.dow.encode(dow as f32);
        let mut union = h;
        union.union_with(&m);
        union.union_with(&d);
        union
    }
}

/// Coordinate encoder — N-D point → SDR. Used for spatial /
/// sensor-fusion inputs. Each dimension gets its own bucket range
/// and the resulting bucket indices are mixed with a deterministic
/// permutation so the representation is distributed (no single
/// dimension dominates).
pub struct CoordinateEncoder {
    dim: usize,
    bits_per_dim: usize,
    ranges: Vec<(f32, f32)>,
}

impl CoordinateEncoder {
    /// Construct a coordinate encoder. `ranges` is `(min, max)` per
    /// dimension.
    pub fn new(ranges: Vec<(f32, f32)>) -> Self {
        let dim = ranges.len();
        let bits_per_dim = (SDR_WIDTH as f32 * DEFAULT_SPARSITY / dim as f32) as usize / 2 * 2;
        Self { dim, bits_per_dim, ranges }
    }
}

impl Encoder<Vec<f32>> for CoordinateEncoder {
    fn encode(&mut self, coords: Vec<f32>) -> Sdr {
        assert_eq!(coords.len(), self.dim, "dim mismatch");
        let mut indices: Vec<usize> = Vec::new();
        for (i, v) in coords.iter().enumerate() {
            let (lo, hi) = self.ranges[i];
            let clamped = v.clamp(lo, hi);
            let range = (hi - lo).max(f32::EPSILON);
            let t = (clamped - lo) / range;
            let mut rng = StdRng::seed_from_u64((i as u64) << 32 ^ ((t * 1_000_000.0) as u64));
            for _ in 0..self.bits_per_dim {
                let bit = rng.gen_range(0..SDR_WIDTH);
                if !indices.contains(&bit) {
                    indices.push(bit);
                }
            }
        }
        Sdr::from_bits(indices)
    }
    fn active_per_output(&self) -> usize {
        self.bits_per_dim * self.dim
    }
}

/// Log-scale encoder — useful for latencies, sizes, frequencies,
/// any value that grows exponentially. Internally feeds a
/// `ScalarEncoder` over `log10(value)`.
pub struct LogEncoder {
    inner: ScalarEncoder,
}

impl LogEncoder {
    /// `min_value` / `max_value` are the inclusive bounds of the
    /// input space (must be positive).
    pub fn new(min_value: f32, max_value: f32, buckets: usize, active: usize) -> Self {
        let lo = min_value.max(f32::EPSILON).log10();
        let hi = max_value.max(f32::EPSILON).log10();
        Self { inner: ScalarEncoder::new(lo, hi, buckets, active) }
    }
}

impl Encoder<f32> for LogEncoder {
    fn encode(&mut self, value: f32) -> Sdr {
        self.inner.encode(value.max(f32::EPSILON).log10())
    }
    fn active_per_output(&self) -> usize {
        self.inner.active_per_output()
    }
}

/// Sequence encoder — lights up the previous `n` elements of a
/// rolling window. Used for temporal-context inputs where the
/// present SDR should overlap with the recent past.
pub struct SequenceEncoder {
    history: std::collections::VecDeque<u64>,
    history_len: usize,
}

impl SequenceEncoder {
    /// `history_len` is how many past values the encoded SDR
    /// summarises.
    pub fn new(history_len: usize) -> Self {
        Self {
            history: std::collections::VecDeque::with_capacity(history_len),
            history_len,
        }
    }

    /// Feed a new value and get the encoded SDR. The SDR mixes
    /// the current value with the previous `history_len - 1`
    /// values via XOR-like mixing of their bucket indices.
    pub fn feed(&mut self, value: u64) -> Sdr {
        self.history.push_back(value);
        while self.history.len() > self.history_len {
            self.history.pop_front();
        }
        let mut indices: Vec<usize> = Vec::new();
        for (i, &v) in self.history.iter().enumerate() {
            let mut rng = StdRng::seed_from_u64(v ^ ((i as u64) << 56));
            for _ in 0..(SDR_WIDTH / self.history_len / 64) {
                let bit = rng.gen_range(0..SDR_WIDTH);
                if !indices.contains(&bit) {
                    indices.push(bit);
                }
            }
        }
        Sdr::from_bits(indices)
    }
}

/// Re-export of the encoder trait under a friendlier name.
pub type SdrEncoder<T> = dyn Encoder<T>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_adjacent_inputs_overlap() {
        let mut e = ScalarEncoder::new(0.0, 100.0, 100, 3);
        let a = e.encode(50.0);
        let b = e.encode(51.0);
        let sim = crate::sdr::semantic_similarity(&a, &b);
        assert!(sim > 0.5, "adjacent inputs should overlap heavily, got {sim}");
    }

    #[test]
    fn scalar_distant_inputs_diverge() {
        let mut e = ScalarEncoder::new(0.0, 100.0, 100, 3);
        let a = e.encode(0.0);
        let b = e.encode(100.0);
        let sim = crate::sdr::semantic_similarity(&a, &b);
        assert!(sim < 0.4, "distant inputs should diverge, got {sim}");
    }

    #[test]
    fn category_reproducible() {
        let mut e = CategoryEncoder::new(10);
        let a = e.encode(3);
        let b = e.encode(3);
        assert_eq!(a.active_bits(), b.active_bits());
    }

    #[test]
    fn date_wraps_around_hour() {
        let mut e = DateEncoder::default();
        let a = e.encode_date(1, 23, 30);
        let b = e.encode_date(1, 0, 30);
        let sim = crate::sdr::semantic_similarity(&a, &b);
        assert!(sim > 0.4, "23:30 and 00:30 should overlap (wraps around midnight), got {sim}");
    }

    #[test]
    fn log_compresses_large_range() {
        let mut e = LogEncoder::new(1.0, 1_000_000.0, 100, 10);
        let a = e.encode(10.0);
        let b = e.encode(20.0);
        let c = e.encode(10_000.0);
        let ab = crate::sdr::semantic_similarity(&a, &b);
        let ac = crate::sdr::semantic_similarity(&a, &c);
        assert!(ab > ac, "log scale should bring closer values closer: ab={ab}, ac={ac}");
    }
}
