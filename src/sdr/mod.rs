//! Sparse Distributed Representations (SDR).
//!
//! An SDR is a fixed-width bit vector where ~2% of the bits are
//! active. The two foundational properties are:
//!
//! 1. **Similarity preservation** — semantically similar inputs map
//!    to SDRs with large overlap. Random inputs map to SDRs with
//!    near-zero overlap.
//! 2. **Robustness** — flipping any subset of bits up to ~20% keeps
//!    overlap with the original SDR above the noise floor.
//!
//! Memory format: `BitVec<u8, Lsb0>` (a packed bit array). The
//! `active_bits()` index set is also materialised on demand because
//! every downstream operation wants it.

mod bitset;
mod encoder;
mod overlap;

pub use bitset::{active_count, density, SdrStats};
pub use encoder::{
    CategoryEncoder, CoordinateEncoder, DateEncoder, Encoder, LogEncoder, ScalarEncoder,
    SdrEncoder, SequenceEncoder,
};
pub use overlap::{
    intersection_size, jaccard_distance, overlap, semantic_similarity, tanimoto_distance,
    union_size, PopulationDistance,
};

use bitvec::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Width of every SDR in the system. 2048 is Numenta's default and
/// gives enough capacity for ~2% sparsity (≈40 active bits) while
/// keeping all bitwise operations trivial on a modern CPU.
pub const SDR_WIDTH: usize = 2048;

/// Default target sparsity for newly-minted SDRs.
pub const DEFAULT_SPARSITY: f32 = 0.02;

/// Serialised representation of an SDR.
///
/// We persist SDRs as their set of active bit indices rather than
/// the full 2048-bit vector — at ~2% sparsity the active set is
/// ~40 integers, versus 256 bytes for the dense representation.
pub type SdrWire = Vec<usize>;

impl Serialize for Sdr {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.cached_active.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Sdr {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bits: Vec<usize> = Vec::<usize>::deserialize(deserializer)?;
        Ok(Sdr::from_bits(bits))
    }
}

/// A fixed-width sparse distributed representation.
///
/// Invariants:
/// * length is always [`SDR_WIDTH`] (2048)
/// * the bit vector may hold any subset of `1`s in `[0, SDR_WIDTH)`
/// * `active_bits()` and the bitvec are kept in sync — bitvec is
///   the source of truth, `active_bits` is a cached `Vec<usize>`
pub struct Sdr {
    bits: BitVec<u8, Lsb0>,
    cached_active: Vec<usize>,
}

impl Clone for Sdr {
    fn clone(&self) -> Self {
        Self { bits: self.bits.clone(), cached_active: self.cached_active.clone() }
    }
}

impl fmt::Debug for Sdr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Sdr")
            .field("width", &self.bits.len())
            .field("active", &self.cached_active.len())
            .field("bits", &format!("[{} active]", self.cached_active.len()))
            .finish()
    }
}

impl PartialEq for Sdr {
    fn eq(&self, other: &Self) -> bool {
        self.bits == other.bits
    }
}

impl Eq for Sdr {}

impl Default for Sdr {
    fn default() -> Self {
        Self::empty()
    }
}

impl Sdr {
    /// Allocate an empty SDR with all bits zeroed.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            bits: bitvec![u8, Lsb0; 0; SDR_WIDTH],
            cached_active: Vec::new(),
        }
    }

    /// Construct an SDR from a known set of active bits. Bits outside
    /// `[0, SDR_WIDTH)` are silently dropped.
    #[must_use]
    pub fn from_bits<I: IntoIterator<Item = usize>>(bits: I) -> Self {
        let mut s = Self::empty();
        for b in bits {
            if b < SDR_WIDTH {
                s.bits.set(b, true);
            }
        }
        s.rebuild_cache();
        s
    }

    /// Construct an SDR with `count` active bits, drawn uniformly at
    /// random. Use [`Sdr::from_bits`] when reproducibility matters.
    #[must_use]
    pub fn random_active<R: rand::Rng + ?Sized>(rng: &mut R, count: usize) -> Self {
        let count = count.min(SDR_WIDTH);
        let mut indices: Vec<usize> = (0..SDR_WIDTH).collect();
        // PartialFisherYatesShuffle: only shuffle the first `count` slots.
        for i in 0..count {
            let j = rng.gen_range(i..SDR_WIDTH);
            indices.swap(i, j);
        }
        indices.truncate(count);
        Self::from_bits(indices)
    }

    /// Construct an SDR with ~`density` × [`SDR_WIDTH`] active bits
    /// (rounded to the nearest integer). `density` is clamped to
    /// `[0.0, 1.0]`.
    #[must_use]
    pub fn random_with_density<R: rand::Rng + ?Sized>(rng: &mut R, density: f32) -> Self {
        let d = density.clamp(0.0, 1.0);
        let n = (d * SDR_WIDTH as f32).round() as usize;
        Self::random_active(rng, n)
    }

    /// All-zero SDR.
    pub fn zero() -> Self {
        Self::empty()
    }

    /// Width of the SDR — always [`SDR_WIDTH`].
    #[must_use]
    pub const fn width(&self) -> usize {
        SDR_WIDTH
    }

    /// Number of active bits.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.cached_active.len()
    }

    /// Active bit indices, sorted ascending.
    #[must_use]
    pub fn active_bits(&self) -> &[usize] {
        &self.cached_active
    }

    /// Density in `[0, 1]` — `active_count / width`.
    #[must_use]
    pub fn density(&self) -> f32 {
        self.active_count() as f32 / SDR_WIDTH as f32
    }

    /// Read a single bit.
    #[must_use]
    pub fn get(&self, index: usize) -> bool {
        self.bits.get(index).map(|b| *b).unwrap_or(false)
    }

    /// Set a single bit; marks the SDR as dense (caller should
    /// call `rebuild_cache` if mutating many bits).
    pub fn set(&mut self, index: usize, value: bool) {
        if index >= SDR_WIDTH {
            return;
        }
        let prev = self.bits.get(index).map(|b| *b).unwrap_or(false);
        if prev != value {
            self.bits.set(index, value);
            // Cache invalidation is the caller's responsibility for
            // hot loops, but the cheap path is good enough here.
            self.rebuild_cache();
        }
    }

    /// Re-derive the cached active-bits vector from the underlying
    /// bitvec. O(N) over the SDR width.
    pub fn rebuild_cache(&mut self) {
        self.cached_active.clear();
        for (i, b) in self.bits.iter().enumerate() {
            if *b {
                self.cached_active.push(i);
            }
        }
    }

    /// Logical OR with another SDR. Bits outside the intersection
    /// become active; existing active bits stay active.
    pub fn union_with(&mut self, other: &Self) {
        for &i in &other.cached_active {
            self.bits.set(i, true);
        }
        self.rebuild_cache();
    }

    /// Logical AND with another SDR. Only bits active in both stay
    /// active.
    pub fn intersection_with(&mut self, other: &Self) {
        for &i in &self.cached_active {
            if !other.bits.get(i).map(|b| *b).unwrap_or(false) {
                self.bits.set(i, false);
            }
        }
        self.rebuild_cache();
    }

    /// Random-resample `keep` of the active bits (in place). Used by
    /// the spatial pooler to keep SDRs from converging to a single
    /// attractor.
    pub fn sparsify_to<R: rand::Rng + ?Sized>(&mut self, rng: &mut R, keep: usize) {
        if self.cached_active.len() <= keep {
            return;
        }
        // Partial Fisher-Yates, then truncate.
        let n = self.cached_active.len();
        for i in 0..keep {
            let j = rng.gen_range(i..n);
            self.cached_active.swap(i, j);
        }
        let keep_indices: std::collections::HashSet<usize> =
            self.cached_active.iter().take(keep).copied().collect();
        for &i in &self.cached_active {
            if !keep_indices.contains(&i) {
                self.bits.set(i, false);
            }
        }
        self.cached_active.truncate(keep);
        self.cached_active.sort_unstable();
    }

    /// An SDR with the same active bits — needed because some
    /// downstream APIs take owned SDRs.
    #[must_use]
    pub fn cloned_active(&self) -> Self {
        Self::from_bits(self.cached_active.iter().copied())
    }

    /// Hamming distance to another SDR.
    #[must_use]
    pub fn hamming(&self, other: &Self) -> usize {
        self.bits.iter().zip(other.bits.iter()).filter(|(a, b)| a != b).count()
    }
}

/// An SDR plus a timestamp + provenance. Used by the hippocampus to
/// record episodes and by the global workspace to label coalitions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LabelledSdr {
    /// The actual representation.
    pub sdr: Sdr,
    /// When this SDR was produced (unix seconds).
    pub timestamp: i64,
    /// Free-form source label, e.g. `"thalamus.gate"` or
    /// `"region[V1].output"`.
    pub source: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn empty_has_no_active_bits() {
        let s = Sdr::empty();
        assert_eq!(s.active_count(), 0);
        assert_eq!(s.density(), 0.0);
        assert_eq!(s.width(), SDR_WIDTH);
    }

    #[test]
    fn from_bits_respects_width() {
        let s = Sdr::from_bits([0, 1, 100, SDR_WIDTH + 50]);
        assert_eq!(s.active_count(), 3);
        assert!(s.get(0) && s.get(1) && s.get(100));
        assert!(!s.get(SDR_WIDTH + 50));
    }

    #[test]
    fn random_active_count_matches() {
        let mut rng = StdRng::seed_from_u64(42);
        let s = Sdr::random_active(&mut rng, 40);
        assert_eq!(s.active_count(), 40);
        let active = s.active_bits().to_vec();
        let mut sorted = active.clone();
        sorted.sort_unstable();
        assert_eq!(active, sorted, "active bits must be sorted ascending");
    }

    #[test]
    fn union_and_intersection() {
        let a = Sdr::from_bits([0, 5, 10]);
        let b = Sdr::from_bits([5, 10, 20]);
        let mut u = a.clone();
        u.union_with(&b);
        assert_eq!(u.active_count(), 4);
        let mut i = a.clone();
        i.intersection_with(&b);
        assert_eq!(i.active_bits(), &[5, 10]);
    }

    #[test]
    fn sparsify_reduces_to_keep() {
        let mut rng = StdRng::seed_from_u64(1);
        let mut s = Sdr::random_active(&mut rng, 100);
        s.sparsify_to(&mut rng, 10);
        assert_eq!(s.active_count(), 10);
    }

    #[test]
    fn hamming_distance() {
        let a = Sdr::from_bits([0, 1, 2, 3]);
        let b = Sdr::from_bits([0, 1, 2, 3]);
        assert_eq!(a.hamming(&b), 0);
        let c = Sdr::from_bits([0, 1, 2, 4]);
        assert_eq!(a.hamming(&c), 2);
    }
}
