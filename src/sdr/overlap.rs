//! Overlap and semantic-similarity metrics on SDRs.
//!
//! In a true sparse code, raw overlap is already a meaningful
//! similarity measure. The two complementary normalised scores
//! (Jaccard, Tanimoto) are also computed for downstream region-level
//! statistics where sparsity drifts.

use super::Sdr;

/// Raw overlap: |a ∩ b|, the number of bits active in both SDRs.
#[must_use]
pub fn overlap(a: &Sdr, b: &Sdr) -> usize {
    a.active_bits().iter().filter(|i| b.get(**i)).count()
}

/// |a ∩ b|.
#[must_use]
pub fn intersection_size(a: &Sdr, b: &Sdr) -> usize {
    overlap(a, b)
}

/// |a ∪ b|.
#[must_use]
pub fn union_size(a: &Sdr, b: &Sdr) -> usize {
    let mut seen = a.active_bits().to_vec();
    for &i in b.active_bits() {
        if !seen.contains(&i) {
            seen.push(i);
        }
    }
    seen.len()
}

/// Semantic similarity — Numenta's "overlap normalised by geometric
/// mean of active counts". This is the canonical SDR similarity:
///
///   `sim(a, b) = |a ∩ b| / sqrt(|a| · |b|)`
///
/// Both inputs are expected to be sparse (~2%); the result is in
/// `[0.0, 1.0]` with `1.0` only for identical SDRs.
#[must_use]
pub fn semantic_similarity(a: &Sdr, b: &Sdr) -> f32 {
    let n = overlap(a, b) as f32;
    let da = a.active_count() as f32;
    let db = b.active_count() as f32;
    if da == 0.0 || db == 0.0 {
        return 0.0;
    }
    n / (da * db).sqrt()
}

/// Jaccard distance: `1 - |a ∩ b| / |a ∪ b|`. 0 = identical, 1 =
/// disjoint. Useful when sparsity has drifted and overlap alone
/// over-estimates similarity.
#[must_use]
pub fn jaccard_distance(a: &Sdr, b: &Sdr) -> f32 {
    let u = union_size(a, b) as f32;
    if u == 0.0 {
        return 0.0;
    }
    1.0 - overlap(a, b) as f32 / u
}

/// Tanimoto / generalised Jaccard for binary vectors. Equivalent
/// to `jaccard_distance` for SDRs but computed via the standard
/// `a·b / (a² + b² - a·b)` form, which is cheaper when both
/// vectors are pre-populated.
#[must_use]
pub fn tanimoto_distance(a: &Sdr, b: &Sdr) -> f32 {
    let dot = overlap(a, b) as f32;
    let na = a.active_count() as f32;
    let nb = b.active_count() as f32;
    let denom = na + nb - dot;
    if denom == 0.0 {
        return 0.0;
    }
    1.0 - dot / denom
}

/// Distance metric used by the population code in the region layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PopulationDistance {
    /// Raw overlap divided by `min(|a|, |b|)`. Bounded in `[0, 1]`,
    /// `0` = perfect match.
    MinNormOverlap,
    /// `1 - semantic_similarity`. Symmetric.
    SemanticComplement,
    /// Jaccard distance.
    Jaccard,
    /// Hamming fraction `hamming / width`. Bounded in `[0, 1]`.
    Hamming,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlap_identical() {
        let a = Sdr::from_bits([1, 5, 9]);
        let b = Sdr::from_bits([1, 5, 9]);
        assert_eq!(overlap(&a, &b), 3);
        assert!((semantic_similarity(&a, &b) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn overlap_disjoint() {
        let a = Sdr::from_bits([0, 1, 2]);
        let b = Sdr::from_bits([10, 11, 12]);
        assert_eq!(overlap(&a, &b), 0);
        assert_eq!(semantic_similarity(&a, &b), 0.0);
        assert!((jaccard_distance(&a, &b) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn semantic_bounded_in_unit_interval() {
        let a = Sdr::from_bits([0, 1, 2, 3, 4]);
        let b = Sdr::from_bits([3, 4, 5, 6, 7]);
        let s = semantic_similarity(&a, &b);
        assert!((0.0..=1.0).contains(&s), "{s}");
    }

    #[test]
    fn zero_sdr_similarity() {
        let a = Sdr::empty();
        let b = Sdr::from_bits([0, 1, 2]);
        assert_eq!(semantic_similarity(&a, &b), 0.0);
    }
}
