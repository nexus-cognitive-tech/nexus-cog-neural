//! Global Workspace — Baars-style consciousness analog.
//!
//! Every tick, every cortical region broadcasts its current SDR
//! to a shared workspace. Regions compete to be part of the
//! winning coalition; the winner's union becomes the conscious
//! content that other regions receive on the next tick.

use crate::sdr::Sdr;
use serde::{Deserialize, Serialize};

/// A coalition of regions that have contributed to the current
/// global broadcast.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Coalition {
    /// Source labels (region names).
    pub members: Vec<String>,
    /// Union of the contributing SDRs.
    pub union: Sdr,
}

impl Default for Coalition {
    fn default() -> Self {
        Self { members: Vec::new(), union: Sdr::empty() }
    }
}

/// Per-tick global workspace broadcast.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GlobalWorkspace {
    /// Most recent broadcast.
    pub current: Coalition,
    /// History of broadcasts (kept for replay / Studio viz).
    pub history: Vec<Coalition>,
}

impl GlobalWorkspace {
    /// New empty workspace.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Hold a competition: regions with SDRs whose overlap to
    /// `salience_seed` exceeds `threshold` form the winning
    /// coalition.
    pub fn compete(
        &mut self,
        candidates: &[(String, Sdr)],
        salience_seed: &Sdr,
        threshold: f32,
    ) -> Coalition {
        let mut members = Vec::new();
        let mut union = Sdr::empty();
        for (name, sdr) in candidates {
            let sim = crate::sdr::semantic_similarity(sdr, salience_seed);
            if sim >= threshold {
                members.push(name.clone());
                union.union_with(sdr);
            }
        }
        let coalition = Coalition { members, union };
        self.current = coalition.clone();
        self.history.push(coalition.clone());
        if self.history.len() > 1024 {
            self.history.remove(0);
        }
        coalition
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coalition_filters_low_overlap() {
        let mut g = GlobalWorkspace::new();
        let seed = Sdr::from_bits([0, 1, 2]);
        let close = seed.clone();
        let far = Sdr::from_bits([1000, 1001, 1002]);
        let c = g.compete(&[("close".into(), close), ("far".into(), far)], &seed, 0.5);
        assert_eq!(c.members, vec!["close".to_string()]);
    }

    #[test]
    fn history_capped() {
        let mut g = GlobalWorkspace::new();
        let seed = Sdr::empty();
        for _ in 0..1500 {
            g.compete(&[("x".into(), Sdr::empty())], &seed, 0.0);
        }
        assert!(g.history.len() <= 1024);
    }
}
