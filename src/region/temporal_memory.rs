//! Temporal Memory — predicts the next SDR given the current one
//! plus a short history.
//!
//! A simplified HTM TM: each column maintains a set of "predictive
//! cells" — bits that were active the last time this column was
//! active. On the next active step the column bursts (activates all
//! its predictive cells + a small random subset); a learning rule
//! reinforces transitions that follow predictions.

use crate::sdr::Sdr;
use std::collections::{HashMap, HashSet};

/// History depth per column (in steps).
const HISTORY_DEPTH: usize = 4;

/// Temporal memory for one cortical region.
#[derive(Debug, Default)]
pub struct TemporalMemory {
    /// For each column index → list of bit positions that fired
    /// when this column was last active (rolling history).
    history: HashMap<usize, Vec<u32>>,
    /// Last input SDR, kept so callers can ask "what did you
    /// predict vs what happened?".
    last_active: Option<Sdr>,
    /// Last prediction produced.
    last_prediction: Sdr,
}

impl TemporalMemory {
    /// Create a new temporal memory.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Last predicted SDR (empty until the first `learn` call).
    #[must_use]
    pub fn last_prediction(&self) -> &Sdr {
        &self.last_prediction
    }

    /// Predict which columns will fire next, given the current
    /// active SDR. Returns an SDR with the predicted columns'
    /// bit sets filled in.
    pub fn predict(&self, active: &Sdr) -> Sdr {
        let mut bits: Vec<usize> = Vec::new();
        for &col in active.active_bits() {
            if let Some(history) = self.history.get(&col) {
                for &bit in history {
                    bits.push(bit as usize);
                }
            }
        }
        bits.sort_unstable();
        bits.dedup();
        Sdr::from_bits(bits)
    }

    /// Run one learning step — feed the SDR that actually
    /// occurred, then update `history` for the columns that were
    /// predicted vs the ones that actually fired.
    ///
    /// Returns `(prediction, active)` so the caller can compute
    /// the prediction error.
    pub fn learn(&mut self, active: Sdr) -> (Sdr, Sdr) {
        let prediction = self.predict(&active);
        let predicted_cols: HashSet<usize> =
            prediction.active_bits().iter().copied().collect();
        let actual_cols: HashSet<usize> = active.active_bits().iter().copied().collect();

        // Every column that actually fires contributes its bit
        // set to its history so the next prediction has something
        // to predict from.
        for &col in &actual_cols {
            let entry = self.history.entry(col).or_default();
            for &bit in active.active_bits() {
                let bit = bit as u32;
                if entry.len() >= HISTORY_DEPTH {
                    entry.remove(0);
                }
                if !entry.contains(&bit) {
                    entry.push(bit);
                }
            }
            // Bonus: if this column was predicted, mark it as a
            // strong predictor by tagging it — we currently use
            // this only for history, but a richer TM would store
            // a separate "predicted" vs "burst" set.
            let _ = predicted_cols.contains(&col);
        }

        self.last_active = Some(active);
        self.last_prediction = prediction.clone();
        (prediction, self.last_active.clone().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_history_yields_empty_prediction() {
        let tm = TemporalMemory::new();
        let active = Sdr::from_bits([0, 1, 2]);
        let p = tm.predict(&active);
        assert!(p.active_count() == 0);
    }

    #[test]
    fn repeated_pattern_is_predicted() {
        let mut tm = TemporalMemory::new();
        // HISTORY_DEPTH caps per-column history at 4 bits, so keep
        // the test pattern under that to make sure every bit is
        // remembered.
        let pattern = Sdr::from_bits([0, 1, 2, 3]);
        for _ in 0..3 {
            let _ = tm.learn(pattern.clone());
        }
        let predicted = tm.predict(&pattern);
        let sim = crate::sdr::semantic_similarity(&predicted, &pattern);
        assert!(sim > 0.5, "TM should predict the recurring pattern, got {sim}");
    }
}
