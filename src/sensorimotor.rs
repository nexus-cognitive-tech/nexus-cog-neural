//! Sensorimotor loop — closed cycle between perception and action.
//!
//! The classical perception-action cycle is a control loop:
//!
//! 1. **Sense** — thalamic / sensory input enters the cortex.
//! 2. **Predict** — the cortex's top-down L1 context carries
//!    predictions about what should be sensed.
//! 3. **Compare** — a comparator computes the prediction error
//!    (PE = actual − predicted). PE is reported as valence to
//!    the neuromodulator panel.
//! 4. **Act** — the basal ganglia reads the cortex's L5 outputs
//!    and selects an action.
//! 5. **Effect** — the action closes the loop: it either moves
//!    sensors (active perception) or sends motor commands that
//!    change the world which is then sensed.
//!
//! This module owns the comparator and the loop counter; the rest
//! of the perception / action machinery is wired by
//! [`crate::cortex::Cortex`].

use crate::spike::SpikeTrain;
use serde::{Deserialize, Serialize};

/// One step of the perception → action → perception cycle.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoopPhase {
    /// Reading sensory input.
    #[default]
    Sensing,
    /// Cortex is generating predictions.
    Predicting,
    /// Comparator is computing prediction error.
    Comparing,
    /// Basal ganglia is selecting an action.
    Acting,
    /// Action is being executed; effect will be sensed next tick.
    Effecting,
    /// NREM/REM sleep — offline consolidation, no sensory input.
    /// Loops back to `Sensing` when the sleep cycle ends.
    Sleeping,
}

/// Result of one comparator step — the prediction error signal
/// that drives learning and neuromodulator updates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionError {
    /// Sum of absolute differences between predicted and actual
    /// spike counts across all columns.
    pub absolute: f32,
    /// Per-layer error breakdown (L1..L6 by canonical order; we
    /// report only the layers that exist).
    pub per_layer: Vec<(String, f32)>,
    /// Surprisal in bits — `−log₂(P(actual | predicted))`.
    pub surprisal_bits: f32,
}

/// Sensorimotor loop controller.
pub struct SensorimotorLoop {
    /// Width of the comparator's working buffer (one entry per
    /// tick). Ring buffer.
    pub history: Vec<PredictionError>,
    /// Phase we're currently in.
    pub phase: LoopPhase,
    /// Total ticks elapsed.
    pub ticks: u64,
}

impl SensorimotorLoop {
    /// Create a new sensorimotor loop in the `Sensing` phase.
    pub fn new() -> Self {
        Self {
            history: Vec::with_capacity(1024),
            phase: LoopPhase::Sensing,
            ticks: 0,
        }
    }

    /// Compute the prediction error between the cortex's predicted
    /// spike train and the actually-observed one, per layer.
    pub fn compute_error(
        &mut self,
        predicted: &[SpikeTrain],
        actual: &[SpikeTrain],
    ) -> PredictionError {
        assert_eq!(predicted.len(), actual.len(), "layer count mismatch");
        let mut absolute = 0.0_f32;
        let mut per_layer = Vec::with_capacity(predicted.len());
        for (i, (p, a)) in predicted.iter().zip(actual.iter()).enumerate() {
            let d = p.distance(a);
            absolute += d;
            per_layer.push((format!("L{}", i + 1), d));
        }
        let layer_count = predicted.len().max(1) as f32;
        let normalised = absolute / layer_count;
        // Surprisal in bits: clip to [0, 1] then take -log2(1-x).
        let surprisal_bits = -(1.0 - normalised.clamp(0.0, 0.999)).log2();
        let err = PredictionError {
            absolute,
            per_layer,
            surprisal_bits,
        };
        self.history.push(err.clone());
        if self.history.len() > 1024 {
            self.history.remove(0);
        }
        self.ticks += 1;
        self.phase = self.next_phase(self.phase);
        err
    }

    fn next_phase(&self, current: LoopPhase) -> LoopPhase {
        match current {
            LoopPhase::Sensing => LoopPhase::Predicting,
            LoopPhase::Predicting => LoopPhase::Comparing,
            LoopPhase::Comparing => LoopPhase::Acting,
            LoopPhase::Acting => LoopPhase::Effecting,
            LoopPhase::Effecting => LoopPhase::Sensing,
            LoopPhase::Sleeping => LoopPhase::Sensing,
        }
    }

    /// Recent mean prediction error over the last `window` ticks.
    pub fn recent_error(&self, window: usize) -> f32 {
        let n = window.min(self.history.len());
        if n == 0 {
            return 0.0;
        }
        let slice = &self.history[self.history.len() - n..];
        slice.iter().map(|e| e.absolute).sum::<f32>() / n as f32
    }
}

impl Default for SensorimotorLoop {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_prediction_has_zero_absolute_error() {
        let mut sm = SensorimotorLoop::new();
        let s = SpikeTrain::silent();
        let err = sm.compute_error(&[s.clone(), s.clone()], &[s.clone(), s.clone()]);
        assert_eq!(err.absolute, 0.0);
        assert_eq!(err.surprisal_bits, 0.0);
    }

    #[test]
    fn perfect_mismatch_is_one_per_layer() {
        let mut sm = SensorimotorLoop::new();
        let silent = SpikeTrain::silent();
        let mut burst = SpikeTrain::silent();
        burst.spike_at(0);
        let err = sm.compute_error(&[silent.clone()], &[burst]);
        assert!(err.absolute > 0.0);
    }

    #[test]
    fn phase_cycles_through_loop() {
        let mut sm = SensorimotorLoop::new();
        let s = SpikeTrain::silent();
        let start = sm.phase;
        // After 5 ticks the loop has visited Predicting →
        // Comparing → Acting → Effecting → Sensing and is back to
        // start.
        for _ in 0..4 {
            sm.compute_error(&[s.clone()], &[s.clone()]);
        }
        assert_ne!(sm.phase, start);
        assert_eq!(sm.phase, LoopPhase::Effecting);
    }
}
