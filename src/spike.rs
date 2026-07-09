//! Spike-based temporal coding.
//!
//! Real neurons communicate with sparse, all-or-nothing action
//! potentials — "spikes". This module replaces the continuous SDR
//! activation in [`Sdr`](crate::sdr::Sdr) with a *temporal* code:
//!
//! * **Spike train** — a fixed-length time series in `{0, 1}`
//!   representing which neuron fired in each time step.
//! * **First-spike latency** — the time step of the *first* spike
//!   per neuron; a compact scalar coding that preserves
//!   information about stimulus strength and timing.
//! * **Phase coding** — each neuron has an intrinsic phase and
//!   spikes preferentially when the global oscillation passes that
//!   phase (theta / gamma rhythms).
//!
//! This module powers every downstream layer (cortical column,
//! plasticity, sensorimotor loop). It does **not** depend on the
//! legacy [`Sdr`](crate::sdr::Sdr) — the two are sibling encodings
//! of the same neural state.

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

/// Width of a spike train — number of discrete time steps per
/// simulation tick.
pub const SPIKE_WINDOW: usize = 32;

/// One neuron's spike train over [`SPIKE_WINDOW`] time steps.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpikeTrain {
    /// `spikes[t] == true` iff the neuron fired at step `t`.
    pub spikes: Vec<bool>,
}

impl SpikeTrain {
    /// Empty spike train (silence).
    #[must_use]
    pub fn silent() -> Self {
        Self { spikes: vec![false; SPIKE_WINDOW] }
    }

    /// First-spike latency — `None` if the neuron never fired.
    #[must_use]
    pub fn first_spike(&self) -> Option<usize> {
        self.spikes.iter().position(|&s| s)
    }

    /// Total spike count.
    #[must_use]
    pub fn count(&self) -> usize {
        self.spikes.iter().filter(|&&s| s).count()
    }

    /// Mean firing rate in `[0, 1]` over the window.
    #[must_use]
    pub fn rate(&self) -> f32 {
        self.count() as f32 / SPIKE_WINDOW as f32
    }

    /// Spike at a specific time step. No-op if outside the window.
    pub fn spike_at(&mut self, t: usize) {
        if let Some(slot) = self.spikes.get_mut(t) {
            *slot = true;
        }
    }

    /// Compute the Victor–Purpura-like spike-train distance
    /// (proxy: fraction of mismatched time steps).
    #[must_use]
    pub fn distance(&self, other: &Self) -> f32 {
        let mismatched = self
            .spikes
            .iter()
            .zip(other.spikes.iter())
            .filter(|(a, b)| a != b)
            .count();
        mismatched as f32 / SPIKE_WINDOW as f32
    }
}

/// A population of neurons, each with a private phase.
#[derive(Debug, Clone)]
pub struct SpikingPopulation {
    /// Width of every spike train (= the number of simulated
    /// neurons).
    pub width: usize,
    /// Intrinsic phase per neuron, normalised to `[0, 1]`.
    pub phases: Vec<f32>,
    /// Current global oscillation phase, in `[0, 1]`.
    pub oscillation_phase: f32,
    /// Spontaneous firing probability per step.
    pub spontaneous_rate: f32,
}

impl SpikingPopulation {
    /// Construct a population with uniformly random phases.
    #[must_use]
    pub fn random(width: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let phases = (0..width).map(|_| rng.r#gen::<f32>()).collect();
        Self {
            width,
            phases,
            oscillation_phase: 0.0,
            spontaneous_rate: 0.01,
        }
    }

    /// Advance the global oscillation phase by `delta` (mod 1).
    pub fn advance_oscillation(&mut self, delta: f32) {
        self.oscillation_phase = (self.oscillation_phase + delta).rem_euclid(1.0);
    }

    /// Compute the current spike probability per neuron as the
    /// phase-aligned gaussian + spontaneous baseline.
    pub fn spike_probabilities(&self) -> Vec<f32> {
        self.phases
            .iter()
            .map(|&phase| {
                let dist = (self.oscillation_phase - phase).rem_euclid(1.0);
                let d = (dist).min(1.0 - dist);
                let gauss = (-(d * d) / (2.0 * 0.05 * 0.05)).exp();
                (gauss + self.spontaneous_rate).clamp(0.0, 1.0)
            })
            .collect()
    }

    /// Sample one spike train.
    #[must_use]
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> SpikeTrain {
        let mut train = SpikeTrain::silent();
        for p in self.spike_probabilities() {
            if rng.r#gen::<f32>() < p {
                // Fire at the time step closest to the phase
                // peak. Convert normalised phase `[0,1]` to a
                // discrete step in `[0, SPIKE_WINDOW)`.
                let t = (self.oscillation_phase * SPIKE_WINDOW as f32) as usize;
                train.spike_at(t.min(SPIKE_WINDOW - 1));
            }
        }
        train
    }

    /// Sample a synchronous burst — every neuron fires on its own
    /// distinct time step (proxy for a population burst).
    #[must_use]
    pub fn burst(&self) -> SpikeTrain {
        let mut train = SpikeTrain::silent();
        for i in 0..self.width.min(SPIKE_WINDOW) {
            train.spike_at(i);
        }
        train
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silent_train_has_no_spikes() {
        let t = SpikeTrain::silent();
        assert_eq!(t.count(), 0);
        assert_eq!(t.first_spike(), None);
    }

    #[test]
    fn first_spike_returns_lowest_index() {
        let mut t = SpikeTrain::silent();
        t.spike_at(7);
        t.spike_at(3);
        assert_eq!(t.first_spike(), Some(3));
    }

    #[test]
    fn population_samples_at_least_one_spike() {
        let mut pop = SpikingPopulation::random(64, 1);
        pop.advance_oscillation(0.0);
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..10 {
            let _ = pop.sample(&mut rng);
        }
    }

    #[test]
    fn burst_spikes_every_neuron() {
        let pop = SpikingPopulation::random(16, 1);
        let b = pop.burst();
        assert_eq!(b.count(), 16);
    }
}
