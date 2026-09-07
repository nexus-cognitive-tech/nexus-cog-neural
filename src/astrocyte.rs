//! Astrocytes — glial cells that modulate the neural environment.
//!
//! In the **tripartite synapse** model, the astrocyte is a third
//! active partner in every synapse: it senses spillover
//! neurotransmitter from pre / post firing, raises its internal
//! `calcium` concentration, and releases gliotransmitters (ATP /
//! glutamate / D-serine) that:
//!
//! 1. modulate the plasticity of nearby synapses,
//! 2. set the local threshold for NMDA-receptor activation,
//! 3. propagate slow calcium waves to neighbouring astrocytes,
//! 4. trigger gliosis (proliferation) when local activity stays
//!    high for too long.
//!
//! One astrocyte is responsible for one cortical column's worth
//! of synapses. Astrocytes are first-class citizens in this
//! cortex — they have state, learn, and influence every spike
//! that passes through their domain.

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

/// Astrocyte state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Astrocyte {
    /// Internal calcium concentration in `[0, 1]`. Rises on
    /// local synaptic activity; decays toward baseline.
    pub calcium: f32,
    /// Resting calcium level in `[0, 1]`.
    pub baseline: f32,
    /// Threshold at which the astrocyte releases gliotransmitter.
    pub activation_threshold: f32,
    /// Time since last activation (in ticks). Drives gliosis if
    /// chronically active.
    pub active_ticks: u32,
    /// Number of neighbouring astrocytes this one is coupled to
    /// for slow calcium-wave propagation.
    pub neighbours: Vec<usize>,
}

impl Default for Astrocyte {
    fn default() -> Self {
        Self {
            calcium: 0.0,
            baseline: 0.05,
            activation_threshold: 0.6,
            active_ticks: 0,
            neighbours: Vec::new(),
        }
    }
}

impl Astrocyte {
    /// Step the astrocyte forward one tick, given the synaptic
    /// activity in its column.
    ///
    /// `column_activity` is the mean weight of all synapses in
    /// the column over `[0, 1]`.
    pub fn step(&mut self, column_activity: f32) -> f32 {
        // Calcium rises with activity, decays toward baseline.
        let drive = column_activity.clamp(0.0, 1.0);
        let target = if drive > 0.1 { drive } else { self.baseline };
        self.calcium = self.calcium * 0.92 + target * 0.08;
        self.calcium = self.calcium.clamp(0.0, 1.0);

        if self.calcium > self.activation_threshold {
            self.active_ticks = self.active_ticks.saturating_add(1);
        } else {
            self.active_ticks = 0;
        }
        self.calcium
    }

    /// Triggered: returns the gliotransmitter release amount in
    /// `[0, 1]`. Callers use this to boost local synapse
    /// plasticity.
    #[must_use]
    pub fn release(&self) -> f32 {
        if self.calcium > self.activation_threshold {
            (self.calcium - self.activation_threshold) / (1.0 - self.activation_threshold)
        } else {
            0.0
        }
    }

    /// Chronic-activity flag — true iff the astrocyte has been
    /// over-threshold for `ticks` consecutive ticks.
    #[must_use]
    pub fn is_chronically_active(&self, ticks: u32) -> bool {
        self.active_ticks >= ticks
    }

    /// Calcium-wave coupling — diffuse some of `incoming` calcium
    /// from a neighbouring astrocyte into this one.
    pub fn receive_wave(&mut self, incoming: f32) {
        self.calcium = (self.calcium + incoming * 0.3).clamp(0.0, 1.0);
    }
}

/// Population of astrocytes, one per cortical column. They form a
/// graph whose edges are the slow calcium-wave couplings.
pub struct AstrocyteNetwork {
    /// One astrocyte per column.
    pub cells: Vec<Astrocyte>,
    /// Global gliosis pressure in `[0, 1]`. High values trigger
    /// astrocyte proliferation (indirectly — the cortex asks for
    /// new columns when pressure is sustained).
    pub gliosis_pressure: f32,
}

impl AstrocyteNetwork {
    /// Build a network with `n` astrocytes wired in a small-world
    /// ring topology.
    #[must_use]
    pub fn ring(n: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut cells: Vec<Astrocyte> = (0..n)
            .map(|i| {
                let mut a = Astrocyte::default();
                // Ring neighbours ±2 indices + one random long-range.
                let prev = if i == 0 { n - 1 } else { i - 1 };
                let next = if i == n - 1 { 0 } else { i + 1 };
                a.neighbours.push(prev);
                a.neighbours.push(next);
                if n > 4 {
                    let r = rng.gen_range(0..n);
                    if r != i {
                        a.neighbours.push(r);
                    }
                }
                a
            })
            .collect();
        // Gliosis pressure drops baseline slightly to break
        // symmetry.
        for c in &mut cells {
            c.baseline = 0.05 + rng.gen_range(0.0..0.05);
        }
        Self { cells, gliosis_pressure: 0.0 }
    }

    /// Tick the whole network. Returns the per-column gliotransmitter
    /// release vector.
    pub fn step(&mut self, column_activities: &[f32]) -> Vec<f32> {
        // First pass: local calcium update.
        let new_calcium: Vec<f32> = self
            .cells
            .iter_mut()
            .zip(column_activities.iter())
            .map(|(c, &a)| c.step(a))
            .collect();

        // Second pass: calcium-wave propagation. Use the
        // just-computed calcium as the source for diffusion.
        let waves: Vec<f32> = self
            .cells
            .iter()
            .map(|c| {
                c.neighbours
                    .iter()
                    .map(|&i| new_calcium.get(i).copied().unwrap_or(0.0))
                    .sum::<f32>()
                    / c.neighbours.len().max(1) as f32
            })
            .collect();

        // Apply received wave as an additional bump.
        for (cell, incoming) in self.cells.iter_mut().zip(waves) {
            cell.receive_wave(incoming);
        }

        // Global gliosis pressure = mean chronic-activity.
        let chronic = self
            .cells
            .iter()
            .filter(|c| c.is_chronically_active(50))
            .count() as f32
            / self.cells.len().max(1) as f32;
        self.gliosis_pressure = self.gliosis_pressure * 0.95 + chronic * 0.05;

        // Returns the gliotransmitter release per cell.
        self.cells.iter().map(Astrocyte::release).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn astrocyte_decays_toward_baseline() {
        let mut a = Astrocyte::default();
        a.calcium = 0.8;
        for _ in 0..200 {
            a.step(0.0);
        }
        assert!(a.calcium < 0.1, "calcium={}", a.calcium);
    }

    #[test]
    fn astrocyte_releases_when_active() {
        let mut a = Astrocyte::default();
        a.calcium = 0.9;
        assert!(a.release() > 0.0);
    }

    #[test]
    fn astrocyte_does_not_release_when_quiet() {
        let a = Astrocyte::default();
        assert_eq!(a.release(), 0.0);
    }

    #[test]
    fn network_ring_propagates_wave() {
        let mut net = AstrocyteNetwork::ring(5, 1);
        let activity = vec![0.0, 0.0, 1.0, 0.0, 0.0];
        let release = net.step(&activity);
        assert_eq!(release.len(), 5);
        // Neighbour of the active cell should have calcium
        // elevated via wave.
        assert!(net.cells[1].calcium > 0.0);
        assert!(net.cells[3].calcium > 0.0);
    }
}
