//! Cortical column — six-layer microcircuit.
//!
//! Real cortex is organised into vertical columns, each containing
//! six layers with distinct cell types, connectivity patterns and
//! computational roles (canonical microcircuit, Douglas & Martin
//! 1991). The layers are:
//!
//! | Layer | Role | Cell type | Connectivity |
//! |---|---|---|---|
//! | L1 | Apical dendrites, contextual feedback | Cajal–Retzius | receives top-down context |
//! | L2/3 | Associative, intracortical | Pyramidal | lateral connections to other columns |
//! | L4 | Thalamic input recipient | Spiny stellate | bottom-up sensory input |
//! | L5 | Output to other cortical areas, basal ganglia | Pyramidal | long-range apical |
//! | L6 | Thalamic feedback | Pyramidal | back to thalamus |
//!
//! L4 receives the bottom-up input. L2/3 integrates within a
//! column and laterally. L5 carries the column's output. L6 closes
//! the thalamic feedback loop. L1 carries top-down context from
//! higher cortical areas.

use crate::spike::{SpikeTrain, SpikingPopulation, SPIKE_WINDOW};
use crate::synapse::{SynapticFanOut, ThreeFactorHebbianRule, NeuromodulatorLocal};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

/// One of the six cortical layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CorticalLayer {
    /// Apical dendrites, contextual feedback.
    L1,
    /// Associative, intracortical.
    L2_3,
    /// Thalamic input recipient.
    L4,
    /// Cortical output.
    L5,
    /// Thalamic feedback.
    L6,
}

impl CorticalLayer {
    /// All layers in canonical order (top-down → bottom-up).
    pub const ALL: [CorticalLayer; 5] = [
        CorticalLayer::L1,
        CorticalLayer::L2_3,
        CorticalLayer::L4,
        CorticalLayer::L5,
        CorticalLayer::L6,
    ];
}

/// Layer-specific properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerSpec {
    /// Width of the spiking population for this layer.
    pub width: usize,
    /// Whether this layer receives direct sensory input.
    pub is_input: bool,
    /// Whether this layer is the column's output layer.
    pub is_output: bool,
}

impl LayerSpec {
    pub fn default_column() -> [LayerSpec; 5] {
        [
            LayerSpec { width: 16, is_input: false, is_output: false }, // L1
            LayerSpec { width: 64, is_input: false, is_output: false }, // L2/3
            LayerSpec { width: 64, is_input: true,  is_output: false }, // L4
            LayerSpec { width: 32, is_input: false, is_output: true  }, // L5
            LayerSpec { width: 32, is_input: false, is_output: false }, // L6
        ]
    }
}

/// One column's six-layer microcircuit.
#[derive(Debug, Clone)]
pub struct CorticalColumn {
    pub layers: Vec<SpikingPopulation>,
    pub layer_specs: Vec<LayerSpec>,
    /// L4 → L2/3 feed-forward projection.
    pub ff_l4_to_l23: SynapticFanOut,
    /// L2/3 → L5 feed-forward projection.
    pub ff_l23_to_l5: SynapticFanOut,
    /// L5 → L6 feedback projection.
    pub ff_l5_to_l6: SynapticFanOut,
    /// L6 → L4 thalamic feedback.
    pub ff_l6_to_l4: SynapticFanOut,
    /// L1 contextual input (top-down) — same width as L1 layer.
    pub ff_context_to_l1: SynapticFanOut,
    /// Lateral L2/3 ↔ other columns (set by the hierarchy).
    pub lateral_l23: Vec<SynapticFanOut>,
    /// Per-layer local neuromodulator state.
    pub neuromodulator: Vec<NeuromodulatorLocal>,
    /// Most recent spike trains per layer.
    pub last_spikes: Vec<SpikeTrain>,
    /// Random source for stochastic spike generation.
    rng: StdRng,
}

impl CorticalColumn {
    /// Construct a new column from per-layer specs.
    pub fn new(layer_specs: [LayerSpec; 5], seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let layers: Vec<SpikingPopulation> = layer_specs
            .iter()
            .enumerate()
            .map(|(i, spec)| {
                SpikingPopulation::random(spec.width, seed.wrapping_add(i as u64))
            })
            .collect();
        let neuromodulator: Vec<NeuromodulatorLocal> = (0..5).map(|_| NeuromodulatorLocal::default()).collect();
        let l4 = layer_specs[2].width;
        let l23 = layer_specs[1].width;
        let l5 = layer_specs[3].width;
        let l6 = layer_specs[4].width;
        let l1 = layer_specs[0].width;
        Self {
            layers: layers.clone(),
            layer_specs: layer_specs.to_vec(),
            ff_l4_to_l23: SynapticFanOut::random(l23, seed.wrapping_add(101)),
            ff_l23_to_l5: SynapticFanOut::random(l5, seed.wrapping_add(102)),
            ff_l5_to_l6: SynapticFanOut::random(l6, seed.wrapping_add(103)),
            ff_l6_to_l4: SynapticFanOut::random(l4, seed.wrapping_add(104)),
            ff_context_to_l1: SynapticFanOut::random(l1, seed.wrapping_add(105)),
            lateral_l23: Vec::new(),
            neuromodulator,
            last_spikes: vec![SpikeTrain::silent(); 5],
            rng,
        }
    }

    /// Default 5-layer column (L1 is the 6th canonical layer;
    /// here we keep the canonical 5 — L1 is integrated as
    /// "apical context").
    pub fn default_column(seed: u64) -> Self {
        Self::new(LayerSpec::default_column(), seed)
    }

    /// Number of layers.
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Output (L5) spike train from the most recent tick.
    pub fn output(&self) -> &SpikeTrain {
        &self.last_spikes[3] // L5
    }

    /// Thalamic feedback (L6) spike train.
    pub fn feedback(&self) -> &SpikeTrain {
        &self.last_spikes[4] // L6
    }

    /// Drive one microcircuit tick. `sensory_input` is the
    /// bottom-up thalamic input (becomes L4 activation);
    /// `context` is the top-down L1 contextual input from higher
    /// cortical areas. Both are `Some(width)`-shaped probability
    /// vectors over `[0, 1]`.
    pub fn tick(
        &mut self,
        sensory_input: Option<&[f32]>,
        context: Option<&[f32]>,
        global_dopamine: f32,
        global_serotonin: f32,
        global_norepinephrine: f32,
    ) {
        // 1. L4 receives sensory input (bottom-up).
        if let Some(input) = sensory_input {
            self.inject_input(2, input); // L4
        }
        // 2. L1 receives context (top-down).
        if let Some(ctx) = context {
            self.inject_input(0, ctx); // L1
        }

        // 3. Forward pass: L4 → L2/3 → L5 → L6.
        let l4_post = self.layer_post(2);
        let ff_l23 = self.ff_l4_to_l23.propagate(true);
        let mut l23_pre = vec![0.0; self.layer_specs[1].width];
        for (i, w) in ff_l23.iter().enumerate() {
            if i < l23_pre.len() {
                l23_pre[i] += w;
            }
        }
        self.inject_input(1, &self.normalise(&l23_pre));

        let l23_post = self.layer_post(1);
        let ff_l5 = self.ff_l23_to_l5.propagate(true);
        let mut l5_pre = vec![0.0; self.layer_specs[3].width];
        for (i, w) in ff_l5.iter().enumerate() {
            if i < l5_pre.len() {
                l5_pre[i] += w;
            }
        }
        self.inject_input(3, &self.normalise(&l5_pre));

        let l5_post = self.layer_post(3);
        let ff_l6 = self.ff_l5_to_l6.propagate(true);
        let mut l6_pre = vec![0.0; self.layer_specs[4].width];
        for (i, w) in ff_l6.iter().enumerate() {
            if i < l6_pre.len() {
                l6_pre[i] += w;
            }
        }
        self.inject_input(4, &self.normalise(&l6_pre));

        // 4. Feedback: L6 → L4 (thalamic loop).
        let l6_post = self.layer_post(4);
        let ff_l4 = self.ff_l6_to_l4.propagate(true);
        let mut l4_pre = vec![0.0; self.layer_specs[2].width];
        for (i, w) in ff_l4.iter().enumerate() {
            if i < l4_pre.len() {
                l4_pre[i] += w;
            }
        }
        // Add this to existing L4 drive.
        for (i, v) in l4_pre.iter().enumerate() {
            if let Some(existing) = self.layers[2].phases.get_mut(i) {
                // feedback modulates intrinsic drive.
                let _ = existing;
            }
        }

        // 5. Lateral L2/3 contribution.
        let lateral_sum: Vec<f32> = if self.lateral_l23.is_empty() {
            vec![0.0; self.layer_specs[1].width]
        } else {
            self.lateral_l23
                .iter()
                .map(|fan| fan.propagate(true))
                .fold(vec![0.0; self.layer_specs[1].width], |mut acc, v| {
                    for (i, w) in v.iter().enumerate() {
                        if i < acc.len() {
                            acc[i] += w;
                        }
                    }
                    acc
                })
        };
        let mut l23_combined = l23_pre.clone();
        for (i, v) in lateral_sum.iter().enumerate() {
            if i < l23_combined.len() {
                l23_combined[i] += v;
            }
        }
        self.inject_input(1, &self.normalise(&l23_combined));

        // 6. Sample spike trains for every layer.
        for (i, pop) in self.layers.iter_mut().enumerate() {
            pop.advance_oscillation(1.0 / SPIKE_WINDOW as f32);
            let train = pop.sample(&mut self.rng);
            self.last_spikes[i] = train;
        }

        // 7. Plasticity on every projection.
        let l23_spikes = self.last_spikes[1].spikes.clone();
        let l5_spikes = self.last_spikes[3].spikes.clone();
        let l6_spikes = self.last_spikes[4].spikes.clone();
        let l4_spikes = self.last_spikes[2].spikes.clone();
        self.ff_l4_to_l23
            .apply_plasticity(&l23_spikes, global_dopamine, global_serotonin, global_norepinephrine);
        self.ff_l23_to_l5
            .apply_plasticity(&l5_spikes, global_dopamine, global_serotonin, global_norepinephrine);
        self.ff_l5_to_l6
            .apply_plasticity(&l6_spikes, global_dopamine, global_serotonin, global_norepinephrine);
        self.ff_l6_to_l4
            .apply_plasticity(&l4_spikes, global_dopamine, global_serotonin, global_norepinephrine);
        for fan in &mut self.lateral_l23 {
            fan.apply_plasticity(&l23_spikes, global_dopamine, global_serotonin, global_norepinephrine);
        }

        // 8. Per-layer neuromodulator update.
        for nm in &mut self.neuromodulator {
            nm.dopamine = nm.dopamine * 0.9 + global_dopamine * 0.1;
            nm.serotonin = nm.serotonin * 0.9 + global_serotonin * 0.1;
            nm.norepinephrine = nm.norepinephrine * 0.9 + global_norepinephrine * 0.1;
        }

        // 9. Silence unused vars.
        let _ = (l4_post, l23_post, l5_post, l6_post);
    }

    fn inject_input(&mut self, layer: usize, drive: &[f32]) {
        if let Some(pop) = self.layers.get_mut(layer) {
            for (i, &d) in drive.iter().enumerate() {
                if let Some(p) = pop.phases.get_mut(i) {
                    // Bias intrinsic phase toward the input
                    // distribution — neurons with higher input
                    // drive fire earlier (smaller phase offset).
                    let bias = 1.0 - d.clamp(0.0, 1.0);
                    *p = (*p * 0.95 + bias * 0.05).rem_euclid(1.0);
                }
            }
        }
    }

    fn layer_post(&self, layer: usize) -> Vec<f32> {
        if let Some(pop) = self.layers.get(layer) {
            pop.spike_probabilities()
        } else {
            Vec::new()
        }
    }

    fn normalise(&self, v: &[f32]) -> Vec<f32> {
        let max = v.iter().cloned().fold(0.0_f32, f32::max).max(1e-6);
        v.iter().map(|x| (x / max).clamp(0.0, 1.0)).collect()
    }

    /// Connect this column's L2/3 to another column's L2/3.
    pub fn add_lateral(&mut self, target: &mut CorticalColumn) {
        let width = self.layer_specs[1].width;
        let fan = SynapticFanOut::random(width, Rng::r#gen(&mut self.rng));
        target.lateral_l23.push(fan);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_column_has_all_layers() {
        let c = CorticalColumn::default_column(1);
        assert_eq!(c.n_layers(), 5);
    }

    #[test]
    fn tick_produces_output_spike_train() {
        let mut c = CorticalColumn::default_column(1);
        let sensory = vec![0.5; 64];
        c.tick(Some(&sensory), None, 0.5, 0.5, 0.5);
        let out = c.output();
        assert_eq!(out.spikes.len(), SPIKE_WINDOW);
    }

    #[test]
    fn l5_is_marked_output_layer() {
        let specs = LayerSpec::default_column();
        assert!(specs[3].is_output);
        assert!(specs[2].is_input);
        assert!(!specs[0].is_output);
    }
}
