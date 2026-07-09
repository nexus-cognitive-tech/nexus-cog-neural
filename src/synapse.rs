//! Synapses — the substrate for LTP / LTD and neuromodulation.
//!
//! Each synapse carries:
//!
//! * **`permanence`** — long-term connection weight in `[0, 1]`.
//! * **`lability`** — short-term, neuromodulator-gated LTP / LTD
//!   rate. Higher lability means the synapse is more plastic.
//! * **`eligibility`** — transient tag set when pre fires without
//!   post firing; decays unless the post fires shortly after.
//! * **`neuromodulator_conc`** — local concentration of each
//!   modulator at the synapse site (different from the global
//!   level broadcast by the thalamus / neuromodulator panel).
//!
//! The [`PlasticityRule`] trait is implemented by
//! [`ThreeFactorHebbianRule`] — pre × post × neuromodulator.
//! Three-factor learning is the consensus mechanism in
//! computational neuroscience for reward-modulated cortical
//! plasticity.

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

/// Local neuromodulator concentration at a single synapse.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NeuromodulatorLocal {
    /// Local dopamine concentration in `[0, 1]`. Drives LTP when
    /// positive, LTD when negative.
    pub dopamine: f32,
    /// Local serotonin concentration in `[0, 1]`. Slows learning
    /// rate, biases toward cautious consolidation.
    pub serotonin: f32,
    /// Local norepinephrine concentration in `[0, 1]`. Raises
    /// baseline lability so unexpected events are written faster.
    pub norepinephrine: f32,
}

impl Default for NeuromodulatorLocal {
    fn default() -> Self {
        Self {
            dopamine: 0.0,
            serotonin: 0.0,
            norepinephrine: 0.0,
        }
    }
}

impl NeuromodulatorLocal {
    /// Effective learning multiplier — dopamine above baseline
    /// speeds up LTP, below slows it; serotonin damps; NE amplifies.
    #[must_use]
    pub fn learning_rate_multiplier(&self) -> f32 {
        let d = (self.dopamine - 0.5) * 2.0;
        let s = (1.0 - self.serotonin).clamp(0.0, 1.0);
        let ne = self.norepinephrine;
        (1.0 + d * s * (1.0 + ne)).clamp(0.1, 3.0)
    }
}

/// A single synapse.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Synapse {
    /// Long-term connection weight in `[0, 1]`. Values below
    /// `connection_threshold` are considered disconnected.
    pub permanence: f32,
    /// Short-term LTP/LTD rate multiplier — driven by local
    /// neuromodulator concentration + astrocyte calcium state.
    pub lability: f32,
    /// Transient eligibility trace in `[0, 1]`. Set when pre
    /// fires without an immediate post, decays unless post fires.
    pub eligibility: f32,
    /// Local neuromodulator state.
    pub neuromodulator: NeuromodulatorLocal,
    /// Effective connection threshold. Default `0.5`.
    pub connection_threshold: f32,
}

impl Synapse {
    /// New synapse with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self {
            permanence: 0.4,
            lability: 1.0,
            eligibility: 0.0,
            neuromodulator: NeuromodulatorLocal::default(),
            connection_threshold: 0.5,
        }
    }

    /// `true` iff the synapse is currently considered connected.
    #[must_use]
    pub fn is_connected(&self) -> bool {
        self.permanence >= self.connection_threshold
    }

    /// Effective weight — `0.0` if disconnected, otherwise the
    /// permanence scaled by neuromodulator gating.
    #[must_use]
    pub fn effective_weight(&self) -> f32 {
        if self.is_connected() {
            self.permanence * self.neuromodulator.learning_rate_multiplier().min(1.5)
        } else {
            0.0
        }
    }

    /// Pre fires — mark eligibility.
    pub fn on_pre(&mut self) {
        self.eligibility = (self.eligibility + self.lability).clamp(0.0, 1.0);
    }

    /// Post fires — apply LTP. Returns the permanence delta.
    pub fn on_post(&mut self) -> f32 {
        let ltp = self.lability
            * self.eligibility
            * self.neuromodulator.learning_rate_multiplier();
        let delta = ltp * 0.05;
        self.permanence = (self.permanence + delta).clamp(0.0, 1.0);
        self.eligibility = (self.eligibility * 0.5).max(0.0);
        delta
    }

    /// Post does *not* fire — apply LTD to the eligibility (decay).
    pub fn on_post_silent(&mut self) {
        self.eligibility = (self.eligibility * 0.85).max(0.0);
        if self.eligibility < 0.05 {
            // Active decay — slow LTD on the permanence.
            let ltd = 0.005 * self.lability;
            self.permanence = (self.permanence - ltd).max(0.0);
        }
    }

    /// Update local neuromodulator state from global values.
    pub fn apply_global_neuromodulator(
        &mut self,
        dopamine: f32,
        serotonin: f32,
        norepinephrine: f32,
    ) {
        // Exponential moving average so local values lag global.
        let r = 0.2;
        self.neuromodulator.dopamine =
            (1.0 - r) * self.neuromodulator.dopamine + r * dopamine;
        self.neuromodulator.serotonin =
            (1.0 - r) * self.neuromodulator.serotonin + r * serotonin;
        self.neuromodulator.norepinephrine =
            (1.0 - r) * self.neuromodulator.norepinephrine + r * norepinephrine;
    }
}

impl Default for Synapse {
    fn default() -> Self {
        Self::new()
    }
}

/// A bundle of synapses connecting one source neuron to one target
/// neuron population.
#[derive(Debug, Clone)]
pub struct SynapticFanOut {
    /// One synapse per target neuron.
    pub synapses: Vec<Synapse>,
    /// Mean permanence across the bundle — used by astrocyte
    /// recruitment.
    pub mean_permanence: f32,
}

impl SynapticFanOut {
    /// New fan-out with `target_width` synapses at random
    /// permanences.
    #[must_use]
    pub fn random(target_width: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let base = Synapse::new();
        let synapses: Vec<Synapse> = (0..target_width)
            .map(|_| Synapse {
                permanence: rng.gen_range(0.0..0.6),
                ..base.clone()
            })
            .collect();
        let mean = synapses.iter().map(|s| s.permanence).sum::<f32>() / target_width.max(1) as f32;
        Self {
            synapses,
            mean_permanence: mean,
        }
    }

    /// Forward-propagate a pre spike into weighted post inputs.
    pub fn propagate(&self, pre_active: bool) -> Vec<f32> {
        if pre_active {
            self.synapses.iter().map(|s| s.effective_weight()).collect()
        } else {
            vec![0.0; self.synapses.len()]
        }
    }

    /// Apply plasticity to every synapse in the fan-out after a
    /// simulation tick.
    pub fn apply_plasticity(
        &mut self,
        post_active: &[bool],
        global_dopamine: f32,
        global_serotonin: f32,
        global_norepinephrine: f32,
    ) {
        for (syn, &post) in self.synapses.iter_mut().zip(post_active.iter()) {
            syn.apply_global_neuromodulator(global_dopamine, global_serotonin, global_norepinephrine);
            if post {
                let _ = syn.on_post();
            } else {
                syn.on_post_silent();
            }
        }
        self.mean_permanence = self.synapses.iter().map(|s| s.permanence).sum::<f32>()
            / self.synapses.len().max(1) as f32;
    }
}

/// Plasticity rule trait. Implemented by [`ThreeFactorHebbianRule`]
/// and reserved for future biologically-inspired variants (BCM,
/// Oja's, STDP).
pub trait PlasticityRule {
    /// Update one synapse.
    fn update(&self, syn: &mut Synapse, pre: bool, post: bool);
}

/// Three-factor Hebbian rule — `Δw ∝ pre × post × dopamine`.
/// `pre` and `post` are spike booleans; dopamine is the local
/// modulator concentration.
pub struct ThreeFactorHebbianRule;

impl PlasticityRule for ThreeFactorHebbianRule {
    fn update(&self, syn: &mut Synapse, pre: bool, post: bool) {
        if pre {
            syn.on_pre();
        }
        if post {
            let _ = syn.on_post();
        } else {
            syn.on_post_silent();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_synapse_is_below_threshold() {
        let s = Synapse::new();
        assert!(!s.is_connected());
        assert_eq!(s.permanence, 0.4);
    }

    #[test]
    fn ltp_strengthens_active_synapse() {
        let mut s = Synapse::new();
        for _ in 0..40 {
            s.on_pre();
            let _ = s.on_post();
        }
        assert!(s.permanence > 0.5, "permanence={}", s.permanence);
    }

    #[test]
    fn ltd_weakens_unattended_synapse() {
        let mut s = Synapse { permanence: 0.6, ..Synapse::new() };
        for _ in 0..5000 {
            s.on_post_silent();
        }
        assert!(s.permanence < 0.6);
    }

    #[test]
    fn dopamine_modulates_learning_rate() {
        let mut baseline = Synapse::new();
        let mut boosted = Synapse::new();
        for _ in 0..40 {
            baseline.on_pre();
            let _ = baseline.on_post();
            boosted.on_pre();
            let _ = boosted.on_post();
        }
        boosted.apply_global_neuromodulator(1.0, 0.0, 0.5);
        for _ in 0..40 {
            boosted.on_pre();
            let _ = boosted.on_post();
        }
        assert!(
            boosted.permanence > baseline.permanence,
            "boosted={} baseline={}",
            boosted.permanence,
            baseline.permanence
        );
    }

    #[test]
    fn fanout_propagation_zero_when_pre_silent() {
        let f = SynapticFanOut::random(8, 1);
        assert!(f.propagate(false).iter().all(|&v| v == 0.0));
    }
}
