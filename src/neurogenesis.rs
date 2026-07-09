//! Neurogenesis — birth, maturation and apoptosis of cortical
//! columns.
//!
//! In the adult mammalian brain, neurogenesis happens in two
//! places: the subgranular zone of the dentate gyrus (hippocampus)
//! and the subventricular zone (which feeds the olfactory bulb).
//! In our cortex model every cortical column is a potential
//! neurogenic niche:
//!
//! * **Birth** — driven by high local novelty + dopamine +
//!   norepinephrine. Newborn columns are seeded with high
//!   lability so they explore fast.
//! * **Maturation** — over `maturation_ticks` ticks the column
//!   develops stable permanence distributions; lability drops to
//!   baseline.
//! * **Apoptosis** — columns whose mean permanence drops below
//!   `apoptosis_floor` for `apoptosis_ticks` consecutive ticks
//!   die and are pruned.
//!
//! The cortex consults [`NeurogenicRate::rate`] every sleep cycle
//! to decide whether to spawn a new column.

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

/// Configuration for the neurogenic controller.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeurogenesisConfig {
    /// Number of ticks a newborn column needs to mature.
    pub maturation_ticks: u32,
    /// Number of consecutive low-activity ticks before a column
    /// is pruned.
    pub apoptosis_ticks: u32,
    /// Permanence floor below which a column is apoptosis-prone.
    pub apoptosis_floor: f32,
    /// Per-tick probability of birthing a new column when all
    /// conditions are met.
    pub birth_rate: f32,
    /// Minimum number of columns the network maintains even when
    /// no activity is present.
    pub min_columns: usize,
}

impl Default for NeurogenesisConfig {
    fn default() -> Self {
        Self {
            maturation_ticks: 200,
            apoptosis_ticks: 400,
            apoptosis_floor: 0.1,
            birth_rate: 0.005,
            min_columns: 3,
        }
    }
}

/// Live birth / apoptosis decision for one column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeurogenicVerdict {
    /// Column survives; no change.
    Survive,
    /// Column should be pruned.
    Apoptose,
}

/// Neurogenic controller — pure logic, no domain knowledge.
#[derive(Debug, Clone)]
pub struct Neurogenesis {
    pub config: NeurogenesisConfig,
    /// Per-column counter of consecutive low-permanence ticks.
    pub low_activity_streak: Vec<u32>,
    /// Random source for stochastic birth.
    rng: StdRng,
}

impl Neurogenesis {
    pub fn new(config: NeurogenesisConfig, seed: u64) -> Self {
        Self {
            config,
            low_activity_streak: Vec::new(),
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Per-tick birth probability in `[0, 1]` based on global
    /// neuromodulators + novelty + current column count.
    pub fn birth_rate(
        &self,
        dopamine: f32,
        norepinephrine: f32,
        novelty: f32,
        n_columns: usize,
    ) -> f32 {
        // The novelty × dopamine × NE gate. Saturates at 1.
        let score =
            (novelty * 1.5).clamp(0.0, 1.0)
            * (0.5 + dopamine).clamp(0.0, 1.5)
            * (0.5 + norepinephrine * 0.5).clamp(0.0, 1.5);
        let density = (1.0 / (n_columns as f32 + 1.0)).clamp(0.0, 1.0);
        (self.config.birth_rate * score * (0.5 + density * 2.0)).clamp(0.0, 1.0)
    }

    /// Decide whether each column survives this tick.
    pub fn tick(
        &mut self,
        mean_permanences: &[f32],
    ) -> Vec<NeurogenicVerdict> {
        if self.low_activity_streak.len() != mean_permanences.len() {
            self.low_activity_streak = vec![0; mean_permanences.len()];
        }
        // First pass: identify candidates for apoptosis.
        let mut candidates: Vec<bool> = mean_permanences
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                if p < self.config.apoptosis_floor {
                    self.low_activity_streak[i] =
                        self.low_activity_streak[i].saturating_add(1);
                } else {
                    self.low_activity_streak[i] = 0;
                }
                self.low_activity_streak[i] >= self.config.apoptosis_ticks
            })
            .collect();

        // Ensure we never drop below `min_columns` — kill apoptosis
        // candidates from the *highest* streaks first until we'd
        // be back at the floor.
        let min = self.config.min_columns;
        let n = mean_permanences.len();
        if min < n {
            while n - candidates.iter().filter(|&&c| c).count() < min {
                // Find the candidate with the lowest streak and
                // spare it.
                let spare = candidates
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| **c)
                    .min_by_key(|(i, _)| self.low_activity_streak[*i])
                    .map(|(i, _)| i);
                if let Some(idx) = spare {
                    candidates[idx] = false;
                } else {
                    break;
                }
            }
        }

        candidates
            .into_iter()
            .map(|c| {
                if c {
                    NeurogenicVerdict::Apoptose
                } else {
                    NeurogenicVerdict::Survive
                }
            })
            .collect()
    }

    /// Decide whether to birth a new column this tick. Returns
    /// `true` if a new column should be added.
    pub fn should_birth(
        &mut self,
        dopamine: f32,
        norepinephrine: f32,
        novelty: f32,
        n_columns: usize,
    ) -> bool {
        let r = self.birth_rate(dopamine, norepinephrine, novelty, n_columns);
        self.rng.r#gen::<f32>() < r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn birth_rate_scales_with_novelty_and_dopamine() {
        let ng = Neurogenesis::new(NeurogenesisConfig::default(), 1);
        let low = ng.birth_rate(0.1, 0.1, 0.1, 5);
        let high = ng.birth_rate(1.0, 1.0, 1.0, 5);
        assert!(high > low);
    }

    #[test]
    fn low_permanence_triggers_apoptosis() {
        let mut ng = Neurogenesis::new(
            NeurogenesisConfig {
                apoptosis_ticks: 3,
                apoptosis_floor: 0.5,
                min_columns: 0,
                ..NeurogenesisConfig::default()
            },
            1,
        );
        let mut verdicts = ng.tick(&[0.1, 0.6]);
        assert_eq!(verdicts[0], NeurogenicVerdict::Survive);
        for _ in 0..3 {
            verdicts = ng.tick(&[0.1, 0.6]);
        }
        assert_eq!(verdicts[0], NeurogenicVerdict::Apoptose);
        assert_eq!(verdicts[1], NeurogenicVerdict::Survive);
    }

    #[test]
    fn minimum_columns_protected() {
        let mut ng = Neurogenesis::new(
            NeurogenesisConfig {
                apoptosis_ticks: 1,
                min_columns: 2,
                ..NeurogenesisConfig::default()
            },
            1,
        );
        let verdicts = ng.tick(&[0.0, 0.0, 0.0]);
        // Three columns but only one (column 0) marked for
        // apoptosis — but we cannot drop below 2.
        assert_eq!(verdicts[1], NeurogenicVerdict::Survive);
    }
}
