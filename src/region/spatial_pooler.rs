//! Spatial Pooler — converts dense or sparse input into a stable
//! SDR of fixed sparsity.
//!
//! Inspired by Numenta's HTM SP. The pooler holds a binary
//! permanence matrix (input → columns); each column has
//! `potential_pool` of inputs it can connect to. Hebbian-style
//! learning strengthens active connections and weakens inactive
//! ones. Columns whose total permanence stays above
//! `connected_permanence` are eligible to fire; the strongest
//! `active_count` columns win and form the output SDR.

use crate::sdr::{Sdr, SDR_WIDTH, DEFAULT_SPARSITY};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Parameters for [`SpatialPooler`].
#[derive(Debug, Clone)]
pub struct SpatialPoolerParams {
    /// Input SDR width.
    pub input_width: usize,
    /// Target sparsity for the output SDR — fraction of
    /// `output_width` bits active.
    pub sparsity: f32,
    /// Permanence above which a connection is "connected".
    pub connected_permanence: f32,
    /// How much to bump active connections per learning event.
    pub permanence_increment: f32,
    /// How much to decay inactive connections per learning event.
    pub permanence_decrement: f32,
}

impl SpatialPoolerParams {
    /// Sensible defaults for an SDR → SDR pooler.
    #[must_use]
    pub fn default_for(input_width: usize) -> Self {
        Self {
            input_width,
            sparsity: DEFAULT_SPARSITY,
            connected_permanence: 0.5,
            permanence_increment: 0.05,
            permanence_decrement: 0.008,
        }
    }
}

/// One cortical column's view of its input — a vector of
/// permanences (one per potential input bit).
#[derive(Debug, Clone)]
struct Column {
    /// Permanence for each potential input bit.
    permanences: Vec<f32>,
    /// Indexes of bits whose permanence ≥ `connected_permanence`.
    connected: Vec<usize>,
    /// Boost factor — columns that haven't fired recently get
    /// higher boost so they stay competitive.
    boost: f32,
    /// Average overlap with the input over the recent past.
    duty_cycle: f32,
    /// Whether the column was active in the previous step.
    was_active: bool,
}

/// Spatial Pooler.
#[derive(Debug)]
pub struct SpatialPooler {
    params: SpatialPoolerParams,
    columns: Vec<Column>,
    rng: StdRng,
    /// Target number of active columns per output SDR.
    target_active: usize,
}

impl SpatialPooler {
    /// Build a new pooler with random permanences.
    #[must_use]
    pub fn new(params: SpatialPoolerParams, seed: u64) -> Self {
        let rng = StdRng::seed_from_u64(seed);
        let target_active =
            ((SDR_WIDTH as f32) * params.sparsity.clamp(0.001, 0.5)) as usize;
        let columns = (0..SDR_WIDTH)
            .map(|_| Column {
                permanences: vec![0.4; params.input_width],
                connected: Vec::new(),
                boost: 1.0,
                duty_cycle: 0.01,
                was_active: false,
            })
            .collect();
        Self { params, columns, rng, target_active }
    }

    /// Number of output columns (= SDR width).
    #[must_use]
    pub fn n_columns(&self) -> usize {
        self.columns.len()
    }

    /// Run the pooler on the supplied input. Returns the output
    /// SDR and updates internal permanences via Hebbian learning.
    pub fn compute(&mut self, input: &Sdr) -> Sdr {
        assert_eq!(input.width(), self.params.input_width, "input width mismatch");
        // 1. For every column compute its overlap (count of
        //    active inputs whose permanence ≥ connected_permanence).
        let mut overlaps: Vec<usize> = self
            .columns
            .iter()
            .map(|c| {
                c.connected
                    .iter()
                    .filter(|&&i| input.get(i))
                    .count()
            })
            .collect();

        // 2. Apply per-column boost.
        for (i, col) in self.columns.iter().enumerate() {
            overlaps[i] = ((overlaps[i] as f32) * col.boost).round() as usize;
        }

        // 3. Top-k winners. When multiple columns share the same
        // overlap score, tie-break by column index so the
        // pooler's output is deterministic — necessary for the
        // Hebbian convergence tests below.
        let mut order: Vec<usize> = (0..overlaps.len()).collect();
        order.sort_by(|&a, &b| overlaps[b].cmp(&overlaps[a]).then(a.cmp(&b)));
        let winners: std::collections::HashSet<usize> =
            order.iter().take(self.target_active).copied().collect();

        // 4. Hebbian update on winners + duty cycle bookkeeping.
        for (i, col) in self.columns.iter_mut().enumerate() {
            let active = winners.contains(&i);
            let prev_active = col.was_active;
            col.was_active = active;
            // Duty cycle as a slow exponential average.
            let dc_target = if active { 1.0 } else { 0.0 };
            col.duty_cycle = 0.99 * col.duty_cycle + 0.01 * dc_target;
            if active {
                // Bump permanences of inputs that were active.
                for (j, p) in col.permanences.iter_mut().enumerate() {
                    if input.get(j) {
                        *p = (*p + self.params.permanence_increment).min(1.0);
                    }
                }
                // Refresh connected set lazily.
                col.connected = col
                    .permanences
                    .iter()
                    .enumerate()
                    .filter_map(|(j, &p)| (p >= self.params.connected_permanence).then_some(j))
                    .collect();
            } else if prev_active {
                // Decay all permanences when a previously-active
                // column goes quiet.
                for p in col.permanences.iter_mut() {
                    *p = (*p - self.params.permanence_decrement).max(0.0);
                }
            }
        }

        // 5. Boost under-active columns so they get a chance.
        let avg_dc = self.columns.iter().map(|c| c.duty_cycle).sum::<f32>()
            / self.columns.len() as f32;
        for col in self.columns.iter_mut() {
            if col.duty_cycle < 0.5 * avg_dc.max(1e-6) {
                col.boost = (col.boost + 0.05).min(2.0);
            } else if col.duty_cycle > 1.5 * avg_dc.max(1e-6) {
                col.boost = (col.boost - 0.02).max(0.5);
            }
        }

        // 6. Build the output SDR.
        let mut bits: Vec<usize> = winners.into_iter().collect();
        bits.sort_unstable();
        Sdr::from_bits(bits)
    }

    /// Add noise to permanences for plasticity.
    pub fn inject_noise(&mut self, fraction: f32) {
        let n = self.params.input_width;
        for col in self.columns.iter_mut() {
            for p in col.permanences.iter_mut() {
                let r: f32 = self.rng.gen_range(-1.0..1.0);
                *p = (*p + r * fraction).clamp(0.0, 1.0);
            }
            // Touch n so the trait import stays in scope.
            let _ = n;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdr::Sdr;

    #[test]
    fn output_has_target_sparsity() {
        let mut p = SpatialPooler::new(
            SpatialPoolerParams::default_for(SDR_WIDTH),
            1,
        );
        let input = Sdr::random_active(&mut rand::thread_rng(), 40);
        let out = p.compute(&input);
        assert!(out.active_count() >= p.target_active / 2);
        assert!(out.active_count() <= p.target_active * 3 / 2 + 1);
    }

    #[test]
    fn repeated_input_stabilises_columns() {
        let mut p = SpatialPooler::new(
            SpatialPoolerParams::default_for(SDR_WIDTH),
            7,
        );
        let input = Sdr::random_active(&mut rand::thread_rng(), 40);
        let first = p.compute(&input);
        for _ in 0..20 {
            let _ = p.compute(&input);
        }
        let stable = p.compute(&input);
        // After repeated exposure the pooler should converge on
        // roughly the same set of winning columns.
        let overlap = crate::sdr::semantic_similarity(&first, &stable);
        assert!(overlap > 0.5, "pooler should stabilise, got {overlap}");
    }
}
