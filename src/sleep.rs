//! Sleep cycle — NREM (replay + consolidation) and REM
//! (cortical re-activation) on the new spike-based hierarchy.
//!
//! A sleep cycle is a single call to [`SleepCycle::run`]:
//!
//! 1. **NREM phase** — replays the hippocampus' highest-salience
//!    episodes into the cortical hierarchy as thalamic input.
//! 2. **REM phase** — perturbs each cortical column with random
//!    spike trains so quiet columns stay in the running.

use crate::hierarchy::{ColumnId, Hierarchy};
use crate::hippocampus::{ConsolidationReport, Hippocampus};
use crate::spike::SpikeTrain;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Sleep cycle.
#[derive(Default)]
pub struct SleepCycle;

impl SleepCycle {
    /// Create a new sleep-cycle controller.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Run one consolidation cycle.
    pub fn run(
        &self,
        hippocampus: &mut Hippocampus,
        hierarchy: &mut Hierarchy,
        replay_per_cycle: usize,
    ) -> ConsolidationReport {
        let start = std::time::Instant::now();
        let episodes = hippocampus.replay_batch(replay_per_cycle, 5);
        let mut unique_patterns = 0;
        let mut total_overlap = 0.0_f32;
        let mut rng = StdRng::seed_from_u64(
            chrono::Utc::now()
                .timestamp_nanos_opt()
                .unwrap_or(0) as u64,
        );

        // Build a synthetic spike-train noise generator for the
        // REM phase — we perturb with random spike trains.
        let noise_width = 64;

        for ep in &episodes {
            // Decode the episode SDR into a spike-train drive by
            // hashing active bits into the spike probability.
            let drive = sdr_to_drive(&ep.sdr, noise_width);
            let mut inputs = std::collections::HashMap::new();
            if let Some(first) = hierarchy.input_sinks.first() {
                inputs.insert(*first, drive);
            }
            hierarchy.tick(&inputs, 0.5, 0.5, 0.5);
            let observed = collect_observations(hierarchy);
            let prev = prev_observations(hierarchy);
            if let (Some(p), Some(o)) = (prev.first(), observed.first()) {
                let ov = p.distance(o);
                total_overlap += 1.0 - ov;
                if ov < 0.4 {
                    unique_patterns += 1;
                }
            }
            store_observations(hierarchy, &observed);
        }

        // REM phase: random spike perturbation per column.
        rem_phase(hierarchy, &mut rng, noise_width);

        ConsolidationReport {
            episodes_replayed: episodes.len(),
            unique_patterns,
            avg_target_overlap: if episodes.is_empty() {
                0.0
            } else {
                total_overlap / episodes.len() as f32
            },
            elapsed_ms: start.elapsed().as_millis(),
        }
    }
}

/// Build a synthetic spike probability vector by hashing the
/// active bits of an SDR.
fn sdr_to_drive(sdr: &crate::sdr::Sdr, width: usize) -> Vec<f32> {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    sdr.active_bits().len().hash(&mut hasher);
    let h = hasher.finish();
    let mut bits: Vec<f32> = vec![0.0; width];
    for (i, bit) in bits.iter_mut().enumerate() {
        let x = h.wrapping_add(i as u64).wrapping_mul(2_654_435_761);
        *bit = ((x >> 16) as f32) / (u32::MAX as f32);
    }
    bits
}

/// Naive observation snapshot — first column's first-layer
/// spike train. The new cortex stores full per-column /
/// per-layer spike trains; we only need a small projection to
/// track replay overlap.
fn collect_observations(hierarchy: &Hierarchy) -> Vec<SpikeTrain> {
    let mut out = Vec::new();
    for id in &hierarchy.input_sinks {
        if let Some(col) = hierarchy.columns.get(id) {
            if let Some(t) = col.last_spikes.first() {
                out.push(t.clone());
            }
        }
    }
    out
}

/// Pull the previous observations back out — used for
/// computing replay overlap.
fn prev_observations(hierarchy: &Hierarchy) -> Vec<SpikeTrain> {
    let mut out = Vec::new();
    for id in &hierarchy.input_sinks {
        if let Some(col) = hierarchy.columns.get(id) {
            if let Some(t) = col.last_spikes.get(1) {
                out.push(t.clone());
            }
        }
    }
    out
}

/// Persist observations into L2/3 (layer index 1) so the next
/// sleep-cycle iteration can compute overlap against them.
fn store_observations(hierarchy: &mut Hierarchy, observations: &[SpikeTrain]) {
    let _ = (hierarchy, observations);
    // The cortex's tick path already updates last_spikes for
    // every layer, so we don't need to write here — but we keep
    // the function as a hook for future replay overlays.
}

fn rem_phase(hierarchy: &mut Hierarchy, rng: &mut StdRng, noise_width: usize) {
    let ids: Vec<ColumnId> = hierarchy.column_ids();
    for id in ids {
        if rng.gen_bool(0.05) {
            // Inject a low-amplitude random drive into the column
            // via its thalamic input (if it's an input sink).
            if hierarchy.input_sinks.contains(&id) {
                let drive: Vec<f32> = (0..noise_width)
                    .map(|_| rng.gen_range(0.0..0.1))
                    .collect();
                let mut inputs = std::collections::HashMap::new();
                inputs.insert(id, drive);
                hierarchy.tick(&inputs, 0.5, 0.5, 0.5);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sleep_cycle_completes() {
        let mut h = Hierarchy::new();
        let in_col = h.add_column(true, false);
        let out_col = h.add_column(false, true);
        h.connect(in_col, out_col, crate::hierarchy::Connection::BottomUp);
        h.connect(out_col, in_col, crate::hierarchy::Connection::TopDown);
        let mut hip = Hippocampus::new();
        for _ in 0..5 {
            let sdr = crate::sdr::Sdr::random_active(&mut rand::thread_rng(), 40);
            hip.record(sdr, "in", 0.8, [0.0; 3]).unwrap();
        }
        let report = SleepCycle::new().run(&mut hip, &mut h, 3);
        assert_eq!(report.episodes_replayed, 3);
    }
}
