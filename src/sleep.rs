//! Sleep cycle — NREM (replay + consolidation) and REM
//! (cortical re-activation).
//!
//! A sleep cycle is a single call to [`SleepCycle::run`].
//! Internally it:
//!
//! 1. **NREM phase** — replays the hippocampus' highest-salience
//!    episodes back into the cortical hierarchy, biasing the
//!    spatial pooler to consolidate co-active patterns.
//! 2. **REM phase** — re-activates quiet cortical columns with
//!    random SDRs to prevent over-fitting.

use crate::hippocampus::{ConsolidationReport, Hippocampus};
use crate::region::{Hierarchy, RegionId};
use crate::sdr::Sdr;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Sleep cycle.
#[derive(Default)]
pub struct SleepCycle;

impl SleepCycle {
    /// New sleep cycle.
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
        let mut rng = StdRng::seed_from_u64(chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0) as u64);
        for ep in &episodes {
            let (overlap, unique) = replay_into_hierarchy(hierarchy, &ep.sdr, &mut rng);
            total_overlap += overlap;
            if unique {
                unique_patterns += 1;
            }
        }
        // REM phase: perturb each region with random noise so
        // quiet columns stay in the running.
        rem_phase(hierarchy, &mut rng);
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

fn replay_into_hierarchy(
    hierarchy: &mut Hierarchy,
    sdr: &Sdr,
    rng: &mut StdRng,
) -> (f32, bool) {
    // Treat the first region as the consolidation target.
    let _ = rng;
    let ids: Vec<RegionId> = (0..hierarchy.len() as u32).map(RegionId).collect();
    let Some(target) = ids.first().copied() else {
        return (0.0, false);
    };
    let mut inputs = std::collections::HashMap::new();
    inputs.insert(target, sdr.clone());
    hierarchy.tick(&inputs);
    hierarchy
        .region(target)
        .map(|r| {
            let pred = r.predict(sdr);
            let ov = crate::sdr::semantic_similarity(sdr, &pred);
            (ov, ov > 0.6)
        })
        .unwrap_or((0.0, false))
}

fn rem_phase(hierarchy: &mut Hierarchy, rng: &mut StdRng) {
    for id in 0..hierarchy.len() {
        let rid = RegionId(id as u32);
        if let Some(region) = hierarchy.region_mut(rid) {
            // Touch `_stats` to keep it visible after the
            // perturbation pass.
            let _ = region.stats();
        }
        // Tiny random perturbation — keeps the spatial pooler
        // exploring without disrupting established patterns.
        if rng.gen_bool(0.05) {
            let noise = Sdr::random_active(rng, 5);
            let mut inputs = std::collections::HashMap::new();
            inputs.insert(rid, noise);
            hierarchy.tick(&inputs);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdr::SDR_WIDTH;

    #[test]
    fn sleep_cycle_completes() {
        let mut h = Hierarchy::new();
        h.add_region("V1", SDR_WIDTH, 1);
        let mut hip = Hippocampus::new();
        for _ in 0..5 {
            hip.record(Sdr::random_active(&mut rand::thread_rng(), 40), "V1", 0.8, [0.0; 3])
                .unwrap();
        }
        let report = SleepCycle::new().run(&mut hip, &mut h, 3);
        assert_eq!(report.episodes_replayed, 3);
    }
}
