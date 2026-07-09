//! A single cortical region — `SpatialPooler ∘ TemporalMemory`.
//!
//! The region's job is to convert its bottom-up input into a
//! stable SDR (via the SP) and then produce a top-down prediction
//! of the next step (via the TM).

use super::spatial_pooler::{SpatialPooler, SpatialPoolerParams};
use super::temporal_memory::TemporalMemory;
use crate::sdr::{Sdr, SDR_WIDTH};
use serde::{Deserialize, Serialize};

/// Stable identifier of a region inside a [`crate::region::Hierarchy`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RegionId(pub u32);

/// Per-region stats exposed for diagnostics and the Studio UI.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RegionStats {
    /// Number of `compute` calls so far.
    pub steps: u64,
    /// Average overlap between consecutive output SDRs.
    pub avg_step_overlap: f32,
    /// Average prediction accuracy over the last 16 steps.
    pub prediction_accuracy: f32,
}

/// One cortical region.
pub struct Region {
    /// Stable id (set by the parent [`crate::region::Hierarchy`]).
    pub id: RegionId,
    /// Human-readable name.
    pub name: String,
    /// Width of the bottom-up input. Set by the hierarchy on
    /// insertion.
    pub input_width: usize,
    sp: SpatialPooler,
    tm: TemporalMemory,
    stats: RegionStats,
}

impl Region {
    /// Build a new region. `input_width` is the SDR width of its
    /// bottom-up input; `seed` randomises permanences.
    pub fn new(id: RegionId, name: impl Into<String>, input_width: usize, seed: u64) -> Self {
        Self {
            id,
            name: name.into(),
            input_width,
            sp: SpatialPooler::new(SpatialPoolerParams::default_for(input_width), seed),
            tm: TemporalMemory::new(),
            stats: RegionStats::default(),
        }
    }

    /// Run the region on one bottom-up input. Returns the SDR
    /// that should be sent up to the next region.
    pub fn step(&mut self, input: &Sdr) -> Sdr {
        let active = self.sp.compute(input);
        let (predicted, _) = self.tm.learn(active.clone());
        let overlap = crate::sdr::semantic_similarity(&predicted, &active);
        self.stats.avg_step_overlap =
            0.95 * self.stats.avg_step_overlap + 0.05 * overlap;
        self.stats.prediction_accuracy =
            0.95 * self.stats.prediction_accuracy + 0.05 * overlap;
        self.stats.steps += 1;
        active
    }

    /// Predict the next SDR without updating any state.
    pub fn predict(&self, active: &Sdr) -> Sdr {
        self.tm.predict(active)
    }

    /// Read-only stats.
    pub fn stats(&self) -> &RegionStats {
        &self.stats
    }

    /// Width of the region's output (= SDR_WIDTH).
    pub fn output_width(&self) -> usize {
        SDR_WIDTH
    }
}
