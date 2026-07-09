//! Consolidation report — outcome of one sleep cycle.

use serde::{Deserialize, Serialize};

/// Statistics returned after [`crate::sleep::SleepCycle::run`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConsolidationReport {
    /// Number of hippocampal episodes replayed during the cycle.
    pub episodes_replayed: usize,
    /// Number of unique SDRs that were re-injected into cortex.
    pub unique_patterns: usize,
    /// Average overlap between replayed patterns and the
    /// cortical regions they targeted. Higher = better
    /// consolidation.
    pub avg_target_overlap: f32,
    /// Total wall-time the cycle took, in milliseconds.
    pub elapsed_ms: u128,
}
