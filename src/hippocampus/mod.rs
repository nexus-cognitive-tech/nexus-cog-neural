//! Hippocampus — fast, capacity-unbounded episodic memory.

mod consolidation;
mod episodic;

pub use consolidation::ConsolidationReport;
pub use episodic::{Episode, Hippocampus};
