//! Hippocampus — fast, capacity-unbounded episodic memory.

mod consolidation;
pub mod episodic;

pub use consolidation::ConsolidationReport;
pub use episodic::{
    episode_metadata, Episode, EpisodeMetadata, EpisodeSink, EpisodeValidationError, Hippocampus,
    NullEpisodeSink,
};
