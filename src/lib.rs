//! # nexus-cog-neural
//!
//! Brain-like cognitive architecture. Every cognitive function in
//! `nexus-cog-cli` (palace, brain, cognitive, intel, intent) is
//! rerouted through [`Cortex`] — a single orchestrator that holds:
//!
//! * an [`Hierarchy`] of [`CorticalColumn`]s with full bottom-up,
//!   top-down and lateral recurrence;
//! * a [`SpikingPopulation`] per cortical layer — spike trains
//!   and first-spike-latency coding replace the legacy SDR
//!   activation;
//! * [`SynapticFanOut`] projections with three-factor Hebbian
//!   plasticity (LTP / LTD) gated by per-synapse neuromodulators;
//! * an [`AstrocyteNetwork`] providing calcium-wave coupling and
//!   gliotransmitter release that boosts plasticity;
//! * a [`Neurogenesis`] controller for column birth and apoptosis;
//! * a [`SensorimotorLoop`] closing the perception → action →
//!   perception cycle;
//! * a [`Hippocampus`] for fast episodic memory, a [`Sleep`]
//!   cycle, a [`BasalGanglia`] for action selection, an
//!   [`Amygdala`] for valence tagging, a [`WorkingMemory`] of
//!   7±2 slots, [`Neuromodulators`], [`Attention`] and a
//!   [`GlobalWorkspace`].
//!
//! ## Building blocks
//!
//! | Subsystem | Module |
//! | --- | --- |
//! | SDR + encoders | [`sdr`] (legacy; still used as a hash-based text → SDR helper) |
//! | Spike coding | [`spike`] |
//! | Synapse + plasticity | [`synapse`] |
//! | Astrocyte network | [`astrocyte`] |
//! | Neurogenesis | [`neurogenesis`] |
//! | 6-layer column | [`cortical_column`] |
//! | Recurrent hierarchy | [`hierarchy`] |
//! | Sensorimotor loop | [`sensorimotor`] |
//! | Thalamus | [`thalamus`] |
//! | Hippocampus | [`hippocampus`] |
//! | Basal ganglia | [`basal_ganglia`] |
//! | Amygdala | [`amygdala`] |
//! | Working memory | [`working_memory`] |
//! | Attention | [`attention`] |
//! | Neuromodulators | [`neuromodulators`] |
//! | Global workspace | [`global_workspace`] |
//! | Sleep cycle | [`sleep`] |
//! | Replay buffer | [`replay`] |
//! | Cortex orchestrator | [`cortex`] |

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod amygdala;
pub mod astrocyte;
pub mod attention;
pub mod basal_ganglia;
pub mod cortex;
pub mod cortical_column;
pub mod global_workspace;
pub mod hierarchy;
pub mod hippocampus;
pub mod neurogenesis;
pub mod neuromodulators;
pub mod replay;
pub mod sdr;
pub mod sensorimotor;
pub mod sleep;
pub mod spike;
pub mod synapse;
pub mod thalamus;
pub mod working_memory;

// Convenience re-exports for embedding shells.
pub use astrocyte::{Astrocyte, AstrocyteNetwork};
pub use cortex::{Cortex, CortexConfig, CortexStats, ThoughtBroadcast};
pub use cortical_column::{CorticalColumn, CorticalLayer, LayerSpec};
pub use global_workspace::{Coalition, GlobalWorkspace};
pub use hierarchy::{ColumnId, Connection, Hierarchy};
pub use hippocampus::{
    episode_metadata, ConsolidationReport, Episode, EpisodeMetadata, EpisodeSink,
    EpisodeValidationError, Hippocampus, NullEpisodeSink,
};
pub use neurogenesis::{Neurogenesis, NeurogenesisConfig, NeurogenicVerdict};
pub use neuromodulators::Neuromodulators;
pub use replay::{ActivationMap, ModulatorSnapshot, ReplayBuffer, ReplayFrame};
pub use sdr::{
    CategoryEncoder, CoordinateEncoder, DateEncoder, Encoder, LogEncoder, ScalarEncoder, Sdr,
    SdrEncoder, SdrStats, SequenceEncoder, DEFAULT_SPARSITY, SDR_WIDTH,
};
pub use sensorimotor::{LoopPhase, PredictionError, SensorimotorLoop};
pub use spike::{SpikeTrain, SpikingPopulation, SPIKE_WINDOW};
pub use synapse::{NeuromodulatorLocal, PlasticityRule, Synapse, SynapticFanOut, ThreeFactorHebbianRule};
pub use thalamus::{GatingDecision, Thalamus, ThalamusChannel};
