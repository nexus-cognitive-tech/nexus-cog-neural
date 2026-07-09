//! # nexus-cog-neural
//!
//! Brain-like cognitive architecture inspired by Numenta's HTM,
//! Global Workspace Theory, basal-ganglia winner-take-all dynamics,
//! and hippocampal-cortical consolidation during sleep.
//!
//! Every cognitive function in `nexus-cog-cli` (palace, brain,
//! cognitive, intel, intent) is rerouted through [`Cortex`] — a
//! single orchestrator that holds:
//!
//! * an HTM-style [`Hierarchy`] of cortical regions,
//! * a [`Thalamus`] that gates sensory inputs by salience,
//! * a fast [`Hippocampus`] for episodic memory,
//! * a [`WorkingMemory`] of 7±2 SDR slots,
//! * an [`Attention`] spotlight (top-down + bottom-up),
//! * a [`Neuromodulators`] panel (dopamine / serotonin / norepinephrine),
//! * an [`Amygdala`] that tags every event with valence,
//! * a [`GlobalWorkspace`] for coalition selection,
//! * a [`BasalGanglia`] for action selection,
//! * a [`SleepCycle`] for NREM/REM consolidation,
//! * a [`ReplayBuffer`] for thought-chain recording and Studio
//!   visualisation.
//!
//! ## Building blocks
//!
//! * [`Sdr`] — 2048-bit sparse distributed representation. Every
//!   state in the brain is one of these.
//! * [`SpatialPooler`] — turns dense/sparse input into a stable
//!   SDR.
//! * [`TemporalMemory`] — predicts the next SDR given a history.
//!
//! ## Top-level
//!
//! ```no_run
//! use nexus_cog_neural::Cortex;
//! use std::collections::HashMap;
//! let mut cortex = Cortex::default_for_tests();
//! let mut inputs = HashMap::new();
//! inputs.insert("channel.0".to_string(), nexus_cog_neural::Sdr::from_bits([1, 2, 3]));
//! let broadcast = cortex.tick(inputs);
//! println!("{} regions competed, winner: {:?}",
//!          broadcast.coalition.members.len(),
//!          broadcast.chosen_action);
//! ```

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod amygdala;
pub mod attention;
pub mod basal_ganglia;
pub mod cortex;
pub mod global_workspace;
pub mod hippocampus;
pub mod neuromodulators;
pub mod region;
pub mod replay;
pub mod sdr;
pub mod sleep;
pub mod thalamus;
pub mod working_memory;

pub use cortex::{Cortex, CortexConfig, CortexStats, ThoughtBroadcast};
pub use global_workspace::{Coalition, GlobalWorkspace};
pub use hippocampus::{ConsolidationReport, Episode, Hippocampus};
pub use region::{Hierarchy, Region, RegionId, RegionStats};
pub use replay::{ActivationMap, ModulatorSnapshot, ReplayBuffer, ReplayFrame};
pub use sdr::{
    CategoryEncoder, CoordinateEncoder, DateEncoder, Encoder, LogEncoder, ScalarEncoder, Sdr,
    SdrEncoder, SdrStats, SequenceEncoder, DEFAULT_SPARSITY, SDR_WIDTH,
};
pub use thalamus::{GatingDecision, Thalamus, ThalamusChannel};
