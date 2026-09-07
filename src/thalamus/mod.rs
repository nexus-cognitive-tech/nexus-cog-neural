//! Thalamus — sensory relay + gating.
//!
//! Every input to cortex passes through the thalamus. The thalamus
//! decides — based on bottom-up salience, top-down attention bias
//! and the current arousal level — whether to forward, attenuate
//! or completely block each input channel.
//!
//! This module is a stub; full implementation lives in `thalamus.rs`.

#![allow(dead_code)]

mod relay;

pub use relay::{GatingDecision, Thalamus, ThalamusChannel};
