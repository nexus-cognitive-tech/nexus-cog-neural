//! Thalamic relay implementation. See `mod.rs` for an overview.

use crate::sdr::Sdr;

/// Per-channel state inside the thalamus.
#[derive(Debug, Clone)]
pub struct ThalamusChannel {
    /// Stable id of the channel (e.g. `"vision"`, `"audio"`).
    pub name: String,
    /// Current salience accumulator — combines bottom-up novelty
    /// with top-down attention bias.
    pub salience: f32,
    /// Last input SDR, kept for replay.
    pub last_input: Option<Sdr>,
    /// Last gating decision.
    pub last_decision: GatingDecision,
}

/// Outcome of a thalamic gating decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatingDecision {
    /// Forwarded verbatim to cortex.
    Forward,
    /// Attenuated (active bits scaled down).
    Attenuate,
    /// Completely blocked — cortex never sees this input.
    Block,
}

/// Sensory relay + gating network.
#[derive(Debug, Default)]
pub struct Thalamus {
    channels: Vec<ThalamusChannel>,
}

impl Thalamus {
    /// Create a new empty thalamus.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a sensory channel.
    pub fn add_channel(&mut self, name: impl Into<String>) -> u32 {
        let id = self.channels.len() as u32;
        self.channels.push(ThalamusChannel {
            name: name.into(),
            salience: 0.0,
            last_input: None,
            last_decision: GatingDecision::Block,
        });
        id
    }

    /// Number of registered channels.
    #[must_use]
    pub fn n_channels(&self) -> usize {
        self.channels.len()
    }

    /// Read-only view of a channel by id.
    #[must_use]
    pub fn channel(&self, id: u32) -> Option<&ThalamusChannel> {
        self.channels.get(id as usize)
    }

    /// Feed an input through the thalamus. Returns the SDR that
    /// should enter cortex (or `None` if the channel is fully
    /// blocked). Updates the channel's salience accumulator.
    pub fn relay(&mut self, id: u32, input: Sdr, arousal: f32, attention_bias: f32) -> Option<Sdr> {
        let channel = self.channels.get_mut(id as usize)?;
        let novelty = novelty_score(&input, channel.last_input.as_ref());
        let instantaneous = 0.6 * novelty + 0.4 * attention_bias + 0.2 * arousal;
        channel.salience = 0.7 * channel.salience + 0.3 * instantaneous;
        // Use the instantaneous salience for the gating decision
        // so the very first input isn't attenuated by the empty
        // EMA history. The EMA still tracks the running level.
        let decision = if instantaneous < 0.15 {
            GatingDecision::Block
        } else if instantaneous < 0.4 {
            GatingDecision::Attenuate
        } else {
            GatingDecision::Forward
        };
        channel.last_decision = decision;
        channel.last_input = Some(input.clone());
        match decision {
            GatingDecision::Forward => Some(input),
            GatingDecision::Attenuate => Some(attenuate(input, 0.5)),
            GatingDecision::Block => None,
        }
    }
}

/// Squared overlap between the new input and the previous one.
/// A novel input has high novelty (low overlap), a constant input
/// has zero novelty (perfect overlap).
fn novelty_score(input: &Sdr, previous: Option<&Sdr>) -> f32 {
    let Some(prev) = previous else { return 1.0 };
    if prev.active_count() == 0 {
        return 1.0;
    }
    let overlap = crate::sdr::semantic_similarity(input, prev);
    (1.0 - overlap).clamp(0.0, 1.0)
}

fn attenuate(mut sdr: Sdr, keep_fraction: f32) -> Sdr {
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    let keep = ((sdr.active_count() as f32) * keep_fraction.clamp(0.0, 1.0)).round() as usize;
    let mut rng = StdRng::seed_from_u64(sdr.active_bits().first().copied().unwrap_or(0) as u64);
    sdr.sparsify_to(&mut rng, keep);
    sdr
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_input_always_forward() {
        let mut t = Thalamus::new();
        let id = t.add_channel("vision");
        let out = t.relay(id, Sdr::empty(), 0.5, 0.5);
        assert!(out.is_some());
        assert_eq!(t.channel(id).unwrap().last_decision, GatingDecision::Forward);
    }

    #[test]
    fn repeated_input_decays_salience() {
        let mut t = Thalamus::new();
        let id = t.add_channel("audio");
        let s = Sdr::from_bits([1, 2, 3]);
        for _ in 0..5 {
            let _ = t.relay(id, s.clone(), 0.0, 0.0);
        }
        assert_eq!(t.channel(id).unwrap().last_decision, GatingDecision::Block);
    }

    #[test]
    fn attention_bias_overrides_repetition() {
        let mut t = Thalamus::new();
        let id = t.add_channel("touched");
        let s = Sdr::from_bits([0, 1, 2]);
        // First a couple of plain inputs to decay salience.
        for _ in 0..3 {
            let _ = t.relay(id, s.clone(), 0.0, 0.0);
        }
        // Now top-down attention forces gating open.
        let out = t.relay(id, s, 0.0, 1.0);
        assert!(out.is_some(), "top-down attention should unblock the channel");
    }
}
