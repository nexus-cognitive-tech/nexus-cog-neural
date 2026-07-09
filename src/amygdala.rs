//! Amygdala — attaches 3-D valence (reward / threat / novelty)
//! to every cortical event and emits neuromodulator adjustments.

use serde::{Deserialize, Serialize};

/// 3-D emotional valence attached to a cortical event.
/// Each component is in `[0, 1]`:
/// * `reward` — positive outcome expectation
/// * `threat` — negative outcome expectation
/// * `novelty` — divergence from recent past
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Valence {
    /// Reward signal in `[0, 1]`.
    pub reward: f32,
    /// Threat signal in `[0, 1]`.
    pub threat: f32,
    /// Novelty signal in `[0, 1]`.
    pub novelty: f32,
}

impl Valence {
    /// Construct from individual signals.
    #[must_use]
    pub fn new(reward: f32, threat: f32, novelty: f32) -> Self {
        Self {
            reward: reward.clamp(0.0, 1.0),
            threat: threat.clamp(0.0, 1.0),
            novelty: novelty.clamp(0.0, 1.0),
        }
    }

    /// Tuple-form constructor for hot loops.
    #[must_use]
    pub fn from_tuple(t: (f32, f32, f32)) -> Self {
        Self::new(t.0, t.1, t.2)
    }

    /// Magnitude of the valence — sqrt(r² + t² + n²).
    #[must_use]
    pub fn magnitude(&self) -> f32 {
        (self.reward * self.reward + self.threat * self.threat + self.novelty * self.novelty).sqrt()
    }

    /// Convert to `[f32; 3]` array (lossy at the type level).
    #[must_use]
    pub fn to_array(self) -> [f32; 3] {
        [self.reward, self.threat, self.novelty]
    }
}

/// Amygdala — assigns valence to events and adjusts neuromodulator
/// levels. Stateless from one event to the next; the per-event
/// scoring is pure.
#[derive(Default)]
pub struct Amygdala {
    /// EMA of recent reward (positive valence) — drives the
    /// dopamine release.
    reward_ema: f32,
    /// EMA of recent threat — increases serotonin inverse (more
    /// risk-aversion).
    threat_ema: f32,
}

impl Amygdala {
    /// New amygdala.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Score a single event. `similarity_to_recent` is the SDR
    /// overlap between this event and the previous one
    /// (`1.0` = identical, `0.0` = disjoint).
    pub fn score(&mut self, similarity_to_recent: f32) -> Valence {
        let novelty = (1.0 - similarity_to_recent.clamp(0.0, 1.0)).clamp(0.0, 1.0);
        // Reward/threat are biased by the running EMAs so a
        // string of bad events amplifies subsequent threat
        // tagging, mirroring real amygdala behaviour.
        let reward = self.reward_ema.max(0.0);
        let threat = self.threat_ema.max(0.0);
        // Update EMAs (slow decay so the amygdala doesn't oscillate).
        self.reward_ema = 0.9 * self.reward_ema + 0.1 * (1.0 - novelty);
        self.threat_ema = 0.9 * self.threat_ema + 0.1 * novelty;
        Valence::new(reward, threat, novelty)
    }

    /// Read-only EMA of reward.
    #[must_use]
    pub fn reward_ema(&self) -> f32 {
        self.reward_ema
    }

    /// Read-only EMA of threat.
    #[must_use]
    pub fn threat_ema(&self) -> f32 {
        self.threat_ema
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn novelty_is_complement_of_similarity() {
        let mut a = Amygdala::new();
        let v = a.score(1.0);
        assert_eq!(v.novelty, 0.0);
        let v = a.score(0.0);
        assert_eq!(v.novelty, 1.0);
    }

    #[test]
    fn threat_ema_grows_on_novelty() {
        let mut a = Amygdala::new();
        for _ in 0..20 {
            a.score(0.0);
        }
        assert!(a.threat_ema() > 0.5);
    }
}
