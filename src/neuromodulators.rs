//! Neuromodulators — global chemical signals that tune the
//! entire system. Each is an EMA in `[0, 1]`.

use serde::{Deserialize, Serialize};

/// Dopamine — reward prediction error. Higher = more learning,
/// more exploration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Dopamine {
    /// Current dopamine level.
    pub level: f32,
    /// Baseline (homeostatic target).
    pub baseline: f32,
}

impl Dopamine {
    /// Inject a reward signal; updates the EMA towards
    /// `baseline + reward - prediction_error`.
    pub fn update(&mut self, reward: f32, prediction: f32) {
        let error = (reward - prediction).clamp(-1.0, 1.0);
        let target = (self.baseline + error).clamp(0.0, 1.0);
        self.level = 0.9 * self.level + 0.1 * target;
    }
}

/// Serotonin — patience / risk-aversion. Higher = slower
/// exploration, longer-term credit assignment.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Serotonin {
    /// Current serotonin level.
    pub level: f32,
    /// Baseline target.
    pub baseline: f32,
}

impl Serotonin {
    /// Inject a satisfaction signal (lower satisfaction = lower
    /// serotonin = more risk-taking).
    pub fn update(&mut self, satisfaction: f32) {
        let target = (self.baseline + 0.2 * (satisfaction - 0.5)).clamp(0.0, 1.0);
        self.level = 0.95 * self.level + 0.05 * target;
    }
}

/// Norepinephrine — arousal / vigilance. Higher = wider
/// attention spotlight, faster but less precise.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Norepinephrine {
    /// Current arousal level.
    pub level: f32,
    /// Baseline target.
    pub baseline: f32,
}

impl Norepinephrine {
    /// Update from observed environmental urgency.
    pub fn update(&mut self, urgency: f32) {
        let target = (self.baseline + 0.3 * (urgency - 0.5)).clamp(0.0, 1.0);
        self.level = 0.85 * self.level + 0.15 * target;
    }
}

/// Aggregated neuromodulator state.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Neuromodulators {
    /// Reward / learning signal.
    pub dopamine: Dopamine,
    /// Patience signal.
    pub serotonin: Serotonin,
    /// Arousal signal.
    pub norepinephrine: Norepinephrine,
}

impl Neuromodulators {
    /// New system with default baselines (0.5).
    #[must_use]
    pub fn new() -> Self {
        Self {
            dopamine: Dopamine { baseline: 0.5, ..Default::default() },
            serotonin: Serotonin { baseline: 0.5, ..Default::default() },
            norepinephrine: Norepinephrine { baseline: 0.5, ..Default::default() },
        }
    }

    /// Apply a single valence vector to all three modulators.
    pub fn apply_valence(&mut self, reward: f32, threat: f32, novelty: f32) {
        self.dopamine.update(reward, self.dopamine.baseline);
        self.serotonin.update(1.0 - threat);
        self.norepinephrine.update(novelty);
    }

    /// Effective learning rate multiplier in `[0, 2]`. Dopamine
    /// above baseline speeds learning up; below baseline slows it.
    #[must_use]
    pub fn learning_rate_multiplier(&self) -> f32 {
        (1.0 + (self.dopamine.level - self.dopamine.baseline) * 2.0).clamp(0.1, 2.0)
    }

    /// Effective attention width in `[0.1, 1.0]` — high
    /// norepinephrine widens the spotlight.
    #[must_use]
    pub fn attention_width(&self) -> f32 {
        (0.3 + 0.7 * self.norepinephrine.level).clamp(0.1, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dopamine_tracks_error() {
        let mut d = Dopamine { baseline: 0.5, ..Default::default() };
        for _ in 0..20 {
            d.update(1.0, 0.0);
        }
        assert!(d.level > 0.5, "positive reward should raise dopamine, got {}", d.level);
        for _ in 0..20 {
            d.update(0.0, 0.8);
        }
        assert!(d.level < 0.5, "negative prediction error should lower dopamine, got {}", d.level);
    }

    #[test]
    fn learning_rate_clamped() {
        let mut n = Neuromodulators::new();
        n.dopamine.level = 10.0;
        assert!(n.learning_rate_multiplier() <= 2.0);
        n.dopamine.level = -10.0;
        assert!(n.learning_rate_multiplier() >= 0.1);
    }
}
