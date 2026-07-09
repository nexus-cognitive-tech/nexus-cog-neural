//! Attention — top-down + bottom-up spotlight.
//!
//! Combines salience (bottom-up) with goal bias (top-down) into a
//! single attention map that gates thalamic inputs and biases
//! which cortical columns are eligible to win.

use crate::sdr::Sdr;
use serde::{Deserialize, Serialize};

/// Per-channel attention bias.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AttentionMap {
    /// One bias per channel, in `[0, 1]`.
    pub biases: Vec<f32>,
    /// One salience per channel (rolling EMA of input salience).
    pub saliences: Vec<f32>,
}

impl AttentionMap {
    /// Effective bias for a channel: 0.6 × salience + 0.4 × top-down.
    #[must_use]
    pub fn effective(&self, channel: usize) -> f32 {
        let sal = self.saliences.get(channel).copied().unwrap_or(0.0);
        let bias = self.biases.get(channel).copied().unwrap_or(0.0);
        (0.6 * sal + 0.4 * bias).clamp(0.0, 1.0)
    }
}

/// Attention controller.
#[derive(Default)]
pub struct Attention {
    /// Current per-channel attention map.
    pub map: AttentionMap,
    /// Goal SDR — top-down bias derived from the active thought.
    pub goal: Option<Sdr>,
    /// EMA rate for salience updates.
    pub salience_ema: f32,
}

impl Attention {
    /// New attention controller. `n_channels` must match the
    /// thalamus.
    #[must_use]
    pub fn new(n_channels: usize) -> Self {
        Self {
            map: AttentionMap {
                biases: vec![0.0; n_channels],
                saliences: vec![0.0; n_channels],
            },
            salience_ema: 0.1,
            ..Self::default()
        }
    }

    /// Set the current top-down goal.
    pub fn set_goal(&mut self, goal: Sdr) {
        self.goal = Some(goal);
    }

    /// Update salience for one channel.
    pub fn observe_salience(&mut self, channel: usize, salience: f32) {
        if let Some(s) = self.map.saliences.get_mut(channel) {
            let r = self.salience_ema;
            *s = (1.0 - r) * (*s) + r * salience;
        }
    }

    /// Update top-down bias for one channel based on overlap
    /// between the goal SDR and a per-channel prototype.
    pub fn bias_against_prototype(&mut self, channel: usize, prototype: &Sdr) {
        let sim = match &self.goal {
            Some(g) => crate::sdr::semantic_similarity(g, prototype),
            None => 0.0,
        };
        if let Some(b) = self.map.biases.get_mut(channel) {
            *b = sim.clamp(0.0, 1.0);
        }
    }

    /// Current effective bias for one channel.
    #[must_use]
    pub fn effective_bias(&self, channel: usize) -> f32 {
        self.map.effective(channel)
    }

    /// SDR representing the current focus of attention — the
    /// union of the top-k channels by effective bias.
    pub fn focus_sdr(&self, top_k: usize) -> Sdr {
        let mut scored: Vec<(usize, f32)> = (0..self.map.biases.len())
            .map(|i| (i, self.map.effective(i)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let chosen: Vec<usize> = scored.into_iter().take(top_k).map(|(i, _)| i).collect();
        let mut bias_indices: Vec<usize> = Vec::new();
        for (i, &c) in chosen.iter().enumerate() {
            let start = (i * (crate::sdr::SDR_WIDTH / chosen.len().max(1)));
            bias_indices.extend(start..start + crate::sdr::SDR_WIDTH / chosen.len().max(1));
            bias_indices.push(c);
        }
        bias_indices.sort_unstable();
        bias_indices.dedup();
        Sdr::from_bits(bias_indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_down_bias_from_goal() {
        let mut a = Attention::new(2);
        let goal = Sdr::from_bits([10, 11, 12]);
        a.set_goal(goal);
        let prototype = Sdr::from_bits([10, 11, 12, 99, 100]);
        a.bias_against_prototype(0, &prototype);
        // Bias is half of the effective signal (top-down = 40% in
        // effective()) so the bias itself should be > 0.5.
        assert!(a.map.biases[0] > 0.5);
    }

    #[test]
    fn focus_sdr_covers_top_k() {
        let mut a = Attention::new(10);
        for i in 0..10 {
            a.map.biases[i] = i as f32 / 10.0;
        }
        let focus = a.focus_sdr(3);
        assert!(focus.active_count() > 0);
    }
}
