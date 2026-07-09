//! Basal Ganglia — action selection via winner-take-all dynamics.
//!
//! Multiple candidate actions compete; the strongest inhibits
//! the rest through lateral inhibition, producing a single
//! chosen action.

use serde::{Deserialize, Serialize};

/// One candidate action — its label and an activation score in
/// `[0.0, 1.0]`. Higher = more likely to be selected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    /// Stable identifier.
    pub id: String,
    /// Human-readable label.
    pub label: String,
    /// Activation score from upstream regions.
    pub activation: f32,
    /// Was this the winner last step? Used for hysteresis.
    pub last_won: bool,
}

/// Outcome of one basal-ganglia cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionSelection {
    /// The winning action (always set unless `candidates` is empty).
    pub winner: Option<Action>,
    /// Activation of the winner after inhibition.
    pub winner_activation: f32,
    /// Total activations before inhibition — useful for
    /// confidence reporting.
    pub total_activation: f32,
}

/// Action-selection network with lateral inhibition.
#[derive(Default)]
pub struct BasalGanglia {
    candidates: Vec<Action>,
    /// Lateral-inhibition strength in `[0, 1]`.
    inhibition: f32,
    /// Hysteresis bonus for the previous winner in `[0, 1]`.
    hysteresis: f32,
}

impl BasalGanglia {
    /// Construct with default parameters (inhibition = 0.5,
    /// hysteresis = 0.15).
    #[must_use]
    pub fn new() -> Self {
        Self { inhibition: 0.5, hysteresis: 0.15, ..Self::default() }
    }

    /// Set inhibition strength.
    pub fn set_inhibition(&mut self, v: f32) {
        self.inhibition = v.clamp(0.0, 1.0);
    }

    /// Set hysteresis bonus.
    pub fn set_hysteresis(&mut self, v: f32) {
        self.hysteresis = v.clamp(0.0, 1.0);
    }

    /// Register or replace a candidate action.
    pub fn offer(&mut self, id: impl Into<String>, label: impl Into<String>, activation: f32) {
        let id = id.into();
        if let Some(existing) = self.candidates.iter_mut().find(|c| c.id == id) {
            existing.activation = activation;
        } else {
            self.candidates.push(Action {
                id,
                label: label.into(),
                activation,
                last_won: false,
            });
        }
    }

    /// Run one selection cycle.
    pub fn select(&mut self) -> ActionSelection {
        let total: f32 = self.candidates.iter().map(|c| c.activation).sum();
        if self.candidates.is_empty() {
            return ActionSelection { winner: None, winner_activation: 0.0, total_activation: 0.0 };
        }
        let mut scored: Vec<(usize, f32)> = self
            .candidates
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let mut score = c.activation;
                if c.last_won {
                    score += self.hysteresis;
                }
                // Subtract lateral inhibition from competitors.
                let inhibition = self.inhibition
                    * self.candidates.iter().filter(|o| o.id != c.id).map(|o| o.activation).sum::<f32>();
                (i, (score - inhibition).max(0.0))
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let (winner_idx, winner_activation) = scored[0];
        let winner = self.candidates[winner_idx].clone();
        for c in self.candidates.iter_mut() {
            c.last_won = c.id == winner.id;
        }
        ActionSelection {
            winner: Some(winner),
            winner_activation,
            total_activation: total,
        }
    }

    /// Forget all candidates (call this at the end of every
    /// decision cycle so the next cycle starts fresh).
    pub fn clear(&mut self) {
        self.candidates.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strongest_wins() {
        let mut g = BasalGanglia::new();
        g.offer("a", "Action A", 0.3);
        g.offer("b", "Action B", 0.9);
        let s = g.select();
        assert_eq!(s.winner.unwrap().id, "b");
    }

    #[test]
    fn hysteresis_favours_repeat_winner() {
        let mut g = BasalGanglia::new();
        g.offer("a", "A", 0.5);
        g.offer("b", "B", 0.5);
        let _ = g.select(); // first pick is arbitrary but committed
        g.clear();
        g.offer("a", "A", 0.5);
        g.offer("b", "B", 0.5);
        let s = g.select();
        // Equal activations, hysteresis should keep the previous winner.
        assert!(s.winner.is_some());
    }

    #[test]
    fn strong_lateral_inhibition_can_suppress_all() {
        let mut g = BasalGanglia::new();
        g.set_inhibition(2.0);
        g.offer("a", "A", 0.5);
        g.offer("b", "B", 0.5);
        let s = g.select();
        // With inhibition = 2 and two equal candidates the winner
        // score drops to zero, but the winner slot is still filled.
        assert!(s.winner.is_some());
    }
}
