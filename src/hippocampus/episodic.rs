//! Fast episodic store — every notable cortical event becomes an
//! [`Episode`] that can later be replayed during sleep.

use crate::sdr::Sdr;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single hippocampal episode: an SDR plus the contextual
/// metadata needed to replay it usefully.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    /// Stable id (sequential within a single hippocampus).
    pub id: u64,
    /// SDR snapshot of the cortical event.
    pub sdr: Sdr,
    /// Source label (region name, "thalamus.audio", etc.).
    pub source: String,
    /// Unix timestamp of the episode.
    pub timestamp: i64,
    /// Salience at the time of encoding — only episodes above the
    /// salience floor are retained.
    pub salience: f32,
    /// 3-D emotional valence (reward / threat / novelty).
    pub valence: [f32; 3],
    /// Number of times this episode has been replayed during sleep.
    pub replay_count: u32,
}

/// Hippocampal store. Append-only on write; replay-driven read
/// during sleep.
#[derive(Default)]
pub struct Hippocampus {
    pub(crate) episodes: HashMap<u64, Episode>,
    pub(crate) by_source: HashMap<String, Vec<u64>>,
    pub(crate) next_id: u64,
    /// Salience floor — episodes below this are dropped on write.
    pub salience_floor: f32,
    /// Capacity ceiling — once exceeded, the lowest-salience
    /// episode is evicted (the hippocampus doesn't grow forever).
    pub capacity: usize,
}

impl Hippocampus {
    /// New empty hippocampus with default capacity of 100,000
    /// episodes and salience floor `0.2`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            capacity: 100_000,
            salience_floor: 0.2,
            ..Self::default()
        }
    }

    /// Total stored episodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.episodes.len()
    }

    /// True if no episodes are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.episodes.is_empty()
    }

    /// Record an episode. Returns `Some(id)` on success,
    /// `None` if the episode was rejected (below salience floor).
    pub fn record(
        &mut self,
        sdr: Sdr,
        source: impl Into<String>,
        salience: f32,
        valence: [f32; 3],
    ) -> Option<u64> {
        if salience < self.salience_floor {
            return None;
        }
        if self.episodes.len() >= self.capacity {
            self.evict_lowest_salience();
        }
        let id = self.next_id;
        self.next_id += 1;
        let ep = Episode {
            id,
            sdr,
            source: source.into(),
            timestamp: chrono::Utc::now().timestamp(),
            salience,
            valence,
            replay_count: 0,
        };
        self.by_source.entry(ep.source.clone()).or_default().push(id);
        self.episodes.insert(id, ep);
        Some(id)
    }

    /// Look up an episode by id.
    pub fn get(&self, id: u64) -> Option<&Episode> {
        self.episodes.get(&id)
    }

    /// Iterate every stored episode, in arbitrary order.
    pub fn episodes(&self) -> impl Iterator<Item = &Episode> {
        self.episodes.values()
    }

    /// Sort every stored episode by timestamp descending.
    pub fn episodes_sorted_by_recency(&self) -> Vec<Episode> {
        let mut v: Vec<Episode> = self.episodes.values().cloned().collect();
        v.sort_by_key(|e| std::cmp::Reverse(e.timestamp));
        v
    }

    /// Iterate episodes for a given source (e.g. "thalamus.vision").
    pub fn by_source(&self, source: &str) -> impl Iterator<Item = &Episode> {
        self.by_source
            .get(source)
            .into_iter()
            .flat_map(|ids| ids.iter().filter_map(|id| self.episodes.get(id)))
    }

    /// Take the `n` most-salient episodes that have been replayed
    /// fewer than `max_replays` times. Used by the sleep cycle.
    pub fn replay_batch(&mut self, n: usize, max_replays: u32) -> Vec<Episode> {
        let mut candidates: Vec<Episode> = self
            .episodes
            .values()
            .filter(|e| e.replay_count < max_replays)
            .cloned()
            .collect();
        candidates.sort_by(|a, b| b.salience.partial_cmp(&a.salience).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(n);
        for ep in &mut candidates {
            if let Some(stored) = self.episodes.get_mut(&ep.id) {
                stored.replay_count += 1;
                ep.replay_count = stored.replay_count;
            }
        }
        candidates
    }

    fn evict_lowest_salience(&mut self) {
        if let Some((&id, _)) = self.episodes.iter().min_by(|a, b| {
            a.1.salience.partial_cmp(&b.1.salience).unwrap_or(std::cmp::Ordering::Equal)
        }) {
            self.episodes.remove(&id);
            for ids in self.by_source.values_mut() {
                ids.retain(|i| *i != id);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn salience_floor_filters_low_events() {
        let mut h = Hippocampus::new();
        assert!(h.record(Sdr::empty(), "x", 0.1, [0.0; 3]).is_none());
        assert!(h.record(Sdr::empty(), "x", 0.5, [0.0; 3]).is_some());
    }

    #[test]
    fn capacity_evicts_lowest() {
        let mut h = Hippocampus::new();
        h.capacity = 2;
        h.record(Sdr::empty(), "low", 0.3, [0.0; 3]).unwrap(); // id 0
        h.record(Sdr::empty(), "mid", 0.5, [0.0; 3]).unwrap(); // id 1
        h.record(Sdr::empty(), "high", 0.9, [0.0; 3]).unwrap(); // id 2 → evicts 'low'
        assert_eq!(h.len(), 2);
        assert!(h.get(0).is_none(), "low-salience id 0 should be evicted");
        assert!(h.get(1).is_some());
        assert!(h.get(2).is_some());
    }

    #[test]
    fn replay_batch_increments_counter() {
        let mut h = Hippocampus::new();
        h.record(Sdr::empty(), "x", 0.9, [0.0; 3]).unwrap();
        let batch = h.replay_batch(5, 10);
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].replay_count, 1);
    }
}
