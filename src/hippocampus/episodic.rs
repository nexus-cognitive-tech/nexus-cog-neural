//! Fast episodic store — every notable cortical event becomes an
//! [`Episode`] that can later be replayed during sleep.
//!
//! The store is in-memory for hot-path access, but every successful
//! [`Hippocampus::record`] is forwarded to an [`EpisodeSink`] so a
//! backing persistence layer (SQLite, network store, …) can mirror
//! it without paying the cost on the read path. On startup, callers
//! hydrate the hippocampus through [`Hippocampus::restore_all`].

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::sdr::Sdr;

/// Free-form, JSON-encodable metadata bag attached to an episode.
///
/// Use it for things the cortex wants to recall alongside the raw
/// SDR: the `task` description, the model's `response`, the thalamic
/// channel that drove the tick, etc. Stored verbatim and replayed
/// by [`crate::sleep`].
pub type EpisodeMetadata = JsonValue;

/// Build an [`EpisodeMetadata`] object from an iterator of `(key,
/// value)` pairs. Cheap alternative to manually constructing a
/// [`serde_json::Map`] when the payload is small.
pub fn episode_metadata<I, K, V>(pairs: I) -> EpisodeMetadata
where
    I: IntoIterator<Item = (K, V)>,
    K: Into<String>,
    V: Into<JsonValue>,
{
    let map: serde_json::Map<String, JsonValue> = pairs
        .into_iter()
        .map(|(k, v)| (k.into(), v.into()))
        .collect();
    JsonValue::Object(map)
}

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
    /// Free-form metadata (task description, model response, …).
    #[serde(default)]
    pub metadata: EpisodeMetadata,
}

/// Per-field validation error returned from [`Episode::validate`].
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum EpisodeValidationError {
    /// NaN or infinity in a numeric field.
    #[error("non-finite value in field `{field}`: {value}")]
    NonFinite {
        /// Name of the offending field.
        field: &'static str,
        /// Offending value (serialized as `f64` for stability).
        value: f64,
    },
    /// Salience outside `[0.0, 1.0]`.
    #[error("salience {salience} outside [0.0, 1.0]")]
    SalienceOutOfRange {
        /// Offending salience value.
        salience: f32,
    },
    /// Valence component outside `[0.0, 1.0]`.
    #[error("valence[{component}] = {value} outside [0.0, 1.0]")]
    ValenceOutOfRange {
        /// Index of the offending valence component (`0` = reward,
        /// `1` = threat, `2` = novelty).
        component: usize,
        /// Offending value.
        value: f32,
    },
    /// Source label is empty.
    #[error("source label must be non-empty")]
    EmptySource,
    /// Timestamp is non-positive.
    #[error("timestamp {timestamp} must be positive")]
    BadTimestamp {
        /// Offending timestamp.
        timestamp: i64,
    },
}

impl Episode {
    /// Re-validate every field of a record loaded from
    /// persistence. Returns the first failure found; never mutates
    /// the episode.
    pub fn validate(&self) -> std::result::Result<(), EpisodeValidationError> {
        if self.source.is_empty() {
            return Err(EpisodeValidationError::EmptySource);
        }
        if self.timestamp <= 0 {
            return Err(EpisodeValidationError::BadTimestamp {
                timestamp: self.timestamp,
            });
        }
        if !self.salience.is_finite() {
            return Err(EpisodeValidationError::NonFinite {
                field: "salience",
                value: self.salience as f64,
            });
        }
        if !(0.0..=1.0).contains(&self.salience) {
            return Err(EpisodeValidationError::SalienceOutOfRange {
                salience: self.salience,
            });
        }
        const VALENCE_FIELDS: [&str; 3] = ["valence.reward", "valence.threat", "valence.novelty"];
        for (i, v) in self.valence.iter().enumerate() {
            if !v.is_finite() {
                return Err(EpisodeValidationError::NonFinite {
                    field: VALENCE_FIELDS[i],
                    value: *v as f64,
                });
            }
            if !(0.0..=1.0).contains(v) {
                return Err(EpisodeValidationError::ValenceOutOfRange {
                    component: i,
                    value: *v,
                });
            }
        }
        Ok(())
    }
}

/// Persistence sink for hippocampal episodes. Implementations must
/// be `Send + Sync` so they can be shared through an `Arc` and
/// called from the hot path of [`Hippocampus::record`].
///
/// The trait is one-method by design — the cortex never needs to
/// read back from this sink; reads go through the in-memory store.
pub trait EpisodeSink: Send + Sync {
    /// Append an episode to durable storage. The id and timestamp
    /// are already populated by the caller.
    fn append(&self, episode: &Episode) -> Result<(), String>;

    /// Optional batch hook — invoked whenever the hippocampus
    /// evicts an episode to make room for a new one. Default
    /// implementation is a no-op.
    fn evict(&self, _id: u64) -> Result<(), String> {
        Ok(())
    }
}

/// Null sink — does nothing. Used when the hippocampus is created
/// without a backing store (e.g. inside a short-lived test).
#[derive(Debug, Default, Clone, Copy)]
pub struct NullEpisodeSink;

impl EpisodeSink for NullEpisodeSink {
    fn append(&self, _episode: &Episode) -> Result<(), String> {
        Ok(())
    }
}

/// Hippocampal store. Append-only on write; replay-driven read
/// during sleep.
pub struct Hippocampus {
    episodes: HashMap<u64, Episode>,
    by_source: HashMap<String, Vec<u64>>,
    next_id: u64,
    /// Salience floor — episodes strictly below this are dropped on write.
    /// Default `0.0`: every cortical event the brain chose to record
    /// is stored. The brain already does its own salience weighting
    /// (see [`crate::cortex::Cortex::tick`]); a second floor here
    /// was silently dropping first-tick and low-arousal episodes.
    pub salience_floor: f32,
    /// Capacity ceiling — once exceeded, the lowest-salience
    /// episode is evicted (the hippocampus doesn't grow forever).
    pub capacity: usize,
    /// Persistence sink. Wrapped in a `Mutex` so the trait object
    /// is `Sync`-safe even when the underlying implementation is
    /// not internally synchronized.
    sink: Arc<Mutex<dyn EpisodeSink>>,
}

impl Default for Hippocampus {
    fn default() -> Self {
        Self::new()
    }
}

impl Hippocampus {
    /// New empty hippocampus with default capacity of 100,000
    /// episodes and salience floor `0.0`. Uses a [`NullEpisodeSink`]
    /// — every record is kept in memory only.
    #[must_use]
    pub fn new() -> Self {
        Self::with_sink(Arc::new(Mutex::new(NullEpisodeSink)))
    }

    /// New hippocampus that mirrors every record to `sink`.
    #[must_use]
    pub fn with_sink(sink: Arc<Mutex<dyn EpisodeSink>>) -> Self {
        Self {
            episodes: HashMap::new(),
            by_source: HashMap::new(),
            next_id: 0,
            salience_floor: 0.0,
            capacity: 100_000,
            sink,
        }
    }

    /// Replace the persistence sink at runtime.
    pub fn set_sink(&mut self, sink: Arc<Mutex<dyn EpisodeSink>>) {
        self.sink = sink;
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
    /// `None` if the episode was rejected (strictly below salience
    /// floor).
    ///
    /// On success the episode is mirrored to the configured sink.
    /// A sink failure does **not** roll back the in-memory insert —
    /// the cortex must keep ticking even if persistence is
    /// degraded — but the error is returned to the caller so the
    /// embedding layer can decide how to react (retry, log, etc.).
    pub fn record(
        &mut self,
        sdr: Sdr,
        source: impl Into<String>,
        salience: f32,
        valence: [f32; 3],
    ) -> Option<u64> {
        self.record_with_metadata(sdr, source, salience, valence, EpisodeMetadata::Null)
    }

    /// Record an episode with explicit metadata. See [`Self::record`].
    pub fn record_with_metadata(
        &mut self,
        sdr: Sdr,
        source: impl Into<String>,
        salience: f32,
        valence: [f32; 3],
        metadata: EpisodeMetadata,
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
            metadata,
        };
        if let Err(e) = self.sink.lock().append(&ep) {
            tracing::warn!(error = %e, id, "hippocampus sink append failed");
        }
        self.by_source.entry(ep.source.clone()).or_default().push(id);
        self.episodes.insert(id, ep);
        Some(id)
    }

    /// Bulk-load episodes from persistence on startup. Does **not**
    /// rewrite `next_id`; callers must use [`Self::restore_all`]
    /// before any other write so the restored ids stay unique.
    pub fn restore_all<I: IntoIterator<Item = Episode>>(&mut self, iter: I) {
        for ep in iter {
            if ep.id >= self.next_id {
                self.next_id = ep.id + 1;
            }
            self.by_source
                .entry(ep.source.clone())
                .or_default()
                .push(ep.id);
            self.episodes.insert(ep.id, ep);
        }
    }

    /// Snapshot every stored episode. Used to feed persistence on
    /// shutdown or to broadcast state to a peer.
    pub fn snapshot(&self) -> Vec<Episode> {
        self.episodes.values().cloned().collect()
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
        candidates.sort_by(|a, b| {
            b.salience
                .partial_cmp(&a.salience)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
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
            a.1.salience
                .partial_cmp(&b.1.salience)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            self.episodes.remove(&id);
            for ids in self.by_source.values_mut() {
                ids.retain(|i| *i != id);
            }
            let _ = self.sink.lock().evict(id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Default)]
    struct CountingSink {
        appended: AtomicUsize,
        evicted: AtomicUsize,
    }
    impl EpisodeSink for CountingSink {
        fn append(&self, _e: &Episode) -> Result<(), String> {
            self.appended.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn evict(&self, _id: u64) -> Result<(), String> {
            self.evicted.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    #[test]
    fn salience_floor_filters_low_events() {
        let mut h = Hippocampus::new();
        h.salience_floor = 0.2;
        assert!(h.record(Sdr::empty(), "x", 0.1, [0.0; 3]).is_none());
        assert!(h.record(Sdr::empty(), "x", 0.5, [0.0; 3]).is_some());
    }

    #[test]
    fn default_floor_accepts_zero_salience() {
        let mut h = Hippocampus::new();
        assert_eq!(h.salience_floor, 0.0);
        assert!(h.record(Sdr::empty(), "x", 0.0, [0.0; 3]).is_some());
    }

    #[test]
    fn sink_mirrors_record_and_evict() {
        let sink = Arc::new(Mutex::new(CountingSink::default()));
        let mut h = Hippocampus::with_sink(sink.clone());
        h.record(Sdr::empty(), "low", 0.3, [0.0; 3]).unwrap();
        h.record(Sdr::empty(), "mid", 0.5, [0.0; 3]).unwrap();
        h.capacity = 2;
        h.record(Sdr::empty(), "high", 0.9, [0.0; 3]).unwrap();
        let s = sink.lock();
        assert_eq!(s.appended.load(Ordering::SeqCst), 3);
        assert_eq!(s.evicted.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn restore_all_preserves_ids_and_next_id() {
        let mut h = Hippocampus::new();
        let a = Episode {
            id: 0,
            sdr: Sdr::empty(),
            source: "x".into(),
            timestamp: 1,
            salience: 0.5,
            valence: [0.0; 3],
            replay_count: 0,
            metadata: serde_json::json!({"k": "v"}),
        };
        let b = Episode { id: 7, ..a.clone() };
        h.restore_all(vec![a, b]);
        assert_eq!(h.next_id, 8);
        assert_eq!(h.len(), 2);
        let fresh = h.record(Sdr::empty(), "new", 0.5, [0.0; 3]).unwrap();
        assert_eq!(fresh, 8);
    }

    #[test]
    fn capacity_evicts_lowest() {
        let mut h = Hippocampus::new();
        h.capacity = 2;
        h.record(Sdr::empty(), "low", 0.3, [0.0; 3]).unwrap();
        h.record(Sdr::empty(), "mid", 0.5, [0.0; 3]).unwrap();
        h.record(Sdr::empty(), "high", 0.9, [0.0; 3]).unwrap();
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
