//! Replay buffer — ring-buffered recording of cortical activity
//! for Studio visualisation and post-mortem analysis.

use crate::sdr::Sdr;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Activation map snapshot for a single tick.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationMap {
    /// Tick counter.
    pub tick: u64,
    /// Per-region activation levels in `[0, 1]`.
    pub per_region: Vec<f32>,
    /// Winning global-workspace SDR for this tick.
    pub workspace: Sdr,
    /// Neuromodulator snapshot.
    pub modulators: ModulatorSnapshot,
}

/// Snapshot of neuromodulator levels at a given tick.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModulatorSnapshot {
    /// Dopamine level in `[0, 1]`.
    pub dopamine: f32,
    /// Serotonin level in `[0, 1]`.
    pub serotonin: f32,
    /// Norepinephrine level in `[0, 1]`.
    pub norepinephrine: f32,
}

/// A single captured frame, used by Studio to playback thought
/// chains step-by-step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayFrame {
    /// Tick at which this frame was captured.
    pub tick: u64,
    /// Timestamp (unix seconds).
    pub timestamp: i64,
    /// Activation map.
    pub activation: ActivationMap,
}

/// Thread-safe replay buffer. Capacity is bounded so long-running
/// agents don't leak memory.
#[derive(Clone)]
pub struct ReplayBuffer {
    inner: Arc<Mutex<Vec<ReplayFrame>>>,
    capacity: usize,
    next_tick: Arc<Mutex<u64>>,
}

impl Default for ReplayBuffer {
    fn default() -> Self {
        Self::with_capacity(4096)
    }
}

impl ReplayBuffer {
    /// New buffer with the given capacity (oldest frames are
    /// dropped once the buffer is full).
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Vec::with_capacity(capacity))),
            capacity,
            next_tick: Arc::new(Mutex::new(0)),
        }
    }

    /// Record a single frame.
    pub fn record(&self, frame: ReplayFrame) {
        let mut buf = self.inner.lock();
        if buf.len() >= self.capacity {
            buf.remove(0);
        }
        buf.push(frame);
    }

    /// Number of stored frames.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.lock().len()
    }

    /// True if no frames are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.lock().is_empty()
    }

    /// Allocate the next tick counter atomically.
    pub fn next_tick(&self) -> u64 {
        let mut t = self.next_tick.lock();
        let v = *t;
        *t += 1;
        v
    }

    /// Snapshot of all stored frames — used by Studio.
    #[must_use]
    pub fn frames(&self) -> Vec<ReplayFrame> {
        self.inner.lock().clone()
    }

    /// Clear the buffer (e.g. at session boundaries).
    pub fn clear(&self) {
        self.inner.lock().clear();
        *self.next_tick.lock() = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_drops_oldest() {
        let buf = ReplayBuffer::with_capacity(3);
        for i in 0..5 {
            buf.record(ReplayFrame {
                tick: i,
                timestamp: 0,
                activation: ActivationMap {
                    tick: i,
                    per_region: vec![],
                    workspace: Sdr::empty(),
                    modulators: ModulatorSnapshot {
                        dopamine: 0.5,
                        serotonin: 0.5,
                        norepinephrine: 0.5,
                    },
                },
            });
        }
        assert_eq!(buf.len(), 3);
        let frames = buf.frames();
        assert_eq!(frames[0].tick, 2);
        assert_eq!(frames[2].tick, 4);
    }

    #[test]
    fn next_tick_increments() {
        let buf = ReplayBuffer::default();
        assert_eq!(buf.next_tick(), 0);
        assert_eq!(buf.next_tick(), 1);
    }
}
