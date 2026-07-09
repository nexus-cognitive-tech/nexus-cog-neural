//! Working Memory — Miller's 7±2 slots with active maintenance.
//!
//! Each slot holds an SDR. Slots decay unless refreshed by
//! attention. When a slot decays below the relevance floor its
//! SDR is returned to cortex and the slot is marked empty.

use crate::sdr::Sdr;
use serde::{Deserialize, Serialize};

/// One slot in working memory.
#[derive(Debug, Clone)]
struct Slot {
    /// Stored SDR, if any.
    sdr: Option<Sdr>,
    /// Current activation level in `[0, 1]`. Decays each tick.
    activation: f32,
    /// Optional label — caller-provided metadata.
    label: Option<String>,
}

/// Working memory.
#[derive(Debug)]
pub struct WorkingMemory {
    slots: Vec<Slot>,
    /// Decay per tick in `[0, 1]`. Higher = faster forgetting.
    pub decay_per_tick: f32,
    /// Slots whose activation drops below this floor are evicted.
    pub relevance_floor: f32,
    /// Hard capacity — Miller's 7±2; default 7.
    pub capacity: usize,
    ticks: u64,
}

impl Default for WorkingMemory {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            decay_per_tick: 0.1,
            relevance_floor: 0.05,
            capacity: 7,
            ticks: 0,
        }
    }
}

/// Snapshot of working-memory state for diagnostics / Studio UI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkingMemorySnapshot {
    /// Tick count when the snapshot was taken.
    pub ticks: u64,
    /// Per-slot state.
    pub slots: Vec<SlotSnapshot>,
}

/// Per-slot snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlotSnapshot {
    /// Slot index.
    pub index: usize,
    /// Active SDR if any.
    pub sdr: Option<Sdr>,
    /// Current activation.
    pub activation: f32,
    /// Optional label.
    pub label: Option<String>,
}

impl WorkingMemory {
    /// New empty WM with default capacity (7).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// WM with explicit capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self { capacity, ..Self::default() }
    }

    /// Add a new SDR to working memory. Returns the slot index
    /// (or `None` if WM is full and no slot could be evicted).
    pub fn push(&mut self, sdr: Sdr, label: Option<String>) -> Option<usize> {
        // Find an empty slot first.
        if let Some((idx, _)) = self.slots.iter_mut().enumerate().find(|(_, s)| s.sdr.is_none()) {
            self.slots[idx] = Slot { sdr: Some(sdr), activation: 1.0, label };
            return Some(idx);
        }
        // No empty slot — evict the least-relevant if it's below
        // the relevance floor; otherwise grow up to capacity.
        if let Some((idx, _)) = self.slots.iter_mut().enumerate().find(|(_, s)| s.activation < self.relevance_floor) {
            self.slots[idx] = Slot { sdr: Some(sdr), activation: 1.0, label };
            return Some(idx);
        }
        if self.slots.len() < self.capacity {
            self.slots.push(Slot { sdr: Some(sdr), activation: 1.0, label });
            return Some(self.slots.len() - 1);
        }
        None
    }

    /// Refresh an existing slot (top-down attention maintains it).
    pub fn refresh(&mut self, index: usize) -> bool {
        if let Some(slot) = self.slots.get_mut(index) {
            slot.activation = 1.0;
            true
        } else {
            false
        }
    }

    /// One tick of decay. Returns the indices of slots that were
    /// evicted this tick.
    pub fn tick(&mut self) -> Vec<usize> {
        self.ticks += 1;
        let mut evicted = Vec::new();
        for (i, slot) in self.slots.iter_mut().enumerate() {
            slot.activation = (slot.activation - self.decay_per_tick).max(0.0);
            if slot.activation < self.relevance_floor {
                slot.sdr = None;
                slot.label = None;
                evicted.push(i);
            }
        }
        evicted
    }

    /// Number of currently-occupied slots.
    #[must_use]
    pub fn n_filled(&self) -> usize {
        self.slots.iter().filter(|s| s.sdr.is_some()).count()
    }

    /// Read-only slot access.
    pub fn slot(&self, index: usize) -> Option<SlotSnapshot> {
        self.slots.get(index).map(|s| SlotSnapshot {
            index,
            sdr: s.sdr.clone(),
            activation: s.activation,
            label: s.label.clone(),
        })
    }

    /// Snapshot of the whole WM.
    #[must_use]
    pub fn snapshot(&self) -> WorkingMemorySnapshot {
        WorkingMemorySnapshot {
            ticks: self.ticks,
            slots: (0..self.slots.len())
                .map(|i| self.slot(i).unwrap())
                .collect(),
        }
    }

    /// Find the slot with the highest overlap to `query`, useful
    /// for attention-driven WM retrieval.
    pub fn best_match(&self, query: &Sdr) -> Option<(usize, f32)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                s.sdr.as_ref().map(|s| {
                    let sim = crate::sdr::semantic_similarity(s, query);
                    (i, sim)
                })
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Restore the slot from a persisted snapshot.
    pub fn restore(&mut self, snapshot: WorkingMemorySnapshot) {
        self.slots.clear();
        self.ticks = snapshot.ticks;
        for s in snapshot.slots {
            self.slots.push(Slot {
                sdr: s.sdr,
                activation: s.activation,
                label: s.label,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_decay() {
        let mut wm = WorkingMemory::new();
        wm.push(Sdr::from_bits([0, 1, 2]), None).unwrap();
        assert_eq!(wm.n_filled(), 1);
        for _ in 0..20 {
            wm.tick();
        }
        assert_eq!(wm.n_filled(), 0);
    }

    #[test]
    fn capacity_enforced() {
        let mut wm = WorkingMemory::with_capacity(3);
        for i in 0..3 {
            wm.push(Sdr::from_bits([i]), None).unwrap();
        }
        assert!(wm.push(Sdr::from_bits([99]), None).is_none());
    }

    #[test]
    fn best_match_returns_highest_overlap() {
        let mut wm = WorkingMemory::new();
        let target = Sdr::from_bits([0, 1, 2, 3]);
        wm.push(target.clone(), None).unwrap();
        wm.push(Sdr::from_bits([100, 101]), None).unwrap();
        let (i, sim) = wm.best_match(&target).unwrap();
        assert_eq!(i, 0);
        assert!(sim > 0.9);
    }
}
