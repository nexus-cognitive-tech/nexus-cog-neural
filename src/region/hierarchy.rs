//! A directed acyclic graph of cortical regions — the
//! "neocortex" in miniature.
//!
//! Each region has a fixed input width, set by the hierarchy on
//! insertion. The hierarchy is wired by [`Hierarchy::connect`]:
//!
//! ```text
//! thalamus → [V1] → [V2] → [V3]
//!                          ↘
//!                           [IT]   ← higher abstraction
//! audio   → [A1] → [A2] ───↗
//! ```
//!
//! On every tick the hierarchy runs regions in topological order;
//! each region's input is the union of its declared inputs.

use super::region::{Region, RegionId};
use crate::sdr::Sdr;
use std::collections::HashMap;

/// Hierarchy of cortical regions.
#[derive(Default)]
pub struct Hierarchy {
    regions: HashMap<RegionId, Region>,
    next_id: u32,
    edges: HashMap<RegionId, Vec<RegionId>>,
}

impl Hierarchy {
    /// New empty hierarchy.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a region. `input_width` is the SDR width that flows
    /// into it — typically `SDR_WIDTH` or the union of several
    /// sources (the caller is responsible for the union).
    pub fn add_region(&mut self, name: impl Into<String>, input_width: usize, seed: u64) -> RegionId {
        let id = RegionId(self.next_id);
        self.next_id += 1;
        let region = Region::new(id, name, input_width, seed);
        self.regions.insert(id, region);
        self.edges.entry(id).or_default();
        id
    }

    /// Declare `from` as a bottom-up source for `to`.
    pub fn connect(&mut self, from: RegionId, to: RegionId) {
        self.edges.entry(to).or_default().push(from);
    }

    /// Read-only access to a region.
    pub fn region(&self, id: RegionId) -> Option<&Region> {
        self.regions.get(&id)
    }

    /// Mutable access — useful for testing (injecting noise, etc.).
    pub fn region_mut(&mut self, id: RegionId) -> Option<&mut Region> {
        self.regions.get_mut(&id)
    }

    /// Number of regions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.regions.len()
    }

    /// True if no regions have been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }

    /// One full tick — every region fires once, in insertion
    /// order. Returns the top region's output as the "neocortical
    /// broadcast" for this tick.
    pub fn tick(&mut self, bottom_up_inputs: &HashMap<RegionId, Sdr>) -> Sdr {
        // Snapshot outputs produced this step so we can fan them in
        // without fighting the borrow checker.
        let mut produced: HashMap<RegionId, Sdr> = HashMap::new();
        let order: Vec<RegionId> = {
            let mut regions: Vec<RegionId> = self.regions.keys().copied().collect();
            regions.sort_by_key(|r| r.0);
            regions
        };
        for id in order {
            let mut combined = Sdr::empty();
            if let Some(srcs) = self.edges.get(&id).cloned() {
                for src in srcs {
                    if let Some(s) = produced.get(&src) {
                        combined.union_with(s);
                    }
                    if let Some(s) = bottom_up_inputs.get(&src) {
                        combined.union_with(s);
                    }
                }
            }
            if combined.active_count() == 0 {
                // No upstream signal — pull from any explicit
                // input keyed by the region's own id.
                if let Some(s) = bottom_up_inputs.get(&id) {
                    combined = s.clone();
                }
            }
            if let Some(region) = self.regions.get_mut(&id) {
                let out = region.step(&combined);
                produced.insert(id, out);
            }
        }
        // The topmost region is the one with no outgoing edges.
        // HashMap has no stable iteration order so we collect and
        // sort.
        let mut sorted_ids: Vec<RegionId> = self.regions.keys().copied().collect();
        sorted_ids.sort_by_key(|r| std::cmp::Reverse(r.0));
        let top = sorted_ids.into_iter().find(|r| {
            self.edges
                .values()
                .all(|targets| !targets.contains(r) || targets == &vec![*r])
        });
        match top.and_then(|id| produced.remove(&id)) {
            Some(s) => s,
            None => Sdr::empty(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_two_region_chain() {
        let mut h = Hierarchy::new();
        let r1 = h.add_region("V1", crate::sdr::SDR_WIDTH, 1);
        let r2 = h.add_region("V2", crate::sdr::SDR_WIDTH, 2);
        h.connect(r1, r2);

        let mut inputs = HashMap::new();
        inputs.insert(r1, Sdr::random_active(&mut rand::thread_rng(), 40));
        let out = h.tick(&inputs);
        assert!(out.active_count() > 0);
    }
}
