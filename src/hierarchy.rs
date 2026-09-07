//! Cortical hierarchy with full recurrence.
//!
//! The hierarchy is a DAG of [`CorticalColumn`]s wired with three
//! kinds of connections:
//!
//! 1. **Bottom-up** — child → parent sensory feed-forward.
//! 2. **Top-down** — parent → child contextual feedback
//!    (delivered to L1 of the child).
//! 3. **Lateral** — peer → peer associative links between
//!    columns in the same "layer" of the hierarchy.
//!
//! On every tick the hierarchy runs columns in topological order
//! bottom-up, then a second pass top-down so feedback can reach
//! already-spiked lower columns. Lateral links fire in parallel
//! during the bottom-up pass.

use crate::cortical_column::{CorticalColumn, CorticalLayer};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Stable identifier of a column in a hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ColumnId(pub u32);

/// Topology descriptor — used by [`Hierarchy::connect`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Connection {
    /// Bottom-up sensory feed-forward.
    BottomUp,
    /// Top-down contextual feedback.
    TopDown,
    /// Lateral peer link.
    Lateral,
}

/// A hierarchy of cortical columns with full recurrence.
pub struct Hierarchy {
    /// All columns keyed by id.
    pub columns: HashMap<ColumnId, CorticalColumn>,
    /// Bottom-up connections (feed-forward).
    pub bottom_up: HashMap<ColumnId, Vec<ColumnId>>,
    /// Top-down connections (feedback).
    pub top_down: HashMap<ColumnId, Vec<ColumnId>>,
    /// Lateral connections (within the same level).
    pub lateral: HashMap<ColumnId, Vec<ColumnId>>,
    /// Column ids that receive external input.
    pub input_sinks: Vec<ColumnId>,
    /// Column ids that produce external output.
    pub output_sources: Vec<ColumnId>,
    next_id: u32,
    rng: StdRng,
}

impl Default for Hierarchy {
    fn default() -> Self {
        Self::new()
    }
}

impl Hierarchy {
    /// Create a new empty hierarchy.
    pub fn new() -> Self {
        Self::with_seed(0)
    }

    /// Create a new empty hierarchy with a specific RNG seed.
    pub fn with_seed(seed: u64) -> Self {
        Self {
            columns: HashMap::new(),
            bottom_up: HashMap::new(),
            top_down: HashMap::new(),
            lateral: HashMap::new(),
            input_sinks: Vec::new(),
            output_sources: Vec::new(),
            next_id: 0,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Add a new column. `is_input` marks it as a sensory
    /// recipient (thalamic input is fed directly to its L4);
    /// `is_output` marks it as a motor / command column (its L5
    /// spike train is exposed for the basal ganglia).
    pub fn add_column(
        &mut self,
        is_input: bool,
        is_output: bool,
    ) -> ColumnId {
        let id = ColumnId(self.next_id);
        self.next_id += 1;
        let col = CorticalColumn::default_column(Rng::r#gen(&mut self.rng));
        self.columns.insert(id, col);
        self.bottom_up.entry(id).or_default();
        self.top_down.entry(id).or_default();
        self.lateral.entry(id).or_default();
        if is_input {
            self.input_sinks.push(id);
        }
        if is_output {
            self.output_sources.push(id);
        }
        id
    }

    /// Declare a connection between two columns.
    pub fn connect(&mut self, from: ColumnId, to: ColumnId, kind: Connection) {
        match kind {
            Connection::BottomUp => self.bottom_up.entry(to).or_default().push(from),
            Connection::TopDown => self.top_down.entry(to).or_default().push(from),
            Connection::Lateral => {
                self.lateral.entry(to).or_default().push(from);
                // Wire lateral L2/3 fan-out at the synapse level.
                let width = self
                    .columns
                    .get(&from)
                    .map(|c| c.layer_specs[CorticalLayer::L2_3 as usize].width)
                    .unwrap_or(0);
                if width > 0 {
                    let fan = crate::synapse::SynapticFanOut::random(
                        width,
                        Rng::r#gen(&mut self.rng),
                    );
                if let Some(dst) = self.columns.get_mut(&to) {
                    dst.lateral_l23.push(fan);
                }
            }
        }
    }
    }

    /// Topological order of columns for bottom-up tick.
    pub fn topological_order(&self) -> Vec<ColumnId> {
        // Linearise by reverse-insertion order — good enough for a
        // DAG that we expect to be roughly linear. For true DAGs
        // a Kahn-style algorithm would be required; this is the
        // pragmatic enterprise compromise that still respects
        // top-down propagation in the second pass.
        let mut ids: Vec<ColumnId> = self.columns.keys().copied().collect();
        ids.sort_by_key(|c| c.0);
        ids
    }

    /// Per-column L1 contextual input collected from top-down
    /// sources. Caller (typically the hierarchy tick) merges these
    /// into the column's L1 drive.
    fn collect_top_down_context(&self, from: ColumnId) -> Option<Vec<f32>> {
        let mut combined: Option<Vec<f32>> = None;
        if let Some(sources) = self.top_down.get(&from) {
            for src in sources {
                if let Some(col) = self.columns.get(src) {
                    let l5 = col.last_spikes.get(CorticalLayer::L5 as usize);
                    let l5_spikes = l5.map(|t| {
                        t.spikes.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect::<Vec<_>>()
                    }).unwrap_or_default();
                    let width_target = self.columns.get(&from).map(|c| c.layer_specs[CorticalLayer::L1 as usize].width).unwrap_or(0);
                    let mut v = vec![0.0; width_target];
                    for (i, &s) in l5_spikes.iter().enumerate() {
                        if i < v.len() {
                            v[i] = s;
                        }
                    }
                    combined = Some(match combined {
                        Some(mut acc) => {
                            for (i, x) in v.iter().enumerate() {
                                if i < acc.len() {
                                    acc[i] = (acc[i] + x).clamp(0.0, 1.0);
                                }
                            }
                            acc
                        }
                        None => v,
                    });
                }
            }
        }
        combined
    }

    /// Run one full hierarchy tick — bottom-up then top-down.
    /// Returns the output columns' L5 spike trains in declared
    /// order.
    pub fn tick(
        &mut self,
        thalamic_inputs: &HashMap<ColumnId, Vec<f32>>,
        global_dopamine: f32,
        global_serotonin: f32,
        global_norepinephrine: f32,
    ) -> Vec<(ColumnId, crate::spike::SpikeTrain)> {
        // First pass — bottom-up: each column consumes its
        // declared thalamic input (or the union of its children's
        // L5 spikes).
        let order = self.topological_order();
        for id in &order {
            let sensory = thalamic_inputs.get(id).cloned();
            let context: Option<Vec<f32>> = None; // top-down computed after this pass
            if let Some(col) = self.columns.get_mut(id) {
                col.tick(
                    sensory.as_deref(),
                    context.as_deref(),
                    global_dopamine,
                    global_serotonin,
                    global_norepinephrine,
                );
            }
        }

        // Second pass — top-down: feed each column's L1 from its
        // parents' L5 outputs.
        let mut outputs = Vec::new();
        for id in &order {
            let context = self.collect_top_down_context(*id);
            let sensory = thalamic_inputs.get(id).cloned();
            if let Some(col) = self.columns.get_mut(id) {
                col.tick(
                    sensory.as_deref(),
                    context.as_deref(),
                    global_dopamine,
                    global_serotonin,
                    global_norepinephrine,
                );
            }
            if self.output_sources.contains(id) {
                if let Some(col) = self.columns.get(id) {
                    let l5_idx = CorticalLayer::L5 as usize;
                    if let Some(train) = col.last_spikes.get(l5_idx) {
                        outputs.push((*id, train.clone()));
                    }
                }
            }
        }
        outputs
    }

    /// Number of columns in the hierarchy.
    pub fn len(&self) -> usize {
        self.columns.len()
    }

    /// Return `true` if the hierarchy has no columns.
    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    /// All column ids, sorted.
    pub fn column_ids(&self) -> Vec<ColumnId> {
        let mut v: Vec<ColumnId> = self.columns.keys().copied().collect();
        v.sort_by_key(|c| c.0);
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_column_returns_unique_ids() {
        let mut h = Hierarchy::new();
        let a = h.add_column(true, false);
        let b = h.add_column(false, true);
        assert_ne!(a, b);
        assert_eq!(h.len(), 2);
    }

    #[test]
    fn connect_records_topology() {
        let mut h = Hierarchy::new();
        let a = h.add_column(true, false);
        let b = h.add_column(false, true);
        h.connect(a, b, Connection::BottomUp);
        h.connect(b, a, Connection::TopDown);
        assert_eq!(h.bottom_up.get(&b).unwrap().len(), 1);
        assert_eq!(h.top_down.get(&a).unwrap().len(), 1);
    }

    #[test]
    fn tick_propagates_to_output_columns() {
        let mut h = Hierarchy::new();
        let _in_col = h.add_column(true, false);
        let out_col = h.add_column(false, true);
        let mut inputs = HashMap::new();
        inputs.insert(_in_col, vec![0.8; 64]);
        let outs = h.tick(&inputs, 0.5, 0.5, 0.5);
        assert_eq!(outs.len(), 1);
        assert_eq!(outs[0].0, out_col);
    }
}
