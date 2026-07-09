//! The `Cortex` — top-level orchestrator that wires together every
//! brain-like subsystem.
//!
//! `Cortex::tick` performs the full per-step pipeline:
//!
//! 1. Inputs arrive as named SDRs.
//! 2. The thalamus gates each input by salience + attention bias.
//! 3. Surviving inputs enter the cortical hierarchy, which runs
//!    every region in topological order.
//! 4. The amygdala attaches valence; neuromodulators are updated.
//! 5. The global workspace chooses a winning coalition.
//! 6. Working memory is refreshed by the broadcast and decays.
//! 7. The basal ganglia picks the next action.
//! 8. A `ReplayFrame` is recorded for the Studio UI.
//!
//! `Cortex::sleep` runs one NREM/REM cycle and produces a
//! `ConsolidationReport`.

use crate::amygdala::{Amygdala, Valence};
use crate::attention::Attention;
use crate::basal_ganglia::{ActionSelection, BasalGanglia};
use crate::global_workspace::{Coalition, GlobalWorkspace};
use crate::hippocampus::{ConsolidationReport, Hippocampus};
use crate::neuromodulators::Neuromodulators;
use crate::region::{Hierarchy, RegionId};
use crate::replay::{ActivationMap, ModulatorSnapshot, ReplayBuffer, ReplayFrame};
use crate::sdr::{Sdr, SDR_WIDTH};
use crate::sleep::SleepCycle;
use crate::thalamus::{Thalamus, ThalamusChannel};
use crate::working_memory::WorkingMemory;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Top-level configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CortexConfig {
    /// Number of thalamic channels to create up-front.
    pub n_channels: usize,
    /// Number of cortical regions to create up-front.
    pub n_regions: usize,
    /// Working-memory capacity (default 7).
    pub wm_capacity: usize,
    /// Hippocampal capacity.
    pub hippocampus_capacity: usize,
    /// Salience floor for the thalamus.
    pub thalamus_salience_floor: f32,
}

impl Default for CortexConfig {
    fn default() -> Self {
        Self {
            n_channels: 4,
            n_regions: 3,
            wm_capacity: 7,
            hippocampus_capacity: 100_000,
            thalamus_salience_floor: 0.15,
        }
    }
}

/// Snapshot of the cortex's high-level state — for the Studio UI.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CortexStats {
    /// Tick count.
    pub ticks: u64,
    /// Total hippocampal episodes.
    pub episodes: usize,
    /// Total thalamic blocks.
    pub blocks: u64,
    /// Total thalamic forwards.
    pub forwards: u64,
    /// Mean overlap between consecutive top broadcasts.
    pub avg_broadcast_overlap: f32,
    /// Most recent action selected by the basal ganglia.
    pub last_action: Option<String>,
}

/// Result of one `Cortex::tick` call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThoughtBroadcast {
    /// Tick index.
    pub tick: u64,
    /// Winning global-workspace coalition.
    pub coalition: Coalition,
    /// SDR that won the basal-ganglia competition (or empty if
    /// no actions were offered).
    pub chosen_action: Option<String>,
    /// Valence of this tick.
    pub valence: Valence,
    /// Per-region activations for visualisation.
    pub activations: Vec<f32>,
}

/// The cortex.
pub struct Cortex {
    config: CortexConfig,
    thalamus: Thalamus,
    attention: Attention,
    hierarchy: Hierarchy,
    hippocampus: Hippocampus,
    amygdala: Amygdala,
    modulators: Neuromodulators,
    working_memory: WorkingMemory,
    basal_ganglia: BasalGanglia,
    global_workspace: GlobalWorkspace,
    replay: ReplayBuffer,
    stats: CortexStats,
    channels: Vec<u32>,
    regions: Vec<RegionId>,
    previous_top: Option<Sdr>,
}

impl Cortex {
    /// Build a cortex from a configuration.
    pub fn new(config: CortexConfig) -> Self {
        let mut thalamus = Thalamus::new();
        let mut channels: Vec<u32> = Vec::with_capacity(config.n_channels);
        for i in 0..config.n_channels {
            channels.push(thalamus.add_channel(format!("channel.{i}")));
        }
        let mut attention = Attention::new(config.n_channels);
        let mut hierarchy = Hierarchy::new();
        let mut regions: Vec<RegionId> = Vec::with_capacity(config.n_regions);
        for i in 0..config.n_regions {
            regions.push(hierarchy.add_region(format!("region.{i}"), SDR_WIDTH, i as u64 + 1));
        }
        // Linear connectivity — every region feeds the next one.
        for win in regions.windows(2) {
            hierarchy.connect(win[0], win[1]);
        }
        let mut hippocampus = Hippocampus::new();
        hippocampus.capacity = config.hippocampus_capacity;
        let mut working_memory = WorkingMemory::new();
        working_memory.capacity = config.wm_capacity;
        Self {
            config,
            thalamus,
            attention,
            hierarchy,
            hippocampus,
            amygdala: Amygdala::new(),
            modulators: Neuromodulators::new(),
            working_memory,
            basal_ganglia: BasalGanglia::new(),
            global_workspace: GlobalWorkspace::new(),
            replay: ReplayBuffer::default(),
            stats: CortexStats::default(),
            channels,
            regions,
            previous_top: None,
        }
    }

    /// Build a default cortex sized for tests.
    #[must_use]
    pub fn default_for_tests() -> Self {
        Self::new(CortexConfig::default())
    }

    /// Read-only access to the underlying subsystems.
    pub fn thalamus(&self) -> &Thalamus { &self.thalamus }
    pub fn attention(&self) -> &Attention { &self.attention }
    pub fn hierarchy(&self) -> &Hierarchy { &self.hierarchy }
    pub fn hippocampus(&self) -> &Hippocampus { &self.hippocampus }
    pub fn amygdala(&self) -> &Amygdala { &self.amygdala }
    pub fn modulators(&self) -> &Neuromodulators { &self.modulators }
    pub fn working_memory(&self) -> &WorkingMemory { &self.working_memory }
    pub fn basal_ganglia(&self) -> &BasalGanglia { &self.basal_ganglia }
    pub fn global_workspace(&self) -> &GlobalWorkspace { &self.global_workspace }
    pub fn replay(&self) -> &ReplayBuffer { &self.replay }
    pub fn stats(&self) -> &CortexStats { &self.stats }

    /// Run one tick. `inputs` maps channel name → SDR.
    pub fn tick(&mut self, inputs: HashMap<String, Sdr>) -> ThoughtBroadcast {
        let tick = self.replay.next_tick();
        let arousal = self.modulators.norepinephrine.level;
        let mut cortical_input: HashMap<RegionId, Sdr> = HashMap::new();
        // 1. Thalamic gating.
        let mut per_region_salience = vec![0.0_f32; self.regions.len()];
        for (name, input) in inputs {
            let Some(&ch) = self.channels.iter().find(|&&c| self.thalamus.channel(c).map(|x| x.name == name).unwrap_or(false)) else { continue };
            let bias = self.attention.effective_bias(ch as usize);
            let Some(sdr) = self.thalamus.relay(ch, input, arousal, bias) else {
                self.stats.blocks += 1;
                continue;
            };
            self.stats.forwards += 1;
            // Spread the gated SDR across all regions. The
            // hierarchy will route them through its own
            // connections.
            if let Some(first_region) = self.regions.first().copied() {
                cortical_input.entry(first_region).or_default().union_with(&sdr);
            }
            // Also record per-channel salience for attention.
            if let Some(ThalamusChannel { salience, .. }) = self.thalamus.channel(ch).cloned() {
                if !per_region_salience.is_empty() {
                    per_region_salience[0] = per_region_salience[0].max(salience);
                }
                self.attention.observe_salience(ch as usize, salience);
            }
        }
        // 2. Cortical hierarchy tick.
        let top = self.hierarchy.tick(&cortical_input);
        // 3. Valence + neuromodulators.
        let similarity = self.previous_top.as_ref().map(|p| crate::sdr::semantic_similarity(p, &top)).unwrap_or(0.0);
        let valence = self.amygdala.score(similarity);
        self.modulators.apply_valence(valence.reward, valence.threat, valence.novelty);
        // 4. Global workspace competition — every region's most
        // recent output competes against the gated SDRs.
        let mut candidates: Vec<(String, Sdr)> = Vec::new();
        for &id in &self.regions {
            if let Some(region) = self.hierarchy.region(id) {
                candidates.push((region.name.clone(), region.predict(&top)));
            }
        }
        let coalition = self.global_workspace.compete(&candidates, &top, 0.3);
        // 5. Working memory refresh.
        if let Some((idx, _)) = self.working_memory.best_match(&coalition.union) {
            self.working_memory.refresh(idx);
        } else {
            self.working_memory.push(coalition.union.cloned_active(), None);
        }
        self.working_memory.tick();
        // 6. Hippocampus: store episodes whose valence × arousal
        // exceeds the salience floor. We compute salience in two
        // parts: a novelty-driven component (favours new patterns)
        // and an arousal-driven component (favours high-norepi
        // ticks).
        let novelty_salience = valence.novelty;
        let arousal_salience = self.modulators.norepinephrine.level;
        let salience = 0.6 * novelty_salience + 0.4 * arousal_salience;
        if salience > self.hippocampus.salience_floor {
            self.hippocampus.record(
                coalition.union.cloned_active(),
                "global_workspace",
                salience,
                valence.to_array(),
            );
        }
        // 7. Basal ganglia: choose next action.
        let mut candidates: Vec<(String, String, f32)> = Vec::new();
        for &id in &self.regions {
            if let Some(region) = self.hierarchy.region(id) {
                candidates.push((
                    format!("{}.respond", region.name),
                    format!("Respond via {}", region.name),
                    crate::sdr::semantic_similarity(&coalition.union, &top),
                ));
            }
        }
        for c in candidates {
            self.basal_ganglia.offer(c.0, c.1, c.2);
        }
        let action = self.basal_ganglia.select();
        self.basal_ganglia.clear();
        // 8. Replay recording.
        let activations = self.activations();
        let frame = ReplayFrame {
            tick,
            timestamp: chrono::Utc::now().timestamp(),
            activation: ActivationMap {
                tick,
                per_region: activations.clone(),
                workspace: coalition.union.clone(),
                modulators: ModulatorSnapshot {
                    dopamine: self.modulators.dopamine.level,
                    serotonin: self.modulators.serotonin.level,
                    norepinephrine: self.modulators.norepinephrine.level,
                },
            },
        };
        self.replay.record(frame);
        // 9. Stats bookkeeping.
        if let Some(prev) = self.previous_top.as_ref() {
            let ov = crate::sdr::semantic_similarity(prev, &top);
            self.stats.avg_broadcast_overlap = 0.95 * self.stats.avg_broadcast_overlap + 0.05 * ov;
        }
        self.previous_top = Some(top);
        self.stats.ticks = tick + 1;
        self.stats.episodes = self.hippocampus.len();
        self.stats.last_action = action.winner.as_ref().map(|w| w.id.clone());

        ThoughtBroadcast {
            tick,
            coalition,
            chosen_action: self.stats.last_action.clone(),
            valence,
            activations,
        }
    }

    /// Run one sleep cycle.
    pub fn sleep(&mut self, replay_per_cycle: usize) -> ConsolidationReport {
        SleepCycle::new().run(&mut self.hippocampus, &mut self.hierarchy, replay_per_cycle)
    }

    /// Set the active top-down goal. Drives the attention map.
    pub fn set_goal(&mut self, goal: Sdr) {
        self.attention.set_goal(goal);
    }

    /// Register a custom action with the basal ganglia.
    pub fn offer_action(&mut self, id: impl Into<String>, label: impl Into<String>, activation: f32) {
        self.basal_ganglia.offer(id, label, activation);
    }

    fn activations(&self) -> Vec<f32> {
        self.regions
            .iter()
            .map(|id| {
                self.hierarchy
                    .region(*id)
                    .map(|r| r.stats().avg_step_overlap)
                    .unwrap_or(0.0)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cortex_tick_produces_broadcast() {
        let mut cortex = Cortex::default_for_tests();
        let mut inputs = HashMap::new();
        inputs.insert("channel.0".into(), Sdr::from_bits([1, 2, 3, 4, 5]));
        let broadcast = cortex.tick(inputs);
        assert!(!broadcast.coalition.members.is_empty());
    }

    #[test]
    fn repeated_ticks_record_replay_frames() {
        let mut cortex = Cortex::default_for_tests();
        for _ in 0..5 {
            let mut inputs = HashMap::new();
            inputs.insert("channel.0".into(), Sdr::from_bits([1, 2, 3, 4, 5]));
            let _ = cortex.tick(inputs);
        }
        assert_eq!(cortex.replay().len(), 5);
    }

    #[test]
    fn sleep_runs_consolidation() {
        let mut cortex = Cortex::default_for_tests();
        // Lower the salience floor so even mildly-novel events get
        // recorded. The amygdala EMAs are seeded to 0, so the
        // first tick's magnitude is 1.0 and easily passes the
        // default floor — but to keep the test deterministic we
        // simply bypass the floor.
        cortex.hippocampus.salience_floor = 0.0;
        for i in 0..5 {
            let mut inputs = HashMap::new();
            // Distinct inputs keep the amygdala valence above the
            // salience floor on every tick.
            let bits: Vec<usize> = (0..10).map(|j| i * 10 + j).collect();
            inputs.insert("channel.0".into(), Sdr::from_bits(bits));
            let _ = cortex.tick(inputs);
        }
        assert!(cortex.hippocampus().len() >= 3, "got {} episodes", cortex.hippocampus().len());
        let report = cortex.sleep(3);
        assert_eq!(report.episodes_replayed, 3);
    }
}
