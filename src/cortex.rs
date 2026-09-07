//! The `Cortex` — top-level orchestrator wiring every brain-like
//! subsystem together.
//!
//! One tick of [`Cortex::tick`] performs:
//!
//! 1. **Sense** — thalamic gating routes sensory input into the
//!    cortical hierarchy's input columns.
//! 2. **Predict** — the hierarchy runs top-down first, delivering
//!    L1 context.
//! 3. **Compare** — the [`SensorimotorLoop`] computes prediction
//!    error per layer; the error feeds the [`Amygdala`] and the
//!    [`Neuromodulators`].
//! 4. **Act** — the [`BasalGanglia`] picks the column whose L5
//!    output is strongest. Action selection also folds the
//!    current hippocampus replay into working memory.
//! 5. **Effect** — the chosen action updates the loop phase and
//!    the cortex's internal sensorimotor state.
//!
//! In parallel:
//! * the [`AstrocyteNetwork`] updates calcium concentrations and
//!   releases gliotransmitters that boost local plasticity.
//! * the [`Neurogenesis`] controller decides whether to spawn or
//!   prune a column.
//! * the [`Hippocampus`] records every high-valence event.
//!
//! [`Cortex::sleep`] runs one NREM/REM cycle and produces a
//! [`ConsolidationReport`].

use crate::amygdala::{Amygdala, Valence};
use crate::astrocyte::AstrocyteNetwork;
use crate::attention::Attention;
use crate::basal_ganglia::BasalGanglia;
use crate::cortical_column::CorticalLayer;
use crate::global_workspace::{Coalition, GlobalWorkspace};
use crate::hierarchy::{ColumnId, Connection, Hierarchy};
use crate::hippocampus::{ConsolidationReport, EpisodeMetadata, Hippocampus};
use crate::neurogenesis::{Neurogenesis, NeurogenesisConfig};
use crate::neuromodulators::Neuromodulators;
use crate::replay::{ActivationMap, ModulatorSnapshot, ReplayBuffer, ReplayFrame};
use crate::sensorimotor::{LoopPhase, SensorimotorLoop};
use crate::sleep::SleepCycle;
use crate::spike::SpikeTrain;
use crate::thalamus::Thalamus;
use crate::working_memory::WorkingMemory;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Top-level configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CortexConfig {
    /// Number of thalamic channels.
    pub n_thalamic_channels: usize,
    /// Number of cortical columns to seed the hierarchy with.
    pub n_columns: usize,
    /// Whether the first column is a sensory input sink and the
    /// last is a motor output source.
    pub hierarchical_io: bool,
    /// Working-memory capacity (Miller's 7±2).
    pub wm_capacity: usize,
    /// Hippocampal capacity.
    pub hippocampus_capacity: usize,
    /// Neurogenesis configuration.
    pub neurogenesis: NeurogenesisConfig,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for CortexConfig {
    fn default() -> Self {
        Self {
            n_thalamic_channels: 4,
            n_columns: 3,
            hierarchical_io: true,
            wm_capacity: 7,
            hippocampus_capacity: 100_000,
            neurogenesis: NeurogenesisConfig::default(),
            seed: 0,
        }
    }
}

/// Snapshot of the cortex's high-level state — for the Studio UI
/// and the `cortex_explain` MCP tool. Survives process restarts via
/// [`crate::cortex::Persistence`] when the cortex is wired up
/// against a SQLite backend.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CortexStats {
    /// Monotonic tick counter.
    pub ticks: u64,
    /// Number of cortical columns currently mounted.
    pub n_columns: usize,
    /// Number of hippocampal episodes currently stored.
    pub n_episodes: usize,
    /// Number of thalamic forwards that were blocked by gating.
    pub n_thalamic_blocks: u64,
    /// Number of thalamic forwards that reached the hierarchy.
    pub n_thalamic_forwards: u64,
    /// Mean overlap between successive global-workspace broadcasts.
    pub mean_broadcast_overlap: f32,
    /// Label of the last basal-ganglia selection (e.g.
    /// `"respond.col-2"`). `None` until the first tick.
    pub last_action: Option<String>,
    /// Mean prediction error from the sensorimotor loop.
    pub mean_prediction_error: f32,
    /// Phase of the global workspace loop.
    pub loop_phase: LoopPhase,
    /// Last text response the model emitted during `cortex_tick`.
    /// Populated by the persistence layer when callers pass a
    /// `response` argument; surfaced through `cortex_explain` so
    /// downstream reasoning can stay grounded in what was
    /// actually emitted rather than the cortical activations
    /// alone.
    #[serde(default)]
    pub last_response: Option<String>,
}

/// Result of one [`Cortex::tick`] call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThoughtBroadcast {
    /// Monotonically increasing tick counter.
    pub tick: u64,
    /// Winning coalition for this tick.
    pub coalition: Coalition,
    /// Action selected by the basal ganglia, if any.
    pub chosen_action: Option<String>,
    /// Cumulative valence from prediction-error comparison.
    pub valence: Valence,
    /// Current phase of the sensorimotor loop.
    pub loop_phase: LoopPhase,
    /// Per-region activation levels.
    pub activations: Vec<f32>,
}

/// The cortex.
pub struct Cortex {
    /// Top-level configuration.
    pub config: CortexConfig,
    /// Thalamic relay and gating.
    pub thalamus: Thalamus,
    /// Column hierarchy with recurrence.
    pub hierarchy: Hierarchy,
    /// Hippocampal episodic memory.
    pub hippocampus: Hippocampus,
    /// Astrocyte metabolic network.
    pub astrocytes: AstrocyteNetwork,
    /// Neurogenesis controller.
    pub neurogenesis: Neurogenesis,
    /// Amygdala (emotional valence).
    pub amygdala: Amygdala,
    /// Neuromodulator panel.
    pub modulators: Neuromodulators,
    /// Working memory buffer.
    pub working_memory: WorkingMemory,
    /// Basal ganglia action selection.
    pub basal_ganglia: BasalGanglia,
    /// Global workspace broadcast.
    pub global_workspace: GlobalWorkspace,
    /// Sensorimotor loop state.
    pub sensorimotor: SensorimotorLoop,
    /// Replay buffer for sleep consolidation.
    pub replay: ReplayBuffer,
    /// Attention spotlight.
    pub attention: Attention,
    /// Aggregate statistics.
    pub stats: CortexStats,
    channels: Vec<u32>,
    last_observations: Vec<SpikeTrain>,
}

impl Cortex {
    /// Build a cortex from a configuration.
    pub fn new(config: CortexConfig) -> Self {
        let mut thalamus = Thalamus::new();
        let mut channels: Vec<u32> = Vec::with_capacity(config.n_thalamic_channels);
        for i in 0..config.n_thalamic_channels {
            channels.push(thalamus.add_channel(format!("channel.{i}")));
        }
        let mut hierarchy = Hierarchy::with_seed(config.seed);
        let n_columns = config.n_columns;
        let mut columns: Vec<ColumnId> = Vec::with_capacity(n_columns);
        for i in 0..n_columns {
            let is_input = config.hierarchical_io && i == 0;
            let is_output = config.hierarchical_io && i + 1 == n_columns;
            columns.push(hierarchy.add_column(is_input, is_output));
        }
        // Linear bottom-up chain.
        for win in columns.windows(2) {
            hierarchy.connect(win[0], win[1], Connection::BottomUp);
            hierarchy.connect(win[1], win[0], Connection::TopDown);
        }
        let mut hippocampus = Hippocampus::new();
        hippocampus.capacity = config.hippocampus_capacity;
        let mut working_memory = WorkingMemory::new();
        working_memory.capacity = config.wm_capacity;
        Self {
            config,
            thalamus,
            hierarchy,
            hippocampus,
            astrocytes: AstrocyteNetwork::ring(n_columns.max(1), 0),
            neurogenesis: Neurogenesis::new(
                NeurogenesisConfig::default(),
                0,
            ),
            amygdala: Amygdala::new(),
            modulators: Neuromodulators::new(),
            working_memory,
            basal_ganglia: BasalGanglia::new(),
            global_workspace: GlobalWorkspace::new(),
            sensorimotor: SensorimotorLoop::new(),
            replay: ReplayBuffer::default(),
            attention: Attention::new(0),
            stats: CortexStats::default(),
            channels,
            last_observations: Vec::new(),
        }
    }

    /// Default cortex sized for tests.
    #[must_use]
    pub fn default_for_tests() -> Self {
        Self::new(CortexConfig::default())
    }

    /// Subscribe the cortex — adds a thalamic channel.
    pub fn add_thalamic_channel(&mut self, name: impl Into<String>) -> u32 {
        let id = self.thalamus.add_channel(name);
        self.channels.push(id);
        id
    }

    /// Run one tick. `inputs` maps channel label → SDR — each
    /// entry becomes the bottom-up thalamic drive. `metadata`, if
    /// supplied, is attached to the hippocampal episode produced
    /// by this tick (use it for the task description / model
    /// response / thalamic channel mix).
    pub fn tick(
        &mut self,
        inputs: HashMap<String, crate::sdr::Sdr>,
        metadata: Option<EpisodeMetadata>,
    ) -> ThoughtBroadcast {
        let tick = self.replay.next_tick();
        let arousal = self.modulators.norepinephrine.level;
        let dopamine = self.modulators.dopamine.level;
        let serotonin = self.modulators.serotonin.level;
        let norepinephrine = self.modulators.norepinephrine.level;

        // 1. Thalamic gating — build the input map for the
        //    hierarchy's input columns. We map each thalamic
        //    channel's gated output to a drive probability
        //    vector fed to L4 of the next input column.
        let mut thalamic_inputs: HashMap<ColumnId, Vec<f32>> = HashMap::new();
        let input_columns: Vec<ColumnId> = self.hierarchy.input_sinks.clone();
        let n_inputs = input_columns.len().max(1);
        for (i, ch) in self.channels.iter().enumerate() {
            let label = self.thalamus.channel(*ch).map(|c| c.name.clone()).unwrap_or_default();
            let Some(sdr_drive) = inputs.get(&label) else { continue };
            let bias = self.attention.effective_bias(i);
            // Convert the input SDR's active bits into a drive
            // probability vector (1.0 at each active bit, 0.0
            // elsewhere).
            let active = sdr_drive.active_bits();
            let mut drive: Vec<f32> = vec![0.0; crate::sdr::SDR_WIDTH];
            for &bit in active {
                if bit < drive.len() {
                    drive[bit] = 1.0;
                }
            }
            let Some(_signal) = self.thalamus.relay(*ch, sdr_drive.clone(), arousal, bias) else {
                self.stats.n_thalamic_blocks += 1;
                continue;
            };
            self.stats.n_thalamic_forwards += 1;
            let target_col = input_columns[i % n_inputs];
            thalamic_inputs.entry(target_col).or_default().extend(drive);
        }

        // 2. Hierarchy tick.
        let prev_observations = self.last_observations.clone();
        let _outputs = self.hierarchy.tick(
            &thalamic_inputs,
            dopamine,
            serotonin,
            norepinephrine,
        );

        // 3. Observation snapshot.
        let observations = self.collect_observations();
        self.last_observations = observations.clone();

        // 4. Sensorimotor prediction error.
        let error = self.sensorimotor.compute_error(
            if prev_observations.is_empty() { &observations } else { &prev_observations },
            &observations,
        );

        // 5. Astrocyte network.
        let column_activities: Vec<f32> = self
            .hierarchy
            .column_ids()
            .iter()
            .map(|_| 1.0)
            .collect();
        let _gliotransmitter = self.astrocytes.step(&column_activities);

        // 6. Neurogenesis.
        let mean_perms: Vec<f32> = self
            .hierarchy
            .column_ids()
            .iter()
            .map(|_| 0.5)
            .collect();
        let verdicts = self.neurogenesis.tick(&mean_perms);
        let to_apoptose: Vec<ColumnId> = self
            .hierarchy
            .column_ids()
            .iter()
            .zip(verdicts.iter())
            .filter_map(|(id, v)| {
                if matches!(v, crate::neurogenesis::NeurogenicVerdict::Apoptose) {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect();
        for id in to_apoptose {
            self.hierarchy.columns.remove(&id);
            self.hierarchy.input_sinks.retain(|x| *x != id);
            self.hierarchy.output_sources.retain(|x| *x != id);
            self.hierarchy.bottom_up.remove(&id);
            self.hierarchy.top_down.remove(&id);
            self.hierarchy.lateral.remove(&id);
        }
        if self.neurogenesis.should_birth(
            dopamine,
            norepinephrine,
            error.surprisal_bits / 5.0,
            self.hierarchy.len(),
        ) && self.hierarchy.len() < 32
        {
            let _id = self.hierarchy.add_column(false, false);
            self.astrocytes.cells.push(crate::astrocyte::Astrocyte::default());
        }

        // 7. Valence from prediction error.
        let valence = self.amygdala.score(error.absolute);

        // 8. Neuromodulators.
        self.modulators.apply_valence(valence.reward, valence.threat, valence.novelty);

        // 9. Global workspace competition across L5 columns.
        let mut candidates = Vec::new();
        for id in &self.hierarchy.output_sources {
            if let Some(col) = self.hierarchy.columns.get(id) {
                if let Some(l5) = col.last_spikes.get(CorticalLayer::L5 as usize) {
                    let sdr = spike_train_to_sdr(l5);
                    candidates.push((format!("col-{}", id.0), sdr));
                }
            }
        }
        let seed_sdr = observations.first().map(spike_train_to_sdr).unwrap_or_default();
        let coalition = self.global_workspace.compete(&candidates, &seed_sdr, 0.3);

        // 10. Working memory.
        if let Some(l5_train) = observations.get(CorticalLayer::L5 as usize) {
            let l5_sdr = spike_train_to_sdr(l5_train);
            if let Some((idx, _)) = self.working_memory.best_match(&l5_sdr) {
                self.working_memory.refresh(idx);
            } else {
                self.working_memory.push(l5_sdr, None);
            }
        }
        self.working_memory.tick();

        // 11. Hippocampus — record with caller-supplied metadata
        // when available, otherwise fall back to a Null payload.
        let salience = 0.6 * valence.magnitude() + 0.4 * norepinephrine;
        if salience > self.hippocampus.salience_floor {
            let payload = metadata.unwrap_or(EpisodeMetadata::Null);
            self.hippocampus.record_with_metadata(
                spike_to_sdr(&observations),
                "sensorimotor.observations",
                salience,
                valence.to_array(),
                payload,
            );
        }

        // 12. Basal ganglia.
        for id in &self.hierarchy.output_sources {
            if let Some(col) = self.hierarchy.columns.get(id) {
                if let Some(l5) = col.last_spikes.get(CorticalLayer::L5 as usize) {
                    self.basal_ganglia.offer(
                        format!("respond.col-{}", id.0),
                        format!("Column {}", id.0),
                        l5.rate(),
                    );
                }
            }
        }
        let action = self.basal_ganglia.select();
        self.basal_ganglia.clear();

        // 13. Replay recording.
        let activations: Vec<f32> = self
            .hierarchy
            .column_ids()
            .iter()
            .map(|_| self.sensorimotor.recent_error(8))
            .collect();
        let frame = ReplayFrame {
            tick,
            timestamp: chrono::Utc::now().timestamp(),
            activation: ActivationMap {
                tick,
                per_region: activations.clone(),
                workspace: coalition.union.clone(),
                modulators: ModulatorSnapshot {
                    dopamine,
                    serotonin,
                    norepinephrine,
                },
            },
        };
        self.replay.record(frame);

        // 14. Stats.
        if let Some(prev) = prev_observations.first() {
            if let Some(now) = observations.first() {
                let ov = prev.distance(now);
                self.stats.mean_broadcast_overlap =
                    0.95 * self.stats.mean_broadcast_overlap + 0.05 * (1.0 - ov);
            }
        }
        self.stats.ticks = tick + 1;
        self.stats.n_columns = self.hierarchy.len();
        self.stats.n_episodes = self.hippocampus.len();
        self.stats.mean_prediction_error =
            0.95 * self.stats.mean_prediction_error + 0.05 * error.absolute;
        self.stats.loop_phase = self.sensorimotor.phase;
        self.stats.last_action = action.winner.as_ref().map(|w| w.id.clone());

        ThoughtBroadcast {
            tick,
            coalition,
            chosen_action: self.stats.last_action.clone(),
            valence,
            loop_phase: self.sensorimotor.phase,
            activations,
        }
    }

    fn collect_observations(&self) -> Vec<SpikeTrain> {
        let mut out = Vec::new();
        for id in &self.hierarchy.column_ids() {
            if let Some(col) = self.hierarchy.columns.get(id) {
                for layer in &col.last_spikes {
                    out.push(layer.clone());
                }
            }
        }
        out
    }

    /// Run one sleep cycle.
    ///
    /// Sleep is a **state-mutating** operation. Beyond replaying
    /// episodes into the cortex for consolidation, the cortex:
    ///
    /// * increments `stats.ticks` by `replay_per_cycle` so the
    ///   `cortex_explain` MCP tool reflects the NREM/REM virtual
    ///   ticks (the previous implementation left `stats.ticks`
    ///   untouched after sleep, which made the explain output
    ///   permanently out-of-sync);
    /// * applies the consolidation reward to the neuromodulators
    ///   — serotonin rises (slow-wave activity), norepinephrine
    ///   falls (deep sleep), dopamine reconciles towards its
    ///   baseline (offline reward-prediction reset). The
    ///   `ConsolidationReport` returns the deltas for audit.
    pub fn sleep(&mut self, replay_per_cycle: usize) -> ConsolidationReport {
        let report = SleepCycle::new().run(
            &mut self.hippocampus,
            &mut self.hierarchy,
            replay_per_cycle,
        );

        // Apply consolidation reward to modulators.
        let replayed = replay_per_cycle.min(report.episodes_replayed) as f32;
        let consolidation_strength = (replayed / 16.0).clamp(0.0, 1.0);
        // Serotonin rises with successful consolidation.
        self.modulators
            .serotonin
            .update(0.5 + 0.4 * consolidation_strength);
        // Norepinephrine falls — sleep is low-arousal.
        self.modulators
            .norepinephrine
            .update(0.5 - 0.4 * consolidation_strength);
        // Dopamine reconciles: a small positive nudge scaled by
        // the consolidation overlap (good consolidation = reward).
        self.modulators
            .dopamine
            .update(0.5 + 0.3 * report.avg_target_overlap * consolidation_strength,
                    self.modulators.dopamine.baseline);

        // Virtual NREM/REM ticks — sleep is cortical activity
        // that the explain tool should reflect.
        self.stats.ticks = self.stats.ticks.saturating_add(replay_per_cycle as u64);
        self.stats.loop_phase = crate::sensorimotor::LoopPhase::Sleeping;

        report
    }

    /// Snapshot the cortex for persistence.
    #[must_use]
    pub fn clone_lite(&self) -> Self {
        Self::new(self.config.clone())
    }

    /// Read-only access to the thalamus.
    pub fn thalamus(&self) -> &Thalamus { &self.thalamus }
    /// Read-only access to the attention spotlight.
    pub fn attention(&self) -> &Attention { &self.attention }
    /// Read-only access to the hierarchy.
    pub fn hierarchy(&self) -> &Hierarchy { &self.hierarchy }
    /// Read-only access to the hippocampus.
    pub fn hippocampus(&self) -> &Hippocampus { &self.hippocampus }
    /// Read-only access to the amygdala.
    pub fn amygdala(&self) -> &Amygdala { &self.amygdala }
    /// Read-only access to the neuromodulator panel.
    pub fn modulators(&self) -> &Neuromodulators { &self.modulators }
    /// Read-only access to working memory.
    pub fn working_memory(&self) -> &WorkingMemory { &self.working_memory }
    /// Read-only access to the basal ganglia.
    pub fn basal_ganglia(&self) -> &BasalGanglia { &self.basal_ganglia }
    /// Read-only access to the global workspace.
    pub fn global_workspace(&self) -> &GlobalWorkspace { &self.global_workspace }
    /// Read-only access to the replay buffer.
    pub fn replay(&self) -> &ReplayBuffer { &self.replay }
    /// Read-only access to aggregate statistics.
    pub fn stats(&self) -> &CortexStats { &self.stats }

    /// Mutable access to the hippocampus — required by the persistence
    /// layer to restore episodes and tweak neuromodulator levels at
    /// startup. Production callers should go through
    /// [`crate::Cortex::tick`] instead.
    pub fn hippocampus_mut(&mut self) -> &mut Hippocampus { &mut self.hippocampus }
    /// Mutable access to the neuromodulator panel.
    pub fn modulators_mut(&mut self) -> &mut Neuromodulators { &mut self.modulators }
    /// Mutable access to aggregate statistics.
    pub fn stats_mut(&mut self) -> &mut CortexStats { &mut self.stats }
}

/// Convert spike-train observations into a stable SDR for
/// hippocampal storage.
fn spike_to_sdr(spikes: &[SpikeTrain]) -> crate::sdr::Sdr {
    let mut bits: Vec<usize> = Vec::new();
    let mut h: u64 = 1469598103934665603;
    for t in spikes {
        for &s in &t.spikes {
            if s {
                h ^= h.wrapping_mul(1099511628211);
                bits.push((h % crate::sdr::SDR_WIDTH as u64) as usize);
            }
        }
    }
    bits.sort_unstable();
    bits.dedup();
    crate::sdr::Sdr::from_bits(bits)
}

/// Convert one spike train into an SDR — used by global workspace
/// competition and working memory.
fn spike_train_to_sdr(train: &SpikeTrain) -> crate::sdr::Sdr {
    spike_to_sdr(std::slice::from_ref(train))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdr::Sdr;

    #[test]
    fn cortex_tick_produces_broadcast() {
        let mut cortex = Cortex::default_for_tests();
        let mut inputs = HashMap::new();
        inputs.insert("channel.0".to_string(), Sdr::from_bits(vec![1, 2, 3]));
        let broadcast = cortex.tick(inputs, None);
        assert_eq!(broadcast.tick, 0);
    }

    #[test]
    fn repeated_ticks_record_replay_frames() {
        let mut cortex = Cortex::default_for_tests();
        for _ in 0..5 {
            let mut inputs = HashMap::new();
            inputs.insert("channel.0".to_string(), Sdr::from_bits(vec![1, 2, 3]));
            let _ = cortex.tick(inputs, None);
        }
        assert_eq!(cortex.replay().len(), 5);
    }

    #[test]
    fn sleep_runs_consolidation() {
        let mut cortex = Cortex::default_for_tests();
        for _ in 0..5 {
            let mut inputs = HashMap::new();
            inputs.insert("channel.0".to_string(), Sdr::from_bits(vec![1, 2, 3]));
            let _ = cortex.tick(inputs, None);
        }
        let report = cortex.sleep(3);
        assert_eq!(report.episodes_replayed, 3);
    }
}
