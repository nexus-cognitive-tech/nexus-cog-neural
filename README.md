# nexus-cog-neural

Brain-like cognitive architecture for the Nexus Cog stack.

The whole system is organised as a single orchestrator ([`Cortex`]) that wires
together every component of a small mammal brain:

| Subsystem | Module | Role |
| --- | --- | --- |
| Thalamus | [`thalamus`] | Sensory relay + gating by salience × attention. |
| Hippocampus | [`hippocampus`] | Fast episodic memory, capacity-bounded, with salience-based eviction. |
| Sleep cycle | [`sleep`] | NREM (replay + consolidation) + REM (cortical re-activation). |
| Basal ganglia | [`basal_ganglia`] | Action selection via winner-take-all + hysteresis. |
| Amygdala | [`amygdala`] | 3-D valence (reward/threat/novelty) per event. |
| Working memory | [`working_memory`] | Miller's 7±2 SDR slots with active maintenance. |
| Attention | [`attention`] | Top-down goal + bottom-up salience → attention map. |
| Neuromodulators | [`neuromodulators`] | Dopamine, serotonin, norepinephrine global signals. |
| Spatial pooler | [`region::spatial_pooler`] | Dense/sparse input → stable SDR (Hebbian learning). |
| Temporal memory | [`region::temporal_memory`] | Predicts the next SDR given a short history. |
| Cortical region | [`region::region`] | `SP ∘ TM`. |
| Hierarchy | [`region::hierarchy`] | DAG of regions, run in topological order. |
| Global workspace | [`global_workspace`] | Baars-style coalition selection. |
| Replay buffer | [`replay`] | Thread-safe recording of every tick for Studio viz. |
| SDR | [`sdr`] | 2048-bit sparse distributed representation + encoders. |

## Why this exists

`nexus-cog-cli` and the `nexus-cog-*` engine crates that predate this one
implemented flat, sequential cognitive operations. Replacing that with a
brain-like architecture is a prerequisite for the upcoming
**Nexus Cog Cloud** (multi-agent broadcast, inter-agent plasticity, valence
routing) and **Nexus Cog Studio** (real-time activation maps, replay,
anatomical metaphor, Hebbian-weight visualisation).

## Quick start

```no_run
use nexus_cog_neural::Cortex;
use nexus_cog_neural::Sdr;
use std::collections::HashMap;

let mut cortex = Cortex::default_for_tests();
for tick in 0..10 {
    let mut inputs = HashMap::new();
    inputs.insert("channel.0".to_string(), Sdr::from_bits([1, 2, 3, 4, 5]));
    let broadcast = cortex.tick(inputs);
    println!("tick {}: {} regions competed, winner = {:?}",
        broadcast.tick, broadcast.coalition.members.len(), broadcast.chosen_action);
}

// One sleep cycle consolidates hippocampal episodes into the cortex.
let report = cortex.sleep(3);
println!("replayed {} episodes ({} unique), target overlap {:.2}",
    report.episodes_replayed, report.unique_patterns, report.avg_target_overlap);
```

## SDR

Every state in the brain is a 2048-bit [`Sdr`] (Numenta's default). SDRs are
the lingua franca: thalamic input, cortical output, hippocampal episodes,
working-memory slots, attention bias and valence tags are all SDRs.

The foundational metrics are:

* **Overlap** — `|a ∩ b|`
* **Semantic similarity** — `overlap / sqrt(|a| · |b|)`
* **Jaccard / Tanimoto distance** — used by the spatial pooler when
  sparsity drifts

## Encoders

Real-world values are mapped to SDRs by [`Encoder`]
implementations. [`ScalarEncoder`] bucketed into adjacent slots with
configurable overlap, [`DateEncoder`] uses three periodic encoders for
day/hour/minute, [`LogEncoder`] compresses log-scale latencies,
[`CategoryEncoder`] reproduces stable random SDRs for categorical IDs, and
[`CoordinateEncoder`] fuses N-D continuous coordinates.

## Bio-fidelity vs utility

Every component is named after the brain region it implements, but
**this is not a neuroscience simulator** — it is a cognitive architecture
that borrows the right amount of structure from biology to be useful for
multi-agent cognitive networks. Permanences decay. Norepinephrine widens
the spotlight. Replay consolidates. The shape is real even when the
numbers are tuned.

## Tests

47 unit tests cover the SDR primitives, all encoders, every brain region,
the cortex orchestrator and the full sleep cycle.

```bash
cargo test --lib
```
