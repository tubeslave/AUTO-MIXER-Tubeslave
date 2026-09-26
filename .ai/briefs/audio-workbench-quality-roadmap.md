# Audio Workbench Quality Roadmap v0.2–v1.0

## Goal
Move Audio Workbench from metric-aware offline mixing toward context-aware, causal, auditable mixing. The system must understand hierarchy and events, diagnose causes through controlled interventions, compare counterfactuals, express uncertainty, learn from accepted/rejected A/B decisions, and preserve whole-song musical structure.

## Principles
1. Metrics are evidence, never artistic verdicts.
2. No correction without a stated observation and causal hypothesis.
3. Every important intervention competes against bypass/no-change.
4. Local improvements must pass whole-song regression.
5. Musical role is section-dependent.
6. Learned observers receive authority capability-by-capability after calibration.
7. Reference matching targets traits/ranges, not waveform or master-EQ cloning.
8. Offline experimentation never grants live-write authority; Automixer safety remains authoritative.

## Milestone 1 — Hierarchical Song Model
Deliver:
- Track -> instrument -> bus -> mix graph.
- Explicit role and priority per section.
- Source-attribution query: mix anomaly -> bus -> instrument -> track contributors.
- Manifest schema v2 with stable IDs and duplicate/linked-mic relationships.
Tests:
- known synthetic hierarchy;
- missing/duplicate IDs rejected;
- section role override wins over global prior;
- contributor sums reconcile to bus/mix measurements.

## Milestone 2 — Event-aware Analysis
Deliver:
- onset/event detector for drums, bass/guitar attacks and vocal phrase activity;
- event envelopes: attack/body/decay, local crest, spectral centroid/bands;
- cross-track event overlap and timing relations;
- event-aware dynamics report.
Tests:
- impulse/train timing accuracy;
- gain invariance of timing;
- no event-quality verdict from detection alone.

## Milestone 3 — Causal Experiment Engine
Deliver:
- structured Problem -> Cause Hypothesis -> Intervention -> Expected effect -> Risks;
- bounded experiment families: gain, EQ, dynamic EQ/ducking proxy, compression, automation/no-change;
- counterfactual candidate always included;
- target metric + protected metrics + regression gates;
- accept/reject/unknown outcomes.
Tests:
- no hypothesis means no autonomous experiment;
- no-change always present;
- candidate improving target but damaging protected metric is rejected;
- source bytes never overwritten.

## Milestone 4 — Dynamic Masking Graph
Deliver:
- time-frequency masking edges;
- section/event-aware edge strength;
- protect/candidate direction from musical priority;
- source attribution before EQ proposal.
Tests:
- controlled overlapping tones;
- inactive source produces no active masking edge;
- priority reversal changes proposed intervention direction, not measurements.

## Milestone 5 — Macro Mix Director
Deliver:
- song energy curve and section contrast model;
- verse/chorus/solo/final lift relationships;
- whole-song regression after local edits;
- transition checks.
Tests:
- synthetic section-level changes;
- local gain fix that collapses chorus lift is rejected.

## Milestone 6 — Perceptual Blind A/B Judge
Deliver:
- deterministic random A/B ordering;
- separate axes: balance, punch, clarity, vocal placement, low end, harshness, depth, stereo, groove, macro contrast;
- identical-file catch trials;
- answer may be tie/uncertain;
- capability calibration for Qwen2-Audio/Audiobox/future observers.
Tests:
- A/B reversal consistency;
- identical files cannot produce confident preference;
- failed capability receives zero decision authority.

## Milestone 7 — Reference DNA / Multi-reference
Deliver:
- ReferenceProfile v2 with traits by domain and section;
- composite reference: drums from A, guitars from B, master density from C;
- target ranges rather than exact point matching;
- contributor-level hypotheses before bus/master correction.
Tests:
- pure level change does not become tonal delta;
- contradictory references produce a range/conflict, not silent averaging.

## Milestone 8 — Uncertainty Engine
Deliver:
- observation_confidence, cause_confidence, intervention_confidence, evaluation_confidence;
- escalation rules: low cause confidence -> diagnostic experiment; low evaluation confidence -> more evidence/human A/B;
- confidence calibration history.
Tests:
- low-confidence action cannot auto-accept;
- disagreement between observers lowers evaluation confidence.

## Milestone 9 — Preference / Decision Memory
Deliver:
- store context -> intervention -> A/B result -> reason;
- accepted and rejected examples;
- nearest-context retrieval;
- user/genre/project scopes kept separate.
Tests:
- rejected decisions are retrievable;
- preferences from one project do not silently become universal rules.

## Milestone 10 — Autonomous Whole-song Loop
Deliver:
Observe -> model context -> prioritize problem -> causal experiment -> blind evaluation -> regression -> accept/reject -> memory -> repeat.
Stop conditions:
- no high-confidence meaningful problem remains;
- next changes are below audibility/importance threshold;
- repeated experiments fail to improve protected objectives;
- operator stop.
Finalization still requires existing technical coverage and immutable render identity.

## Implementation order
Phase A: milestones 1–3 together. They are the quality foundation.
Phase B: milestones 4–5.
Phase C: milestone 6.
Phase D: milestones 7–9.
Phase E: milestone 10 and integration with Automixer safety bridge.

## Definition of quality progress
A milestone is not successful because code exists. It must improve decisions on a held-out set of controlled perturbations and real multitrack sessions without increasing regression rate. Premiera is one integration case, not the training/evaluation set.
