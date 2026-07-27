# System Inspector Product

Status: Phase 0 / offline snapshot replay vertical

## Product goal

System Inspector gives the operator an evidence-backed view of how a macOS
computer, Dante topology and WING Rack patch are connected. It detects
observable mismatches, explains impact, lists evidence and recommends manual
actions. It does not fix anything.

Inspector is the first mode in the product progression:

1. Inspector Mode — read only
2. Proposal Mode — recommendations
3. Soundcheck Mode — controlled analysis
4. Assist Mode — monitoring
5. Automix Mode — supervised actuation

Inspector evidence is a required safety input, not live authority. Low routing
confidence, unknown clock state, unresolved critical findings or insufficient
coverage must block automatic correction promotion.

## Phase 0 outcome

The implemented vertical slice:

- loads a safe YAML show profile;
- loads versioned simulated macOS and WING snapshots;
- loads an imported Dante JSON topology snapshot;
- loads an immutable known-good baseline;
- builds a normalized graph;
- compares expected and observed routes;
- detects missing devices, network-interface drift, sample-rate mismatch,
  routing drift, duplicated Playback L/R and Click/Cues routed to Main;
- emits evidence-backed findings;
- calculates an integer readiness score only when coverage is at least 50%;
- always displays coverage, confidence, unknown checks, timestamp and
  freshness;
- renders deterministic CLI and JSON output.

Phase 0 performs no device, network, audio or backend contact.

## Evidence levels

These levels are distinct and must not be promoted by inference:

1. `NETWORK_REACHABLE`
2. `DANTE_DISCOVERED`
3. `SUBSCRIPTION_PRESENT`
4. `CLOCK_SYNCED`
5. `AUDIO_SIGNAL_CONFIRMED`

An imported subscription may support level 3 with explicit import provenance.
It does not establish live discovery, clock sync or audio signal.

## Finding contract

Every non-INFO finding contains:

- stable ID;
- severity and category;
- title and summary;
- evidence source, path and value;
- confidence;
- impact;
- recommended manual action;
- `automatic_fix_available: false`;
- manual-confirmation requirement;
- missing evidence when status is `UNKNOWN`.

`UNKNOWN` is not OK. It records that the evidence needed for a check is absent
or insufficient.

## Readiness semantics

Readiness is an operator aid, not a failure prediction.

- Category scores use visible fixed penalties.
- Unknown categories remain visible and reduce the overall score through the
  observed-coverage multiplier.
- `observed_coverage` is the proportion of categories with enough evidence.
- Below 50% coverage, the overall score is `null` and status is `UNKNOWN`.
- At or above 50% coverage, any remaining unknown check keeps the overall
  status `UNKNOWN`; an evidenced critical/high fault takes precedence as
  `MISMATCH`.
- Confidence is reduced by incomplete coverage.
- A score never grants WING or Dante write authority.

## CLI

Repository entrypoint:

```bash
PYTHONPATH=backend python tools/automixer.py inspector scan \
  --profile tests/fixtures/system_inspector/healthy_system/profile.yaml \
  --snapshot tests/fixtures/system_inspector/healthy_system \
  --format json
```

Equivalent module entrypoint:

```bash
PYTHONPATH=backend python -m backend.system_inspector inspector replay-snapshot \
  --profile tests/fixtures/system_inspector/healthy_system/profile.yaml \
  --snapshot tests/fixtures/system_inspector/playback_right_duplicate
```

`--read-only` is always true in v1. No disable flag exists.

## Non-goals in Phase 0

- live WING connection;
- network discovery or probing;
- reading Dante Controller state;
- licensed SDK/API use;
- CoreAudio capture;
- Ableton control or project parsing;
- HTML/PDF or UI delivery;
- continuous monitoring;
- baseline mutation;
- automatic or supervised fixes.

Each item above is behind a later checkpoint and cannot be inferred from replay
tests.
