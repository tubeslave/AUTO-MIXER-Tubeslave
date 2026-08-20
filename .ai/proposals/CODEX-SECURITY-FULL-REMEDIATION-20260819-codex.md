# Codex proposal — full Codex Security remediation

## 1. Short thesis

Treat the 143 findings as a finite evidence ledger, but implement fixes by
shared trust boundary. Default legacy live-write and remote-control surfaces to
fail closed; preserve only paths that already have operation-specific
authorization, exact peer binding, readback, rollback, and tests.

## 2. What I understood

The current progress report contains substantial verified remediation but is
not exhaustive. A later zero-finding checkpoint covered only selected surfaces.
The collaborative tree is heavily dirty, so broad patch application or cleanup
would destroy unrelated work. The result must remain offline software proof and
cannot promote WING/dLive readiness.

## 3. Proposed solution

Build a machine-readable occurrence matrix from sealed `findings.json`, the
progress report, checkpoint reports, current diffs, and tests. For each
occurrence, identify the current source-to-sink path and assign it to one of six
root-cause classes in the brief.

Remediate in this order:

1. High-severity live mutation, transport replacement, false-success, and
   unsafe proof/evidence paths.
2. Credential exposure, destination/peer binding, filesystem escape, and native
   memory safety.
3. Training, archive, audio, and parser resource bounds.
4. Replay integrity, strict boolean/numeric validation, and fail-closed
   readiness.
5. Medium/low legacy tool, MCP, Telegram, and offline utility findings.

Where a legacy tool has no canonical supervised transaction integration, make
its mutation mode unavailable and retain an explicit dry-run/plan mode. Do not
invent a second authorization framework merely to preserve old direct writes.

## 4. Likely files to touch

- `backend/server.py`, selected `backend/handlers/*.py`, transport clients,
  session/config stores, snapshot and routing utilities.
- `backend/ml/*`, replay and sandbox modules under `ai_mixing_pipeline/`.
- Legacy live tools under `tools/`, MCP/Telegram bridges, and native CoreAudio
  source.
- Focused tests plus one security-ledger/report module.
- `.ai/`, `Docs/adr/`, `Docs/reports/`, and autopilot truth surfaces only after
  final source/test state is known.

## 5. Alternatives considered

- Apply all generated workbench patches mechanically: rejected because several
  patches target the same heavily modified files and the workbench already
  refused persistence against unrelated dirty changes.
- Close only the 12 checkpoint findings: rejected because the sealed scan has
  143 reportable findings and checkpoint coverage is explicitly partial.
- Preserve every legacy live tool by adding bespoke confirmation prompts:
  rejected because reusable phrases are not operation-specific authority and
  do not provide readback or rollback.

## 6. Risks

- Disabling unsupported legacy mutation paths can change undocumented operator
  workflows. Mitigation: preserve dry-run plans and state explicit unavailable
  reasons.
- Overlapping fixes can cause duplicate or inconsistent guards. Mitigation:
  centralize reusable validators but keep sink-local assertions.
- Full validation is expensive and the filesystem is nearly full. Mitigation:
  run targeted checks first and generate artifacts in bounded temporary roots.
- Current dirty-tree CP05 failures are expected and must not be mislabeled as
  security regressions.

## 7. Test plan

- Add exploit-focused tests proving rejection before transport, write, file,
  subprocess, allocation, or authority sinks.
- Preserve positive bounded behavior and explicit proposal-only behavior.
- Run security modules by class, relevant owning suites, no-WING quick, standard
  full pytest, frontend tests/build where touched, dependency audits, native
  syntax/build checks, and diff hygiene.
- Reconcile all 143 occurrence IDs against evidence, then request a fresh full
  current-checkout Codex Security scan when the connector is available.

## 8. Where the other agent may disagree

Kimi may prefer a new central capability broker or a clean-room branch. I
expect the safer immediate route is sink-local fail-closed remediation on the
current collaborative tree, followed by a later architecture consolidation;
moving the existing dirty state now would risk losing or misbinding work.
