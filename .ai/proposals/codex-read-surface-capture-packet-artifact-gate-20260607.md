# Proposal: Read-Surface Capture Packet Artifact Gate

## 1. Short thesis

Make the generated read-surface capture packet a validated no-WING artifact, not
just an offline smoke output.

## 2. What I understood

Automixer needs broad read-surface physical proof before live console autonomy
can safely widen. The current capture packet identifies the next operator
physical-readback target, but the release check should verify the generated
artifact itself before it passes.

## 3. Proposed solution

- Generate the capture packet as JSON inside no-WING release runs.
- Add a schema validator for the generated packet file.
- Require read-only/operator-required flags and false live/execution/autonomy
  flags.
- Require a consistent next surface/action pair, physical fixture contract,
  blockers, forbidden actions, and focused validation command.
- Add the validator result to `summary.json` and `summary.md`.

## 4. Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/read-surface-capture-packet-artifact-gate-20260607.md`
- `Docs/adr/read-surface-capture-packet-artifact-gate-20260607.md`
- `Docs/reports/AUTOMIXER_READ_SURFACE_CAPTURE_PACKET_ARTIFACT_GATE_20260607.md`
- `Docs/INDEX.md`

## 5. Alternatives considered

- Keep relying on capture packet unit tests only.
- Validate only the underlying capture plan.
- Let acceptance-template and fixture-review gates catch bad inputs later.

## 6. Risks

The main risk is overstating progress. This gate proves a generated offline
operator packet is structurally safe; it does not prove real console readback or
grant live authority.

## 7. Test plan

- py_compile for the no-WING tool and capture packet builder.
- Focused pytest for valid/missing/unsafe capture-packet artifacts.
- Broader no-WING focused pytest.
- no-WING quick run.
- Memory lint, diff check, and full backend pytest.

## 8. Where the other agent may disagree

Kimi may argue that this is another offline gate rather than direct hardware
progress. The counterpoint is that physical readback should only start after the
operator packet contract is machine-validated and fail-closed.
