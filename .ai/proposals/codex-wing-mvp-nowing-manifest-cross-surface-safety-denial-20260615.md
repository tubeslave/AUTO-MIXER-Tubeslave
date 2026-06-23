# Codex Proposal: WING MVP no-WING Manifest Cross-Surface Safety Denial

## 1. Short thesis

Extend the no-WING manifest cross-surface check so both JSON and Markdown
surfaces must prove readiness closure and live authority remain denied.

## 2. What I understood

The operator-facing no-WING bundle is part of the current WING read-only proof
handoff. It can prepare real hardware work, but it must not be mistaken for
authorization to close readiness or start live control.

## 3. Proposed solution

Add four explicit computed verdict fields:

- `artifact_manifest_summary_json_cross_surface_readiness_closure_denied`
- `artifact_manifest_summary_json_cross_surface_live_authority_denied`
- `artifact_manifest_summary_md_cross_surface_readiness_closure_denied`
- `artifact_manifest_summary_md_cross_surface_live_authority_denied`

Each field must be true for the aggregate cross-surface check to pass.

## 4. Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- Council docs and WING MVP report/index entries

## 5. Alternatives considered

Only checking line presence was rejected because a line rendered with `true`
would still satisfy the current surface check. Adding a separate external
validator was rejected because this contract is already enforced in the no-WING
summary/manifest code path.

## 6. Risks

The main risk is expanding a heavily asserted summary contract and missing a
fixture update. Targeted tests should cover the exact changed surface.

## 7. Test plan

- Targeted manifest cross-surface tests.
- Focused no-WING, Project Director, and WING MVP rails tests.
- Full standard pytest.

## 8. Where the other agent may disagree

Kimi may prefer a smaller check that only validates the aggregate safety fields.
I prefer explicit per-surface verdicts because they make stale or unsafe
operator-readable output diagnosable without opening the code.
