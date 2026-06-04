# ADR: Autonomy Evidence Overlay

Date: 2026-06-04

## 1. Context

The project has a static autonomy promotion readiness snapshot and a dynamic
supervised proof artifact report. Static readiness is policy-oriented; proof
reports are evidence-oriented. To move toward supervised AutoMix safely, the
current evidence needs to be visible against the exact promotion targets.

## 2. Options considered

- Merge dynamic proof evidence into static readiness rows.
- Keep proof evidence and readiness disconnected.
- Add a separate read-only overlay.

## 3. Decision

Add `autonomy_evidence_overlay`. It maps current fader/gain proof evidence,
promotion evidence, and remediation actions onto
`fader_supervised_promotion` and `gain_trim_supervised_promotion`.

## 4. Why this won

It makes the path from proof artifacts to promotion review explicit while
preserving a clean separation between evidence and policy. Physical candidates
can become manual safety-review candidates; blocked or legacy artifacts become
remediation-required rows. Nothing is promoted automatically.

## 5. Rejected alternatives

Merging into static readiness was rejected because current artifact state should
not mutate policy rows. Leaving the views disconnected was rejected because it
makes promotion review harder to audit. Automatic safety updates were rejected
because physical proof still requires manual safety review.

## 6. Implementation plan

1. Add a pure overlay builder.
2. Map fader/gain proof rows to promotion target ids.
3. Reuse nested promotion evidence and remediation plans when present.
4. Embed the overlay in proof report output.
5. Register it as shadow-only in the processing registry.

## 7. Test plan

- Compile touched Python files.
- Run focused overlay/report/registry tests.
- Run broader supervised proof/readiness/no-WING tests.
- Run the standard full pytest command.

## 8. Risks and rollback

Risk: manual review candidates are mistaken for permission. The overlay sets
`live_write=false`, `live_widening_allowed=false`,
`automatic_safety_update_allowed=false`, and keeps autonomous-ready ids empty.
Rollback is to remove overlay embedding and registry registration; no live path
is introduced.
