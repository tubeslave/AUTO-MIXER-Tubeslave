# ADR: WING MVP no-WING Operator Markdown Summary Self-Check

## 1. Context

The WING MVP remains blocked on real read-only WING proof. Offline handoff
artifacts must stay strict because they are the operator path into a physical
preflight session. The no-WING `summary.md` proof-plan section now needs a
machine-readable verdict showing that the required operator-facing lines were
rendered.

## 2. Options considered

- Keep only tests around markdown rendering.
- Add a separate validator command.
- Add a self-check inside no-WING summary generation.

## 3. Decision

Add the self-check inside no-WING summary generation and require its verdict in
Project Director baseline coverage.

## 4. Why this won

The summary generator already has the rendered markdown text and can check the
operator surface without hardware, network, or live authority. The verdict stays
inside the same artifact bundle an operator reviews.

## 5. Rejected alternatives

Markdown-only tests were rejected because they do not make the generated bundle
self-describing. A separate validator was rejected as unnecessary ceremony.

## 6. Implementation plan

- Add required markdown line constants and a summary self-check helper.
- Render self-check fields in `next_wing_proof_plan`.
- Make missing markdown lines fail the summary closed.
- Add Project Director/current command markers for the rendered verdict.

## 7. Test plan

Run focused no-WING summary tests and focused Project Director/WING rail tests,
then run repository whitespace checks and memory lint.

## 8. Risks and rollback

Risk is limited to stale marker wording. Roll back by removing the self-check
fields and matching Project Director markers. This ADR does not grant live WING
authority.
