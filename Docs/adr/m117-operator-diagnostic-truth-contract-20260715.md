# ADR: M117 Operator Diagnostic Truth Contract

## 1. Context

M116 truthfully recorded an operator-attended route diagnostic that contacted
the real WING, backend, OSC and CoreAudio, but produced no route artifact,
reset, correction, live write or standing authority. The structured state and
ledger result classify the extra capture facts as operator-observed and not
independently machine-verified. Several current prose surfaces still state the
capture result without that qualifier, and one present-tense contract says all
externally blocked ledger rows must have no hardware contact. M116 also loses
direct test coverage as soon as another ledger row becomes latest.

Kimi CLI was attempted in read-only plan mode but returned HTTP 402 before a
proposal was produced. The second proposal and reciprocal critique therefore
come from a clearly labelled independent read-only Codex reviewer.

## 2. Options considered

- Leave the prose and rely on structured fields.
- Rewrite the M116 ledger row to no-contact.
- Generalize the authority schema for all hardware sessions.
- Append a narrow M117 control-plane correction and pin M116 history.

## 3. Decision

Use STATE as the canonical current blocker, append one no-contact M117 ledger
row, synchronize every tested current mirror, and preserve the immutable M116
row as the typed operator-attended diagnostic record. Add durable historical
M116 coverage plus current-provenance wording rails. Also close the stale M116
plan status and describe the M113 identity gate as fail-closed.

## 4. Why this won

It corrects operator-facing safety truth without falsifying physical history,
widening authority or introducing a new runtime abstraction. It remains a
named control-plane gap allowed by the M105 anti-loop guard.

## 5. Rejected alternatives

Structured truth does not excuse contradictory operator prose. Rewriting the
append-only M116 row would destroy valid evidence. A generalized authority
schema is unnecessary because the current typed ledger branches already
distinguish offline, backend-only and operator-approved rows.

## 6. Implementation plan

Update STATE wording and validation narrative; append the M117 row; update
current blocker, timestamp, ledger and validation mirrors; close M116 CP04-CP06;
correct ROADMAP gate wording; add two regression rails; validate on the source
checkpoint; commit; release the lease; then recompute and commit only the CP05
marker pair.

## 7. Test plan

Run focused M116/M117 truth tests, all WING MVP rails, JSON/JSONL validation,
the standard full backend suite, the canonical no-WING manifest verifier,
memory lint and `git diff --check`. Perform independent read-only diff review
and verify dynamic CP05 on the final clean marker-only HEAD.

## 8. Risks and rollback

Partial mirror sync can leave contradictory instructions, and any non-marker
edit invalidates CP05 until the marker-only rebind. Roll back the M117 source
checkpoint and its paired CP05 rebind together. Never reinterpret the M117
offline row as erasing M116 contact or granting hardware authority.
