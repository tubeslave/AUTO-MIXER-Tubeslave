# Proposal: fx-send-snapshot-freshness-gate-20260605

## 1. Short thesis

Do not let FX-send proposals promote from stale `fx_inserts_snapshot` schema
evidence, even when the schema family is mapped.

## 2. What I understood

The Product Layer snapshot bridge gives `fx_send_proposal` useful read-only FX
schema evidence. That evidence is only safe for proposal-quality decisions when
the source snapshot is fresh. A stale snapshot should be analysis evidence, not a
proposal promotion gate pass.

## 3. Proposed solution

When normalized schema evidence includes `snapshot_type="fx_inserts_snapshot"`,
check `snapshot_cache_freshness_status`. If it is not `fresh`, add
`fx_inserts_snapshot_cache_not_fresh`, which keeps the recommendation
`analysis_only`.

## 4. Likely files to touch

- `backend/fx_send_proposal.py`
- `backend/processing_registry.py`
- `tests/test_fx_send_proposal.py`
- `.ai/briefs/fx-send-snapshot-freshness-gate-20260605.md`
- `Docs/adr/fx-send-snapshot-freshness-gate-20260605.md`

## 5. Alternatives considered

Trust mapped schema even if stale. Rejected because FX slot models and inserts
can change outside Automixer.

Require both explicit freshness and snapshot freshness for every caller.
Rejected for now because legacy callers may not pass snapshots yet.

## 6. Risks

This may keep more recommendations in `analysis_only` until callers pass fresh
snapshots. That is acceptable for live sound safety and explainability.

## 7. Test plan

- Compile changed files.
- Focused `tests/test_fx_send_proposal.py`.
- Broad Product Layer/registry proposal tests.
- Full standard pytest suite.

## 8. Where the other agent may disagree

Kimi may want stale snapshot severity to be high. I keep it medium because it is
a promotion blocker, not an immediate feedback or headroom emergency.
