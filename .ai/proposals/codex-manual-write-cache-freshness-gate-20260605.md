# Proposal: manual-write-cache-freshness-gate-20260605

## Short thesis

Move cache freshness enforcement into the supervised apply path so stale cache
cannot be used as a write pre-read or rollback baseline.

## What I understood

The Automixer is moving toward supervised and eventually autonomous control, but
the only live write path currently allowed is `fader` and `gain` through the
manual gate. Producer-level freshness checks are useful, but the central apply
path must also fail closed because not every future producer will be perfectly
gated.

## Proposed solution

Add server-side helpers that compute cache freshness for a mixer address using
`state_timestamps`. Attach that evidence to `_manual_write_read_value()` when it
returns `mixer_state_cache`. In `_apply_manual_console_write()`, block cache
pre-read unless freshness is `fresh`. In `apply_operator_proposal`, use the same
read result before calling the manual gate, so mocked or alternate producers
cannot bypass freshness at apply time. Add `state_timestamps` support to the
Mixing Station client.

## Likely files to touch

- `backend/server.py`
- `backend/handlers/product_state_handlers.py`
- `backend/mixing_station_client.py`
- `tests/test_server.py`
- `tests/test_operator_product_state.py`

## Alternatives considered

- Rely only on producer-level freshness gates. Rejected because the apply path is
  the safety boundary for all current and future producers.
- Always query physical readback before writes. Preferable when available, but
  not every supported client exposes `query_parameter`; cache fallback still
  needs a freshness contract.
- Allow unknown timestamps for legacy clients. Rejected for live safety; runtime
  clients that use cache-backed pre-read should timestamp cache updates.

## Risks

- Legacy or fake clients without timestamps will now block cache-backed writes.
- Mixing Station gain read parity remains incomplete for WING-style trim
  addresses and may require a later read DTO/parity pass.

## Test plan

- Focused manual write tests for fresh cache, stale cache, and physical readback.
- Focused operator proposal apply tests for stale cache pre-read.
- Broader operator/server/readiness tests.
- Full suite and `git diff --check`.

## Where the other agent may disagree

Kimi may prefer adding physical queries to every apply attempt. My narrower
recommendation is to keep physical readback as preferred evidence while making
cache fallback explicit, timestamped, and fail-closed.
