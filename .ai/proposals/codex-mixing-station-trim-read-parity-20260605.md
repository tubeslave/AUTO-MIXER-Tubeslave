# Proposal: mixing-station-trim-read-parity-20260605

## Short thesis

Normalize Mixing Station gain/trim through the same `/ch/{N}/in/set/trim`
contract used by WING and the supervised manual gate.

## What I understood

Autonomy depends on one consistent read model for pre-read, cache freshness,
rollback, and proposal evidence. The Mixing Station client had timestamped
cache support, but gain writes and reads still used a legacy `/ch/{N}/gain`
alias that did not match the manual gate's canonical trim address.

## Proposed solution

Add canonical state-address normalization in `backend/mixing_station_client.py`.
Map incoming Mixing Station `ch.N.preamp.gain` and gain/trim aliases to
`/ch/{N}/in/set/trim`. Map outgoing canonical trim to `ch.N.preamp.gain`.
Update `set_channel_gain()`, add `get_channel_gain()` and `query_parameter()`,
and include trim in the initial read-only state scan. Route callbacks through
canonical keys as well as raw Mixing Station addresses.

## Likely files to touch

- `backend/mixing_station_client.py`
- `tests/test_mixing_station_client.py`
- `.ai/briefs/`
- `.ai/proposals/`
- `Docs/adr/`

## Alternatives considered

- Keep `/ch/{N}/gain` as a separate cache key. Rejected because it fragments
  cache freshness and rollback evidence.
- Change the manual gate to accept Mixing Station-specific aliases. Rejected
  because the gate should stay mixer-agnostic and canonical.
- Defer until hardware testing. Rejected because address normalization is safe
  to prove offline with dummy sockets.

## Risks

- Mixing Station's preamp gain path may vary by mixer model. The change keeps
  this as read/write parity for the existing client's documented internal path
  and does not promote any new action family.
- Physical behavior still requires later hardware/proxy proof.

## Test plan

- Unit tests for incoming preamp gain cache normalization.
- Unit tests for outgoing canonical trim OSC address construction.
- Unit tests for `query_parameter()` alias normalization.
- Focused supervised gate tests and full suite.

## Where the other agent may disagree

Kimi may prefer leaving Mixing Station outside the supervised path until full
hardware proof. My counterpoint is that canonical read parity is a prerequisite
for proof; it does not by itself widen live write permissions.
