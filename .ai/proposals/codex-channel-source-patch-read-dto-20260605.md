# Codex Proposal: Channel Source Patch Read DTO - 2026-06-05

## Short thesis

Add cached source-patch visibility to the channel Product Layer so future routing and source-aware decisions can inspect input assignments without gaining write authority.

## What I understood

The autonomous Automixer goal needs full console awareness. Channel source assignment is currently only partly represented by static Dante labels. The WING cache may already contain main and ALT input patch addresses, but the Product Layer does not normalize them into the channel inventory contract.

## Proposed solution

- Add a transport-free `source_patch` builder in `operator_product_state.py`.
- Read only cached `/ch/N/in/conn/*` addresses.
- Add per-field `read_coverage` and `cache_freshness`.
- Keep routing/source-patch mutation blocked with policy markers.
- Update read-surface inventory, fixture evidence, tests, and ADR.

## Likely files to touch

- `backend/operator_product_state.py`
- `backend/read_surface_inventory.py`
- `backend/read_surface_fixture_evidence.py`
- `tests/test_operator_product_state.py`
- `tests/test_read_surface_inventory.py`
- `tests/test_read_surface_fixture_evidence.py`
- `Docs/adr/channel-source-patch-read-dto-20260605.md`

## Alternatives considered

- Query WING through `get_channel_input_routing`. Rejected because it sends OSC read requests and the current slice must be cached/read-only.
- Change routing proposals directly. Rejected because route changes are destructive and not part of the supervised fader/gain executor.
- Leave fallback `In NN` labels only. Rejected because autonomous source-aware routing decisions need explicit cached source group/channel evidence.

## Risks

- Consumers may mistake cached source patch for complete physical proof. The DTO therefore keeps `full_autonomous_ready=False`, coverage flags, freshness metadata, and inventory blockers.

## Test plan

- Add channel inventory tests for cached main/ALT source patch and missing source patch.
- Run operator Product Layer, read inventory, and fixture evidence tests.
- Run `py_compile`, `git diff --check`, and the full backend suite.

## Where the other agent may disagree

Kimi may ask to build physical WING query fixtures first. That is required before promotion, but cached DTO coverage is the smaller prerequisite that lets proposals reason from current state without contacting the console.
