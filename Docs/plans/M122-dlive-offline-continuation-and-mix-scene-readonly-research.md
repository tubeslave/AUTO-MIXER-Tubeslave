# M122 — dLive offline continuation and Mix Scene read-only research

Status: IN PROGRESS — A2/A3 offline evidence gates implemented; hardware unavailable; no execution authority

Date: 2026-08-04

## Objective

Continue the second-console track for Allen & Heath dLive without losing the
evidence already obtained, and evaluate a Mix Scene Model without coupling the
research to console transport or widening current live authority.

The two workstreams are deliberately independent:

1. **dLive parity track** — finish the smallest real read/apply/readback/rollback
   proof when the console is available again.
2. **Mix Scene research track** — validate a transport-free semantic scene model
   against existing audio features and replay fixtures.

Success on either track cannot promote the other.

## Current truth and frozen boundary

- Hardware is currently disconnected. No route lookup, TCP connection, MIDI
  send, discovery scan or physical proof is authorized by this plan.
- Accepted read-only dLive evidence remains the CDM32 + C3500, firmware 2.0,
  show `lv 1 master`, Input 1 `KICK` observation recorded in
  `Docs/runbooks/DLIVE_READONLY_PREFLIGHT.md:32-41`.
- The paired-online Surface attempt is terminal: the compact fader packet was
  sent once, the operator saw no movement, and four readbacks remained raw 106.
  The receipt and required next diagnostics are recorded at
  `Docs/runbooks/DLIVE_READONLY_PREFLIGHT.md:147-158`.
- Normal `DLiveClient` remains read-only/zero-send. The only mutation surface is
  the sealed, one-shot restore proof; all other dLive write families remain
  blocked (`backend/dlive_client.py:1-12`, `backend/dlive_client.py:99-101`).
- No previous approval, confirmation token or proof plan may be reused.
- The active WING milestone, its scores and its authority are unchanged.

## Workstream A — dLive while hardware is offline

Owner: backend/control-plane implementation owner. Reviewer: independent safety
reviewer. Operator is not required until A4.

### A1. Preserve terminal evidence

- Keep the terminal receipt, SHA-256, WAL result and operator observation as
  immutable no-op evidence.
- Add no interpretation that a successful socket send proves console acceptance.
- Assert in offline tests that the consumed approval cannot be replayed and the
  terminal receipt grants no standing authority.

Exit: the no-op remains terminal, recovery is not pending, and all operator
surfaces describe the same blocker.

### A2. Separate protocol encoding from delivery diagnosis

Implementation checkpoint 2026-08-04: `backend/dlive_midi_capture_evidence.py`
and `tools/dlive_review_midi_capture.py` now review only an operator-supplied
capture file. They require a known-good client, selected-endpoint ingress,
direction/timestamp/endpoint identity and an exact byte match with the planned
compact NRPN. The accepted report explicitly leaves console acceptance and
physical movement false and never opens a socket or capture interface.

- Build transport-free golden vectors for documented compact NRPN request/write
  packets and running-status parsing.
- Add an offline parser for a future operator-owned packet capture. It may read a
  supplied capture file but must not open an interface or start capture itself.
- Record direction, endpoint, timestamp, exact bytes and request/response
  correlation. Never infer physical application from outbound bytes alone.
- Compare AUTO-MIXER bytes with a known-good dLive client capture when one is
  supplied. Do not add speculative packet variants to the live runner.

Exit: encoder correctness and wire delivery are independently reportable.

### A3. Harden the next-proof preflight without contacting hardware

Implementation checkpoint 2026-08-04: the restore-proof approval and its
SHA-bound confirmation now include effective security mode, whether security
was changed for setup, reboot-after-change confirmation, and an owner-controlled
accepted wire-review artifact. Both the zero-contact plan validator and the
attended runner reject missing, mismatched or rebound evidence before route
lookup/socket creation. A security change without reboot confirmation is
fail-closed.

- Add/retain fail-closed checks for endpoint/TLS/port, MIDI base channel, user
  profile, source identity, target quantization, downward-only delta, receipt
  ownership, approval freshness and terminal WAL state.
- Add an explicit security-state attestation field and reboot-after-security-
  change attestation. These fields are evidence, not standing authority.
- Require a known-good-client comparison or accepted packet capture before a new
  mutation plan can reach `ready_for_operator_confirmation`.
- Keep the plan validator zero-contact and require a new UUID and SHA-bound token.

Exit achieved offline: focused dLive evidence/proof/client/preflight validation
passes 139 dLive tests, including no-socket capture review and missing/mismatched
security, reboot and wire-evidence rejection. This does not satisfy A4.

### A4. Next attended dLive checkpoint — only after reconnection

This phase is deliberately not executable while the console is off.

Preconditions:

1. Surface and MixRack are paired and online; exact model, firmware, show and
   endpoint are freshly attested.
2. Effective MIDI security is verified. If changed, the affected components are
   rebooted and the setting is rechecked.
3. PA and monitor outputs are physically isolated; selected input is muted;
   operator watches the C3500.
4. A known-good client or durable capture proves what reaches the selected
   endpoint.
5. A fresh GET-only artifact agrees with the physical Surface.
6. A new plan, approval ID and exact SHA-bound confirmation are generated.

Execution remains one bounded cycle:

`fresh preimage -> durable WAL -> one downward apply -> two fresh readbacks ->
CAS -> exact rollback if required -> two fresh readbacks -> revoke -> disconnect`

Stop immediately on mismatch, no movement, timeout, unexpected value, loss of
audio/control freshness, security ambiguity or operator stop. Do not try a
second encoding in the same authority window.

Exit: both physical movement and correlated readback prove apply and restore.
Otherwise record another terminal diagnostic result without promotion.

### A5. dLive promotion sequence

- Promote only the exact input-fader family proven on the exact firmware and
  endpoint; do not generalize to mute, gain, EQ, dynamics, routing or scenes.
- Add the dLive adapter to the existing supervised transaction path only after
  independent review of readback, CAS rollback, disconnect and emergency-stop
  behavior.
- Re-run targeted dLive/transport/safety tests, the full suite, and the no-WING
  release capsule. A green software suite remains separate from hardware proof.
- Record the accepted decision in a dedicated ADR before broadening production
  behavior.

## Workstream B — Mix Scene Model research

Owner: analysis/DSP owner. It must not share an implementation change with the
dLive transport track.

### B1. Architecture decision

Use a dual-path target architecture, introduced first as an independent
read-only module:

```text
Channel-level evidence ───────────────> existing proposal/safety path
           |
           +-> Mix Scene Builder -> diagnostics/snapshot/drift report
```

The scene branch has no mixer client, proposal queue, FSM mutation or actuation
dependency. Its first release is evidence only. The detailed assessment is in
`Docs/reports/MIX_SCENE_MODEL_ARCHITECTURAL_ASSESSMENT_20260804.md`.

### B2. Read-only prototype

Progress 2026-08-04: the first transport-free vertical slice is implemented as
`automixer.instrument_space.v1` in `backend/instrument_space.py`, embedded in
the existing channel inventory and rendered by an inspect-only dashboard tab.
It exposes pan and level when measured. A bounded `AudioCapture` worker now
publishes real per-channel centroid/brightness every 0.5 seconds outside the
realtime callback and UI snapshot path; spectral Y remains null only while
audio is offline, silent, pending or stale. The accepted prototype boundary and
next steps are recorded in `Docs/adr/MIX-SCENE-VISUALIZER-20260804.md`.

Follow-on decision integration is now owned by the separate M123 software
milestone. M123 adds deterministic relation/drift evidence and proposal-review
annotation, but it does not change this plan's dLive hardware blocker or grant
scene-derived execution authority.

- Reuse existing `ChannelMetrics`, role/activity/section data, ERB masking and
  inter-channel metrics.
- Introduce immutable scene object, relation and snapshot DTOs only after an ADR
  accepts their schema.
- Compute deterministic object aggregation, spectral/temporal overlap, masking
  evidence, confidence/uncertainty and snapshot drift.
- Emit a JSON report with evidence provenance and human-readable explanations.
- Use replay/offline fixtures; do not open audio devices or console transports.

### B3. Guardrails

- Dependency tests must reject imports from mixer clients, `live_apply`,
  correction executors, supervised transactions and proposal queue modules.
- A scene result cannot be a `TypedCorrectionAction` and cannot contain apply
  authority.
- Missing/stale audio, source identity changes, unstable grouping or disagreement
  with channel-level analysis forces `advisory_only` or `invalid`.
- Feature flag defaults off; removing the module must leave current behavior and
  tests unchanged.

### B4. Promotion gates

1. Read-only visualizer/report.
2. Advisory diagnostics.
3. Structured non-queueable proposals.
4. Proposal-queue integration only after a separate ADR and evaluation corpus.
5. Soundcheck automation only after hardware-backed proof per action family.
6. LIVE use stays drift-detection/report-only until false-positive, oscillation,
   CPU/latency and operator-usability gates pass.

No phase authorizes unattended LIVE actuation.

## Verification matrix

| Gate | Offline now | Hardware later | Authority effect |
|---|---:|---:|---|
| dLive encoder/parser golden vectors | yes | no | none |
| dLive security/wire evidence contract | yes | capture supplied later | none |
| fresh dLive read-only parity | no | yes | read-only evidence only |
| one-shot fader apply/readback/rollback | no | yes | exact proof only |
| Mix Scene deterministic builder | yes | no | none |
| Mix Scene replay/snapshot/drift tests | yes | no | none |
| Mix Scene advisory UI | yes | no | none |
| Scene-derived console proposal/apply | no | separate future gate | none in M122 |

## Required decision records

- Accept or reject the read-only Mix Scene schema and dual-path boundary.
- Accept or reject dLive known-good-client/capture evidence as the prerequisite
  for a new physical fader proof.
- Create separate ADRs for any future scene-to-proposal integration and any dLive
  write-family promotion.

## Done criteria

M122 planning is complete when:

- the architectural assessment is reviewed by the owner;
- the dLive offline tasks and next attended checkpoint are unambiguous;
- no old dLive authority remains reusable;
- the Mix Scene prototype boundary is transport-free and removable;
- implementation ownership is assigned before code work begins.

M122 does **not** close dLive hardware parity, WING readiness, Mix Scene product
readiness or LIVE automation.
