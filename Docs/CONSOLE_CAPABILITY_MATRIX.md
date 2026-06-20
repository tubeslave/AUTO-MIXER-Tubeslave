# Console Capability Matrix

Status: WING/live authority snapshot

| Console category | Current maturity | Source files | Allowed unattended? | Allowed supervised WING live? | Required next gate | Hardware evidence |
| --- | --- | --- | --- | --- | --- | --- |
| Raw state/read-only connection | SPECIFIED | `tools/wing_readonly_preflight_runner.py`, `backend/wing_readonly_preflight_contract.py` | read-only validation only | read-only with operator/session setup | M001/M002 real artifact | none in setup |
| Physical readback | SPECIFIED | physical readback contract/intake modules | no hardware contact unattended | read-only under runbook | accepted real artifact review | none |
| Channel fader | PROTOTYPE | `backend/wing_client.py`, `backend/operator_mode_policy.py` | no live write | yes only after explicit approval | supervised fader proof | none |
| Preamp trim/gain-down | PROTOTYPE | `backend/wing_client.py`, gain proposal code | no live write | yes only after explicit approval and gain-down bounds | supervised gain-down proof | none |
| Gain-up / boost | SPECIFIED | `CLAUDE.md`, gain safety docs | no | no by default | explicit opt-in plus hardware proof | none |
| Pan | PROTOTYPE | `backend/operator_mode_policy.py`, pan proposal code | no | no for WING MVP | separate ADR and hardware proof | none |
| Mute/unmute | QUARANTINED | recent quarantine commits and safety docs | no | no | separate ADR and proof | none |
| EQ / HPF | PROTOTYPE | EQ modules, `backend/phase_aware_eq_guard.py` | no | no | protected-band, feedback, readback, rollback proof | none |
| Compressor / dynamics | PROTOTYPE | compressor modules | no | no | headroom, GR and readback proof | none |
| Gate / expander | PROTOTYPE | gate modules | no | no | chatter/bleed replay plus hardware proof | none |
| De-esser | SPECIFIED | de-esser briefs/proposals | no | no | unit/replay first | none |
| FX sends / returns | PROTOTYPE | `backend/fx_send_proposal.py` | no | no | readback and rollback proof | none |
| Routing / patch | SPECIFIED | `backend/routing_management_proposal.py`, routing readback contracts | no | no | fresh read-only/no-authority route context, rollback and physical readback first | none |
| Bus/group/main/DCA faders | SPECIFIED | bus/DCA physical readback contracts | no | no except later approved supervised proof | physical readback | none |
| Matrices | SPECIFIED | matrix physical readback contracts | no | no | physical readback | none |
| Scenes / snapshots / presets | SPECIFIED | snapshot handlers and contracts | no | no | destructive-action ADR and proof | none |
| Rollback | SPECIFIED | WING proof runbooks | no live action unattended | required before closure | supervised rollback proof | none |
| Emergency stop | SPECIFIED | WING MVP docs | no live action unattended | required before closure | emergency-stop proof | none |
| Rate limits / bounds / allowlists | PROTOTYPE | `backend/autofoh_safety.py`, `backend/wing_client.py`, `backend/fault_injection_recovery_contract.py` | docs/tests only | required | fault-injection validation | none |
| Stale state / disconnect fail-closed | PROTOTYPE | preflight and runtime safety modules | no hardware unattended | required | timeout/fail-closed proof | none |
| Audit logging | PROTOTYPE | WING runbooks, reports, ledger patterns | yes for local state | required | artifact review | none |
