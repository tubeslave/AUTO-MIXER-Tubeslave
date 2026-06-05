# settings-read-surface-proof-quality-20260605

## Context

The Product Layer now exposes read-surface proof-quality metadata under
`supervised_proof_evidence.summary`. The frontend adapter maps it into module
status profiles, but Settings still only summarizes proof source freshness and
autonomy overlay status.

## Goal

Show read-surface proof quality in the existing Settings Module Status
inspector so an operator can see physical readback gaps and validation gates
without opening no-WING artifacts manually.

## Non-goals

- Do not add apply controls, queue mutations, or live console commands.
- Do not promote read-surface fixture evidence into hardware proof.
- Do not hide physical readback gaps behind a generic "ready" label.

## Safety constraints

- The `supervised_proof_evidence` row remains `canWriteLive=false`.
- Physical gap counts must stay visible when proof-quality metadata exists.
- Contract/gap enforcement is display evidence only, not promotion authority.

## Expected evidence

- Settings inspector shows read-surface proof metadata availability.
- Settings inspector shows physical gap count and fixture test count.
- Settings inspector shows whether the contract/gap gate was enforced.
- E2E backend DTO path proves browser-visible rows and keeps no-live-write
  command assertions.
