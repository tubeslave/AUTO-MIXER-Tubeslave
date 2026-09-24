# Independent software review

Reviewer: Codex subagent McClintock, 01a0d291-a0f0-7280-b469-0b41338709ef.
Read-only review and isolated memory-socket probes; no hardware/network access.
Kimi attempt failed HTTP 403 (access_terminated_error); no Kimi approval.

## Initial findings and resolution

- Real WingClient lacked an installed boundary: exact typed operation grants
  now cover only a journalled send and its queue drain, then revoke.
- A delivered-but-failed send lacked rollback ownership: preimages now enter
  the recovery stack before the first possible packet.
- Persistent WAL failure prevented restoration: the durable original prepared
  record permits only its exact preimage; pending audit remains unresolved.
- Reentrant stop could report applied after restoration: stop latches and defers
  to the active operation; its outer frame cannot publish success afterward.
- Expired batch entry skipped recovery: prior owned changes are restored.
- Terminal audit failure skipped transport teardown: cleanup runs independently.
- Repeated-address audit failure could leave an intermediate value: physical
  restoration and pending audit are tracked separately. Retry never replays an
  intermediate preimage over a restored original baseline.

## Final review verdict

No blocking findings in the final repeated-address change. Independent probes
with throttle enabled and disabled verified original-baseline restoration,
zero packets on audit-only retry, preserved verification timestamps, denied
write authority for restored rows and successful release after recovery.

Accept the software fix, subject to the pending clean full-suite pass. This is
not hardware authorization, a CP05 approval or acoustic validation.

## Codex critique of review

Keep the review's distinction between physical restoration and audit completion.
The repeated-address disk failure was a real missed case, not a reason to loosen
transport guards. The patch stays inside lab lifecycle/transport integration;
DSP algorithms, production allowlists and CP05 files are unchanged.
