# Console Write Safety Policy

## Purpose

This document defines safety rules for any real console write in this repository, including direct console writes, OSC writes, and any future path that can mutate a real mixer state.

- It applies to current code and to future write paths.
- It is based on `docs/runtime_ownership_matrix.md` and `docs/current_architecture_map.md`.
- The AutoSoundcheck vs feedback split is further defined in `Docs/TUB-194_AUTOSOUNDCHECK_FEEDBACK_POLICY_SPLIT.md`.
- It does not approve new write paths by itself.

## Scope

This policy applies to:

- manual UI writes;
- voice-command writes;
- automation writes;
- feedback or emergency writes;
- snapshot, reset, and restore utility writes;
- offline pipelines if a real sender is ever attached to them.

## Core Safety Rules

### Dry-Run By Default

- Any write path must default to dry-run unless explicit approval says otherwise.
- Dry-run means the path may analyze, queue, or report intended changes, but must not mutate the real console.

### Explicit Confirmation For Real Writes

- Real writes require an explicit confirmation step before mutation.
- Implicit UI actions, background automation state, or prior operator intent are not enough.

### Readback Confirmation Where Supported

- Where the client and console support readback, the system must verify the resulting console state after a write attempt.
- A write cannot be treated as applied only because a send call returned success.

### `send_status=sent` Does Not Mean Applied

- `sent` means the command was emitted toward the transport.
- `applied` or `accepted` means the system received confirmed readback showing the expected result.
- For real writes, confirmed readback is the only documented basis for `applied` or `accepted`.

### Audit Event For Every Attempt

- Every write attempt must produce an audit event.
- The audit record should preserve at least:
  - requested operation;
  - target channel or parameter;
  - dry-run vs real-write intent;
  - confirmation state;
  - send result;
  - readback result when available;
  - final operator-visible status;
  - `audit_id`.

### Panic Stop Semantics

- Panic stop is not system-wide.
- Current documented panic-stop coverage is limited to manual `LiveApplyService` paths only.
- No write path outside that scope may claim panic-stop coverage without separate audit and approval.

### Max Step And Rate Limits

- Real write paths must use bounded per-step changes and bounded send rate.
- Safety limits must be explicit in the design for any new mutation path.
- Rate limiting and step limiting are required because transport success alone does not guarantee safe console state.

### Fail-Closed Behavior

- On unsupported operation, disconnected transport, ambiguous confirmation state, or policy mismatch, the write path must fail closed.
- Fail closed means no real mutation is treated as approved or successful.
- Missing safety metadata must default to blocked behavior, not permissive behavior.

## Current Approved Gate

The current approved real-write scope is narrow.

- `LiveApplyService` is the current approved gate only for manual `set_fader` and manual `set_gain`.
- Panic stop is documented only for manual `LiveApplyService` paths.
- Panic stop is not system-wide.
- Reuse of `LiveApplyService` elsewhere does not automatically expand approved scope.

## Blocked / Not Approved Paths

The following paths are not approved for live console mutation under this policy:

- manual UI `set_eq`;
- manual UI `set_compressor`;
- `AutoFaderController` in `backend/auto_fader.py`;
- arrangement automation in `backend/arrangement_automation/controller.py`;
- live input trim outside approved canonical path rules;
- `backend/auto_soundcheck_engine.py`;
- feedback reaction path;
- any voice path outside `LiveApplyService`;
- snapshot/reset/restore scripts and related direct utility writes;
- offline sender attachment without a unified console-write gate and separate approval.

Additional interpretation rules:

- `LiveInputTrimController` may be canonical architecture-wise, but it is not broadened into general approved write scope by this document.
- A path can be active in code and still remain not approved for live use.

## Emergency Exception Policy

The feedback detector path has a special status.

- `backend/feedback_detector.py` plus its current reaction path is a safety-critical emergency exception candidate only.
- It is not an approved exception.
- It must remain policy-separated from `backend/auto_soundcheck_engine.py` broad live-apply expansion work.
- It requires a separate future policy review before any live-scope expansion or exception approval.
- Until that review happens, it must be documented conservatively as not approved.

## Offline Sender Attachment Rule

- Offline pipelines are not live-console paths by default.
- `automixer/production_mix_v1` and `ai_mixing_pipeline` remain offline or research surfaces unless separately approved otherwise.
- Attaching a real sender to an offline pipeline is not approved without:
  - a unified console-write gate;
  - explicit policy approval;
  - dedicated safety review;
  - clear operator-visible status semantics.

## Migration Rules

For any new or upgraded write path:

- design first;
- focused tests first for safety and mutation semantics;
- minimal implementation only;
- explicit review before widening scope;
- explicit approval before live use;
- no broad rewrites justified only by cleanup;
- no live-default behavior.

Additional migration constraints:

- New write paths must not bypass current documented safety expectations.
- A path should not claim approval merely because it shares a client, queue, or transport abstraction with an approved path.
- Expanding approved scope requires documentation update before product/UI claims change.

## Operator Visibility Rules

UI and API surfaces should expose write state clearly. At minimum, operator-visible status semantics should include:

- `dry-run`
- `blocked`
- `not sent`
- `sent but not confirmed`
- `confirmed`
- `timeout`
- `mismatch`
- `disconnected`
- `unsupported`
- `panic stop active`
- `audit_id`

Interpretation rules:

- `sent but not confirmed` must not be rendered as applied.
- `confirmed` should be reserved for confirmed readback where supported.
- `blocked` and `unsupported` must be visible, not silently downgraded into generic failure.

## Open Decisions

- Whether `AutoSoundcheck` live-apply and feedback emergency reaction remain clearly separated in future docs, gating, and product language.
- Whether the feedback reaction path should ever become an approved emergency exception.
- Whether operator-only snapshot/reset/restore tools should remain available as legacy tools or later move into archival/removal planning.
- Whether any offline pipeline may ever attach to a real sender, and under which unified gate model.
- Whether canonical live input trim should later become an explicitly approved write scope or remain architecturally canonical but policy-limited.
- Whether system-wide panic stop should ever be designed, implemented, and approved as a separate capability.
