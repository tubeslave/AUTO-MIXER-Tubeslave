# Independent proposal — M120 HW01 teardown invalidation

## 1. Short thesis

Keep the backend broadcast unchanged and add an explicit teardown-only opt-in
to the runner's terminal receiver so only the expected disconnect
`readiness_invalidated` notification is tolerated.

## 2. What I understood

The accepted hardware artifact proves HW01 route/capture success and zero
console writes. Backend teardown also physically stopped capture, revoked route
authority, disconnected WING/OSC and terminated cleanly. The false failure is
inside the runner: after it receives `connection_status/connected=false`, the
generic duplicate-grace read sees the backend's next truthful
`auto_soundcheck_status/readiness_invalidated` broadcast and raises
`RuntimeError` before `mixer_disconnected` is assigned.

The regression entered when active-session readiness latching was added without
adapting the older disconnect teardown path. Existing fakes return only the
disconnect response and therefore omit the real backend's second broadcast.

## 3. Proposed solution

Add a narrowly named opt-in such as
`allow_expected_disconnect_invalidation=False` to the generic terminal and
duplicate-grace receivers. Only when true may they consume the exact
`auto_soundcheck_status` plus `status=readiness_invalidated` payload during
disconnect teardown. The disconnect acknowledgement is still mandatory.

Call the opt-in only from `ProofOwnedBackendSession._close_impl()` while
waiting for `connection_status`. Do not modify the backend broadcast or normal
active-session behavior.

## 4. Likely files to touch

- `tools/wing_attended_soundcheck_runner.py`
- `tests/test_wing_attended_soundcheck_runner.py`
- Council/ADR/truth documentation required by the repository contract

## 5. Alternatives considered

- Remove or reorder the backend readiness-invalidated broadcast: rejected;
  the notification truthfully invalidates readiness for all clients.
- Swallow teardown `RuntimeError`: rejected; it could hide a missing disconnect.
- Treat the accepted artifact as enough despite teardown failure: rejected;
  verified teardown is a required HW01 invariant.

## 6. Risks

- A broad opt-in could leak into active operations. Mitigate with an explicit
  default-false name and a single call site in `_close_impl()`.
- Readiness invalidation alone must never count as disconnect. The receiver
  must still time out/fail if `connection_status/connected=false` is absent.
- Duplicate disconnect terminals and unrelated control failures must remain
  rejected.

## 7. Test plan

1. Disconnect acknowledgement followed by exact readiness invalidation yields
   verified teardown.
2. Readiness invalidation without disconnect acknowledgement remains blocked.
3. Existing final-audit interleaved-readiness test stays unchanged and fatal.
4. Run the full M110 runner and route-proof suites plus hardware/static rails.

## 8. Where the other agent may disagree

Codex may prefer a separate teardown receiver to keep the generic receiver
structurally untouched. The opt-in is a smaller patch, while the separate
method provides a narrower API boundary. Both are acceptable only if the
normal receiver remains strict and the disconnect acknowledgement is still
required.
