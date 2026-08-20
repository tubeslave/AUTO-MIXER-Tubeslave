# Codex critique of Kimi proposal

## 1. Strongest part of the other proposal

The frozen 143-row disposition ledger is the strongest improvement. It turns
"all findings" into a checkable completeness invariant and prevents the
existing progress report from silently standing in for occurrences it never
addresses. File clustering inside severity tiers is also well suited to this
heavily dirty checkout.

## 2. Possible mistakes or blind spots

- Requiring zero `unknown` before any production edit can deadlock the work:
  some reachability questions are answered only by writing and running a
  focused regression. The correct Phase 0 exit is 143 inventoried rows with
  every unknown assigned a concrete proof task, not premature disposition.
- A single shared live-write guard cannot replace sink-local transport-result,
  readback, compare-and-swap, and rollback assertions. It is suitable only to
  quarantine unsupported legacy entrypoints.
- The reported full-suite pipe hang is not yet reproduced in this current
  task. Changing that test before reproduction would exceed the evidence.
- Running no-WING quick after every small slice may be too expensive and create
  many artifact directories on a nearly full disk. Use it at trust-boundary
  cohort boundaries, with targeted invariant tests for individual slices.

## 3. Where the solution is too complex, fragile, or costly

A general runner that executes arbitrary commands stored in a JSONL ledger
would itself create a command-injection and maintenance surface. The checker
should validate schema/completeness and map stable test IDs from code-owned
allowlists; it must not execute ledger-provided shell text. Storing dirty-diff
hashes for every slice is useful, but full diff snapshots in the ledger would
be noisy and disk-costly.

## 4. What should be kept from the other proposal

- Exact 143-row ledger and severity histogram.
- `locally_verified_unsealed` as the strongest local success state.
- Severity tiers with file clustering.
- A tiny shared quarantine guard only for legacy paths with no canonical
  supervised transaction.
- Reproduction of the alleged test hang before any harness change.
- Bounded temp roots and explicit separation from hardware proof.

## 5. My revised recommendation

Freeze the inventory and proof obligations first, allowing `unknown` only when
it has an assigned executable reachability task. Remediate high findings in
file-clustered cohorts. Use a shared quarantine guard for unsupported legacy
mutators, while retaining sink-local validation for supported transports,
files, resource limits, numeric checks, and evidence chains. Make the
disposition checker data-only plus a code-owned static mapping of test node IDs,
never a shell-command interpreter. Run targeted tests per slice, no-WING quick
per root-cause cohort, and the standard full suite at severity milestones and
finalization.
