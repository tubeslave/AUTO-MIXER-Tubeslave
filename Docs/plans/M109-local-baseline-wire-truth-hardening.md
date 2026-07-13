<!-- /autoplan restore point: /Users/dmitrijvolkov/.gstack/projects/tubeslave-AUTO-MIXER-Tubeslave/codex-automixer-autopilot-control-plane-autoplan-restore-20260713-093613.md -->

# M109 — local baseline, wire-contract, and truth hardening

## Context

The July 13 real-WING run proved read-only connectivity and stable physical
readback, but performed no DSP reset and applied no mix corrections. The
normal-family live allowlist remains empty and the soundcheck status remains
`soundcheck_incomplete`.

Independent review found that the current standard full suite is red because
four positive no-WING contract tests depend on mutable repository evidence. It
also confirmed that the full decision queue can exceed the 256 KiB WebSocket
control-plane limit because `raw`/`result` evidence is serialized without a
byte budget.

The current HEAD scores **85/100** under release validation because its standard
suite is red. The last strict live assessment was **88/100**, but it is not a
current-HEAD verification claim while the baseline and truth surfaces are
stale. M109 restores local evidence quality; it does not claim to move strict
live readiness from 88 to 95.

## Safety boundary

- No console contact or write in unattended checkpoints.
- No reset implementation or live-authority widening in this plan's first two
  checkpoints.
- Failed hardware artifacts remain immutable evidence.
- Apply always uses the complete server-side proposal, never a compact wire
  projection.
- STOP and single-writer lease rules remain mandatory.

## Checkpoint M109-CP00 — publish the current verification downgrade

Goal: before implementation proceeds, make the canonical current sections say
that HEAD is not verified while the standard suite is red, without rewriting
historical evidence.

Minimum truth update:

- current HEAD and failing command/result;
- strict live evidence remains read-only and `soundcheck_incomplete`;
- local 95 and strict-live 88 are historical, not current-HEAD closure claims;
- normal-family allowlist remains empty;
- no reset or soundcheck correction was applied.

The final evidence-bound synchronization still occurs in CP03 after CP01 and
CP02 are committed and verified.

## Checkpoint M109-CP01 — restore hermetic full-suite baseline

Goal: isolate three static runner-contract tests and one synthetic aggregate
release-run test from real WING artifact history without changing production
safety behavior.

Files:

- `tests/test_no_wing_release_check.py`
- `.ai/briefs/M109-no-wing-test-isolation.md`
- `.ai/proposals/M109-no-wing-test-isolation-*.md`
- `.ai/reviews/M109-no-wing-test-isolation-*.md`
- `Docs/adr/M109-no-wing-test-isolation.md`

Required evidence:

1. Four affected positive tests pass.
2. Their pass assertions expose `result.detail["violations"]` on failure.
3. Deterministic negative tests for forbidden/non-physical evidence and
   authority claims remain fail-closed.
4. Complete no-WING release-check test module passes.
5. Standard full suite passes.
6. Production validator and canonical artifacts are unchanged.

## Checkpoint M109-CP02 — byte-bounded decision-queue wire contract

Goal: guarantee a compact decision-list response of at most 128 KiB while
preserving exact blockers and server-side full-detail/apply semantics.

Design requirements:

- freeze canonical v2 review content and SHA-256 at proposal creation;
- separate internal/full and compact wire/list serializers and never call full
  `to_dict()` while selecting list rows;
- no `raw`, `result`, plans, blobs or unbounded arrays in list rows;
- pin explicit global hazards, unresolved `can_apply=false` rows, and
  high/critical rows before ordinary newest-first rows;
- use connection-bound frozen snapshots with 30-second TTL and hard count/byte
  retention limits;
- return exact total/included/remaining counts and a degraded counts-only
  envelope when pinned metadata cannot fit;
- both operator proposals and stable analysis-only MixingAgent rows obey row and
  whole-payload byte limits;
- authenticated detail returns complete canonical review content in digest-bound
  base64 chunks of at most 64 KiB raw each;
- canonical apply and every internal executor require the matching digest before
  any mutation, idempotency reservation, backend contact or console send;
- pre-v2 proposals and the bounded v1 read shim have no live apply authority;
- close code 1009 is terminal/manual-retry and cannot activate mock data;
- hard 131,072-byte assertions use the actual WebSocket serialization path for
  list, detail and compact mutation responses.

Required evidence:

1. A 1 MiB raw proposal produces a response no larger than 128 KiB.
2. Long Cyrillic/emoji rows remain within budget with exact counts.
3. Legacy soundcheck blockers and `can_apply=false` survive compaction.
4. Existing read consumers receive a bounded compatibility shim; apply paths
   migrate atomically and have no digestless compatibility bypass.
5. An old critical blocker cannot be displaced by newer low-risk rows.
6. Missing detail and digest mismatch return explicit fail-closed errors.

## Checkpoint M109-CP03 — operator truth and current-state sync

Goal: remove stale M005 wording, give all three readiness states explicit
operator meaning, and bind truth surfaces to the verified HEAD and evidence.

The source of truth must be one machine-readable readiness manifest containing
HEAD, console identity/firmware, route proof, artifact digests, freshness/TTL,
family coverage, readback/rollback/audio proof, lease state, and exact blockers.
Documentation either derives from it or is protected by equality tests.

Required wording:

- `soundcheck_incomplete`: important corrections remain unapplied;
- `safe_core_ready`: safe core is ready, but dynamics/spatial/FX are incomplete;
- `show_mix_ready`: console mix is ready, while room/PA validation remains
  required.

Sync targets:

- `Docs/CURRENT_STATE.md`
- `Docs/ACTIVE_MILESTONE.md`
- `ops/autopilot/STATE.json`
- `ops/autopilot/BLOCKERS.md`
- `Docs/ROADMAP.md` when its current section changes

## Deferred high-risk work

- Typed proposal-only `maintenance_dsp_reset` with a fixed preserve set.
- Single atomic reset transaction, address/time budgets, complete preimage,
  WAL, CAS rollback, and restart recovery.
- Immutable Main 47/48 route/headroom proof.
- Isolated real-WING apply/readback/rollback/audio proofs for M006 families.
- Staged real soundcheck to `safe_core_ready` and then `show_mix_ready`.
- Developer ID signing, notarization, and Gatekeeper release evidence.

These items require separate ADRs and, where hardware is involved, an attended
operator session. Reset must not be impersonated as a soundcheck transaction.

## Verification contract

- Targeted backend/frontend tests for each checkpoint.
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q` at every
  checkpoint boundary.
- Frontend test/build when CP02 or CP03 touches frontend code.
- `git diff --check`, final diff audit, independent regression/safety review.
- No readiness score increase without committed evidence and clean-checkout
  verification.

## Current status

- M109-CP01: committed as `b3f7d05ac`; `6368 passed, 1 skipped`; independent
  diff review found no P0/P1 and its documentation findings are resolved.
- M109-CP00: superseded by the restored validated working-tree status.
- M109-CP02: exact v2 wire messages, digest projection, cursor, retention,
  overflow, chunked detail, immutable execution snapshot and no-bypass apply
  semantics are accepted in
  `Docs/adr/M109-CP02-decision-queue-wire-contract.md`; two independent
  architecture reviews and the engineering gate returned GO. Server
  implementation is in progress.
- M109-CP03: planned.
- Hardware-dependent work: deferred to an attended session.

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 1 | COMPLETE | Renamed local milestone; added CP00, priority queue and digest contract |
| Codex Review | `/codex review` | Independent 2nd opinion | 1 | DEGRADED | CLI/model mismatch; Council fallback independently agreed on CP01 |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 2 | COMPLETE | CP02 GO after version/digest, TOCTOU, retention, hazard, cursor and post-write receipt contracts were made exact |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | SKIPPED | CP01 has no UI change |
| DX Review | `/plan-devex-review` | Developer experience gaps | 1 | COMPLETE | CP01 clear; CP02 held for an exact versioned API/error contract |
| Final diff review | independent subagent | Regression and safety | 1 | COMPLETE | No P0/P1; post-commit truth binding required |

**UNRESOLVED:** 0 user decisions; autonomous scope is bounded by existing safety rules.

**VERDICT:** CP01 CLEARED; CP02 ADR and engineering gate CLEARED for bounded
implementation. Hardware authority remains unchanged.

## CEO Review — scope and strategy

### Premise challenge

| Premise | Assessment | Decision |
|---|---|---|
| Green local validation is prerequisite to further readiness work | Valid: a red mandatory suite invalidates current-HEAD closure claims | Keep CP01 first after immediate truth downgrade |
| M109 can move strict live 88 to 95 without hardware | False | Rename M109; create a separate attended golden-path milestone |
| Failed real-WING attempts may be deleted to make release checks green | False: they are provenance | Preserve artifacts; isolate static tests only |
| Row count bounds WebSocket response size | False: one row exceeds 1 MiB | Add a hard serialized-byte budget and list/detail split |
| Newest-first trimming is sufficient | False: it can hide old critical blockers | Pin hazards/blockers, then paginate ordinary rows |

The user's original direction already confirms the governing premise: continue
working toward a truthful 95/100 without bypassing live safety. M109 stays
inside that approval and does not widen hardware authority.

### What already exists

| Sub-problem | Existing leverage |
|---|---|
| Test evidence isolation | Empty-evidence injection precedent in `tests/test_no_wing_release_check.py` |
| Fail-closed artifact intake | `backend/wing_readonly_preflight_evidence.py` and `tools/no_wing_release_check.py` |
| Full proposal storage/apply | `backend/operator_proposal_queue.py` |
| Queue product snapshot | `backend/operator_product_state.py` |
| Actual wire serialization | `backend/ws_transport.py` |
| Conservative mix readiness | `backend/soundcheck_mix_readiness.py` |
| Truth surfaces | `ops/autopilot/STATE.json`, `BLOCKERS.md`, `Docs/CURRENT_STATE.md`, `ACTIVE_MILESTONE.md` |

### Dream-state delta

```text
CURRENT
  red suite + stale truth + unbounded queue + zero normal live authority
      |
      v
M109
  hermetic green baseline + byte-bounded review UI + one readiness manifest
      |
      v
ATTENDED GOLDEN PATH
  useful proposal -> explicit approval -> one bounded correction
  -> physical readback -> rollback -> Main/audio proof -> staged mix readiness
```

M109 improves trust and operator visibility. It deliberately stops short of the
12-month ideal: a complete show mix formed through verified, staged, musically
useful transactions.

### Alternatives

| Option | AI-assisted effort | Completeness | Risk | Decision |
|---|---:|---:|---:|---|
| Delete/ignore invalid artifacts | minutes | 2/10 | critical safety/provenance loss | Reject |
| Patch three tests, then row-only queue limit | under 1 hour | 5/10 | code 1009 remains possible | Reject |
| Hermetic tests + byte-budgeted list/detail + manifest truth | several hours | 9/10 | bounded migration/compatibility work | Select |
| Implement reset and full live apply now | multi-session | 10/10 aspirational | unacceptable without attended proof | Separate milestone |

### Temporal interrogation

- Hour 1: publish current downgrade, isolate three tests, re-run no-WING tests.
- Hours 2–4: full-suite proof and atomic CP01 commit.
- Hours 4–6: design and implement compact list/detail contract with adversarial
  payload tests.
- After verification: synchronize the manifest-derived truth surfaces.
- Attended session only: reset proposal, route proof, physical apply/rollback,
  and audible validation.

### Error & Rescue Registry

| Error | Detection | Rescue | Safety result |
|---|---|---|---|
| Canonical invalid artifacts contaminate static tests | unexpected aggregate violation | inject explicit empty evidence only in static tests | production fail-closed unchanged |
| Queue exceeds 128 KiB | exact serialized byte count | pin critical rows, trim/paginate, counts-only fallback | operator transport remains available |
| Detail missing or digest changed | detail lookup/digest check | deny apply and require re-review | stale approval cannot execute |
| Truth manifest stale | HEAD/TTL/digest mismatch | downgrade to incomplete/unverified | no readiness promotion |
| Hardware proof required during unattended work | attended-session gate | stop at proposal/evidence preparation | no WING contact/write |

### Failure Modes Registry

| Failure mode | Severity | Plan control | Open? |
|---|---:|---|---|
| Green-but-blind hermetic suite | high | retain explicit invalid-evidence tests and verify canonical aggregate separately | controlled in CP01 |
| Old critical blocker omitted from wire list | critical | pinned safety ordering + pagination | CP02 |
| Compact row used as apply authority | critical | server-side full object + reviewed digest required | CP02 |
| Five truth files drift again | high | canonical manifest + equality tests | CP03 |
| M109 misrepresented as live 95 | critical | renamed milestone and separate attended milestone | closed in plan |

### NOT in scope

- Real reset, apply, rollback, or WING writes: requires attended operator proof.
- Signing/notarization: independent distribution track, not a substitute for
  strict-live proof.
- Raw OSC escape hatches or wider normal live registry: prohibited.
- Broad cleanup of every environment-dependent test: defer until demonstrated.

### Completion summary

- Mode: HOLD SCOPE for CP01; selective completeness for CP02/CP03.
- Strategic correction: M109 no longer claims strict-live 88-to-95 progress.
- Critical gaps added: safety-prioritized queue ordering, stable detail digest,
  canonical readiness manifest, and immediate truth downgrade.
- Premise gate: passed from the user's explicit continue-to-95 instruction and
  the existing no-bypass safety contract.
- CEO outside voice: subagent found 3 critical and 4 high issues; Codex CLI was
  unavailable because local CLI 0.138.0 cannot run configured model
  `gpt-5.6-sol`.

## Decision Audit Trail

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|---|---|---|---|---|---|
| 1 | CEO | Rename M109 to local hardening; strict 88-to-95 becomes attended milestone | Auto-decided | Truth over score | No hardware/apply occurs in M109 | Keep misleading 88-to-95 title |
| 2 | CEO | Add CP00 current verification downgrade | Auto-decided | Fight uncertainty | Canonical truth must not stay knowingly stale | Wait until final sync |
| 3 | CEO | Pin critical blockers before ordinary queue rows | Auto-decided | Safety first | New traffic must not hide hazards | Pure newest-first trimming |
| 4 | CEO | Choose stable detail-by-ID plus content digest | Auto-decided | Explicit over clever | One defined contract is testable and fail-closed | Ambiguous path-or-detail design |
| 5 | CEO | Create one machine-readable readiness manifest | Auto-decided | Single source of truth | Prevent repeated cross-file drift | Manual five-file synchronization only |

## Engineering Review — CP01 architecture and tests

### Scope challenge

CP01 is deliberately test-only. The result is artifact-history-hermetic for
four positive contract checks, not a claim that the entire no-WING validator
is environment-independent. Production discovery remains intentionally tied to
canonical evidence and must keep rejecting any invalid candidate.

### Architecture

```text
positive contract test
  -> prebuild empty evidence (explicit roots/payloads)
  -> opt-in monkeypatch of release_check module global
  -> production validator logic
  -> contract-field assertions

deterministic negative test
  -> injected malformed/authority-bearing evidence
  -> unchanged production validator logic
  -> exact fail-closed violation

real canonical evidence
  -> unchanged discovery and aggregate blocker
  -> never rewritten by CP01
```

### Code quality and leverage

- Reuse the empty-evidence setup already present in
  `tests/test_no_wing_release_check.py`; extract one private helper.
- Precompute the evidence before patching to avoid recursion.
- Patch only `release_check.build_wing_readonly_preflight_evidence` and invoke
  the helper explicitly in four positive tests.
- Do not add an autouse fixture, production seam, dependency or artifact-count
  assertion tied to today's repository population.

### Verification layers

```text
4 repaired positive tests
        |
        v
explicit malformed/authority negative tests
        |
        v
complete no-WING release-check module
        |
        v
standard repository suite
        |
        v
path-scoped proof: production validator + canonical artifacts unchanged
```

The executable checklist is preserved outside the repository at
`/Users/dmitrijvolkov/.gstack/projects/tubeslave-AUTO-MIXER-Tubeslave/codex-automixer-autopilot-control-plane-test-plan-20260713T065148Z.md`.

### Performance, security, and failure modes

- Performance impact: none in production; four tests build one small in-memory
  empty evidence object each.
- Security impact: no authority or discovery change; monkeypatch lifetime is
  test-scoped.
- Failure mode: helper recursion. Control: precompute before monkeypatch.
- Failure mode: green-but-blind suite. Control: explicit deterministic
  invalid-evidence and authority-claim tests remain unpatched.
- Failure mode: accidental evidence mutation. Control: path-scoped diff and
  status checks, not only `git diff --check`.
- Failure mode: broad environment masking. Control: no autouse fixture.

### NOT in scope

- Production validator refactoring.
- Deleting, moving or normalizing real WING artifacts.
- Freezing the current canonical artifact counts as a software contract.
- Queue wire-contract implementation, reset, or console contact.

### Completion and consensus

| Reviewer | Verdict | Required condition |
|---|---|---|
| Codex proposal | GO | explicit empty evidence in positive synthetic tests |
| Council fallback | GO | precompute helper; target a negative test |
| Engineering review | GO | CP00 first; add deterministic negative and path-scope proof |

CP00 is now published. There are no unresolved engineering decisions for CP01.

## Decision Audit Trail — engineering additions

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|---|---|---|---|---|---|
| 6 | Eng | Call CP01 artifact-history-hermetic, not globally hermetic | Auto-decided | Precise claims | Other validator inputs remain repository-backed | Overclaim full isolation |
| 7 | Eng | Reuse one explicit helper; no autouse fixture | Auto-decided | Smallest safe diff | Limits masking to four positive tests | Global fixture |
| 8 | Eng | Require path-scoped immutable-production proof | Auto-decided | Evidence preservation | Formatting checks cannot prove unchanged validator/artifacts | `git diff --check` alone |

## Developer Experience Review

### Persona and empathy narrative

The primary user is the maintainer of the live-safety backend and its operator
UI integrator. They understand Python and React, but should not need to reverse
engineer the repository's physical WING artifact history to run a static
contract test.

The first three CP01 tests failed quickly but surfaced only `failed != passed`;
the useful reason, `wing_readonly_preflight_invalid_artifacts_present`, is
buried in `result.detail["violations"]`. CP02 has a different problem: the
intent is sound, but a contributor would still have to invent message names,
cursor semantics, digest scope, retention and the error vocabulary.

### Journey and target time

| Step | Current experience | Target |
|---|---|---|
| Find CP01 | clear plan split | keep |
| Reproduce | exact command lives in external test plan | first useful failure under 2 minutes |
| Repair | safe precedent exists but recursion trap is implicit | one precomputed opt-in helper |
| Verify safety | negative checks were abstract | exact negative tests plus immutable-path proof |
| Begin CP02 | conceptual contract | versioned request/response examples before code |

Measured CP01 time to understand and repair is 7–12 minutes; the clarified
path targets 2–5 minutes. CP02 time-to-first-contract-test must be under
10 minutes after its ADR.

### DX score

| Dimension | Score / 10 |
|---|---:|
| Getting started | 6 |
| API design | 5 |
| Error messages | 4 |
| Documentation | 6 |
| Upgrade path | 4 |
| Dev environment and testability | 8 |
| Internal findability | 5 |
| Measurement and feedback | 7 |

Overall: **5.6/10**. CP01 is low-risk and clear after the recorded corrections;
CP02 is not implementation-ready.

### CP01 implementation checklist

- Prebuild explicit empty evidence, then monkeypatch the module global.
- Use the helper only in the four positive synthetic/contract tests.
- Include the violations list in each pass-assert failure message.
- Run the repaired tests and explicit negative safety tests.
- Run the full module, standard suite and path-scoped immutable evidence proof.

### CP02 contract required before implementation

- Message names: `get_decision_queue` v2 and
  `get_operator_proposal_detail`.
- `wire_contract_version`, opaque cursor, snapshot revision and exact
  total/included/omitted/pinned counts.
- Digest over immutable review-content only, excluding dynamic status/expiry.
- Apply requires proposal ID, version and reviewed content digest.
- Explicit retention/eviction behavior for pending and history.
- Budget covers the complete serialized payload, including summaries and
  agent status.
- Defined counts-only emergency response when pinned hazards alone exceed the
  budget.
- Stable errors: `reviewed_digest_required`, `proposal_digest_mismatch`,
  `proposal_version_mismatch`, `proposal_detail_not_found`, and
  `decision_queue_cursor_stale`.
- WebSocket close 1009 is non-retryable and must not trigger mock-state fallback.

### DX completion summary

- CP01: CLEAR, low risk, no production API or live-safety change.
- CP02: HOLD until the versioned wire-contract ADR resolves every item above.
- Competitive benchmark: not applicable to this repository-internal control
  plane; the relevant benchmark is deterministic first-party contract testing.

## Decision Audit Trail — DX additions

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|---|---|---|---|---|---|
| 9 | DX | Include violations in positive-test assertion messages | Auto-decided | Useful errors | Makes the real cause visible immediately | Bare status assertion |
| 10 | DX | Hold CP02 until a versioned wire contract exists | Auto-decided | API before implementation | Prevents incompatible list/detail/apply semantics | Invent fields during coding |
| 11 | DX | Treat close 1009 as non-retryable without mock fallback | Auto-decided | Honest operator state | Prevents reconnect loops and fabricated live state | Silent reconnect/mock loop |
| 12 | Verification | Expand CP01 from three to four explicit positive tests | Evidence-driven | Follow the full module | The fourth synthetic aggregate test had the identical hidden canonical-root dependency | Autouse fixture or production change |
| 13 | Verification | Update canonical M006 truth rail to invalid-artifact blocker | Evidence-driven | Fail closed on current evidence | Seven preserved invalid attempts correctly remove candidate availability | Keep stale candidate-available assertion |
| 14 | Verification | Keep offline candidate review but expose blocked aggregate status | Evidence-driven | Separate review work from readiness | Valid candidates may be reviewed, while any invalid artifact keeps readiness blocked | Claim global candidate-ready status |
| 15 | Final review | Exclude three byte-identical generated mix reports | Auto-decided | Preserve provenance | They are test side effects outside M109 source scope | Commit or delete them casually |
| 16 | Final review | Use a second post-commit truth-binding checkpoint | Auto-decided | Machine-checkable truth | A commit cannot contain its own hash in pre-commit evidence | Leave pending-commit truth stale |
