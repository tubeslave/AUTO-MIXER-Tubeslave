# Project Memory

## Stable Conventions

- Read `AGENTS.md` and `CLAUDE.md` before planning or editing.
- Read `.ai/agent-core/` and `.ai/project-layer/` before starting new Automixer
  tasks; these files are the current Markdown memory layer.
- Use `Docs/adr/` for accepted architecture decisions.
- Default test gate: `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- Keep diffs small and avoid new production dependencies without written justification.

## Safety-Critical Rules

- Prefer safer output changes over louder ones.
- Preserve true-peak headroom before increasing gain.
- Respect feedback protection and mixer-safety constraints from `CLAUDE.md`.
- Live OSC/WING writes require explicit approval for the exact action. Default
  WING, OSC, Dante, gain, trim, fader, routing, mute, snapshot, EQ, dynamics,
  and FX work to audit-only or dry-run.
- Never change secrets or `.env` files as part of council work.

## Architecture Reminders

- `server.py` is a coordinator; keep logic in dedicated handlers or modules.
- `AudioCapture` is the shared audio service; do not duplicate capture pipelines casually.
- Repeated DSP or protocol rules belong in `CLAUDE.md` or `Docs/CONVENTIONS.md`, not only in task-specific notes.
- Live WING shared-mix analysis lives in `backend/live_shared_mix.py`; keep it as a pure planner and send actions through `AutoFOHSafetyController`.

## Council Lessons

- Separate planning, implementation, and review into different phases.
- The writing agent and reviewing agent should use different worktrees.
- Durable lessons should be generic enough to survive beyond a single task.
- Routing and channel-name writes on a live WING require an explicit expected patch/name map; audit/readback can be automatic, but patch writes should stay disabled by default.
- Automation foundation evidence uses `AutomationDecisionTrace` as the canonical row shape for proposal, replay, blocked, apply, readback, rollback, and safety decisions. This is an evidence and audit layer, not a new live executor.
- Supervised apply remains limited to operator-approved `fader` and `gain` proposals. Apply requests require an idempotency key and bounded deltas before the manual gate; EQ, compressor, gate, pan, phase/delay, FX, routing, snapshots, scenes, training, and rules remain proposal-only, shadow-only, offline-only, or quarantined.
- Cockpit replay visibility belongs on evidence surfaces, not live controls. `start_replay_run` is safe only as offline replay/artifact generation; it must not become a WING/OSC write path.
- Shadow gain ride v0 is evidence-only: `backend/shadow_gain_rides.py` emits bounded `gain_ride_shadow` traces during explicit operator analysis, blocks oscillation and pumping risk, and must not create queueable live proposals until separately promoted with hardware proof.
- Decision memory v1 is append-only and derived only from verified/offline decision traces. Shadow-only traces stay out of durable memory, and memory rows provide context only; they do not grant executor authority or live readiness.
- Offline replay runs must leave audit evidence in the shared trace/memory flow: `start_replay_run` records one `offline_replay_v1` `replay_run` trace after artifact generation, with `offline_only=true`, `live_write=false`, and `mixer_transport_allowed=false`.
- Replay DTO rows should carry their own policy evidence for cockpit rendering: `trace_count`, `offline_only`, and `live_write` come from replay metrics so the UI can show proof facts without parsing report files.
- Frontend replay launch should update the cockpit immediately from the `replay_run_started` event and then refresh `get_replay_runs`; this keeps the latest offline run visible while preserving read-only/no-live-write behavior.
- Keep `Docs/FRONTEND_DTO_CONTRACTS.md` in sync with Product Layer evidence DTOs: `decision_traces`, `safety_monitor`, `replay_runs`, and `channel_intelligence` are display-safe read models, not apply authority.
- MuQ-A1 offline rock passes should A/B `large_system_polish` and extra drum-focus boosts instead of treating them as defaults; on SONG REPA, MuQ preferred the full agent/source-knowledge/AutoFOH/phase stack with `large_system_polish` disabled.
- MuQ-A1 FX-led rock passes should treat `0.9` whole-mix score as an aspirational, not fixed, gate unless calibrated against references; on SONG REPA the provided commercial reference scored `0.7959`, while subtle reference-guided FX plus section-selective modulation reached `0.7725` without clipping.
- Operator accepted the `MIX_AYAIC_FX_SOFTPRINT_FINAL` direction for Desktop/MIX: use AYAIC Mix Monolith-inspired level-plane/relative balance, shared filtered FX returns, real MuQ-A1 audit, and light mix-print mastering. Avoid MuQ score chasing through segment EQ, clipping, heavy limiting, or aggressive compression when this direction is requested.
- Operator accepted Ayaic level-plane balancing as the default rule for offline mixes and real-console soundcheck/live work: gain staging and balance must measure both individual sources and the summed stems/groups. Use Ayaic planes as the primary default (`-22` lead/main, `-25` strong sources and Drums/Bass/Guitars/BG Vocal buses, `-27` tucked backing vocals, `-30` keys/synth/ambient support, `-35` FX/deep effects), while live safety limits override target loudness.
- No-WING release checks must capture supervised gain dry-run proof through the Product Layer proposal path and classify it as read-only, non-live, non-promoting evidence until physical WING pre-read/apply/rollback proof exists.
- No-WING release summaries may expose next physical WING proof plan metadata, but only with `operator_required=true`, `execution_allowed_by_plan=false`, and no implication that Codex or automation may run it.
- Read-surface fixture evidence is part of no-WING readiness: declared fixture paths and pytest function names must be drift-checked before release proof reports are trusted, and no-WING summaries should keep remaining physical readback fixture gaps visible.
- Autonomy evidence overlays may connect proof artifacts to fader/gain promotion targets in release summaries and cockpit status rows, but they remain read-only review evidence: `live_write=false`, no automatic safety-row update, no live widening, and `full_autonomous_ready=false`.
- Product Layer supervised proof evidence snapshots may read existing proof-report artifacts to expose sanitized autonomy overlays, but snapshot reads must not generate artifacts, query mixers, or trust source safety flags that claim live authority.
- `artifacts/no_wing_release_check_current_cycle` is a Product Layer proof-evidence input root; treat its contents as read-only artifacts and normalize safety before cockpit display.
- Cockpit proof-evidence status should separate source availability (`supervised_proof_evidence`) from target-level readiness (`autonomy_evidence_overlay`) so operators can see whether missing proof is a missing artifact or a target-specific blocker.
- Proof-report artifacts need freshness metadata in Product Layer and cockpit rows; stale proof remains read-only evidence and should trigger rerun/review before any promotion discussion.
- Settings proof-evidence visibility should show source freshness and autonomy overlay status as read-only operator evidence; browser/e2e coverage must keep no-live-write command assertions in the same path.
- no-WING read-surface fixture evidence summaries must be recomputed from row data at the release gate; stale or malformed summary counts should fail validation while keeping physical readback gaps visible.
- no-WING read-surface fixture evidence rows must validate declared `tests/...` files and pytest function names at the release gate; stale fixture declarations are missing evidence, not proof.
- no-WING read-surface fixture evidence must keep row status, covered contracts, and remaining gaps internally consistent: cached DTO fixture coverage is partial until physical readback and hardware parity gaps are cleared.
- no-WING summaries should expose read-surface proof quality directly: declaration counts, row-summary recomputation, declaration validation, and contract/gap consistency enforcement belong in the top-level summary, not only verbose result details.
- Product Layer supervised proof evidence may read sibling no-WING `summary.json` as read-only metadata, but it must normalize proof-quality fields and keep physical readback gaps separate from promotion readiness.
- Settings should show read-surface proof quality as read-only operator evidence: metadata availability, physical gap count, fixture test count, and contract/gap enforcement are visibility facts, not live authority.
- Settings read-surface proof visibility should include compact physical gap surface IDs so the next hardware readback proof target is operator-visible without granting live read/write authority.
- Product Layer read-surface proof quality should expose `next_physical_readback_surface_id` as read-only planning evidence; frontend surfaces must not infer this as hardware permission or readiness.
- Codex autoloop reports should keep `## Три обязательных вопроса` directly adjacent to `## Объяснение простым языком`; extra top-level evidence belongs earlier so the operator-facing ending stays predictable.
- no-WING read-surface summaries should include `next_physical_readback_surface_id` and fail validation if it drifts from the first `physical_fixture_missing_surface_ids` entry.
- Product Layer should expose read-surface next-target consistency separately from the target label so malformed no-WING proof artifacts stay visible without granting live authority.
- Read-surface remediation actions should include physical fixture gaps even for cached/queryable inventory rows, so the next readback proof target is actionable without reclassifying cache coverage or granting live authority.
- Cockpit Module Status should render the read-surface remediation plan separately from supervised proof evidence, exposing next physical readback action IDs as read-only planning evidence.
- no-WING release summaries should validate read-surface remediation target/action consistency against fixture evidence so next physical readback actions cannot drift silently.
- no-WING read-surface remediation summaries should fail if multiple actions are flagged as the next physical readback target; the next target is planning evidence, not a queue.
- Raw mixer-state physical readback needs a schema contract before hardware proof artifacts are trusted: the contract may validate a future physical artifact, but it does not close `physical_readback_fixture_missing` or grant live authority.
- no-WING release summaries should validate the raw mixer-state physical readback contract as its own schema gate; contract readiness is release evidence, not a physical fixture or live-read permission.
- Product Layer and cockpit proof-evidence views should show raw mixer-state readback contract status as read-only planning metadata, while preserving `physical_fixture_gap_remains=true` and no live authority.
- Bus/group/matrix and mute promotion targets should have explicit fail-closed safety coverage rows instead of generic missing-row blockers; these contracts name the proof gates while preserving no bus/DCA/matrix/mute live authority.
- Promotion readiness should serialize bus/group/matrix and mute `evidence_requirement_ids` as read-only proof planning metadata, so gap summaries can name required freshness, readback, headroom, intent, physical fixture, rollback, and emergency proof without widening live authority.
- no-WING release autonomy gap rollups should preserve family-level evidence maps from promotion readiness (`evidence_requirement_ids_by_action_family` and `next_required_evidence_by_action_family`) so summary artifacts do not collapse bus/group/matrix and mute proof gates back into bucket counts.
- Channel identity/basic-control physical readback needs its own schema contract before fader, gain, mute, or pan proof artifacts are trusted; contract visibility must preserve `physical_fixture_gap_remains=true` and normalize live-authority claims to false.
- Channel EQ/dynamics/gate physical readback needs its own schema contract before EQ, compressor, gate, de-esser, frequency-separation, or unmasking proof artifacts are trusted; contract visibility must preserve `physical_fixture_gap_remains=true` and normalize live-authority claims to false.
- Bus/group/main/DCA physical readback needs its own schema contract before bus balance, DCA, master-bus, mix-balance, or performance-adaptation proof artifacts are trusted; contract visibility must preserve `physical_fixture_gap_remains=true` and normalize live-authority claims to false.
- Matrix-state physical readback needs its own schema contract before matrix balance, matrix EQ/dynamics/inserts, FX-send, routing, master-bus, or performance-adaptation proof artifacts are trusted; contract visibility must preserve `physical_fixture_gap_remains=true` and normalize live-authority claims to false.
- FX slots/inserts physical readback needs its own schema contract before FX send/insert, FX parameter, routing, mix-balance, or performance-adaptation proof artifacts are trusted; contract visibility must preserve `physical_fixture_gap_remains=true` and normalize live-authority claims to false.
- Routing/sends physical readback needs its own schema contract before routing, FX-send/insert, bus/matrix balance, scene/preset, mix-balance, or performance-adaptation proof artifacts are trusted; contract visibility must preserve `physical_fixture_gap_remains=true` and normalize live-authority claims to false.
- Snapshots/scenes physical readback needs its own schema contract before scene/preset, routing, mix-balance, or performance-adaptation proof artifacts are trusted; contract visibility must preserve `physical_fixture_gap_remains=true`, `scene_load_allowed=false`, `scene_save_allowed=false`, and normalize live-authority claims to false.
- User-presets physical readback needs its own schema contract before scene/preset, routing, mix-balance, or performance-adaptation proof artifacts are trusted; contract visibility must preserve `physical_fixture_gap_remains=true`, `preset_recall_allowed=false`, `preset_store_allowed=false`, and normalize live-authority claims to false.
- dLive-state physical readback needs its own schema contract before dLive/WING parity, fader, gain, mute, routing, scene/preset, mix-balance, or performance-adaptation proof artifacts are trusted; contract visibility must preserve `physical_fixture_gap_remains=true`, `midi_send_allowed=false`, `tcp_send_allowed=false`, `dlive_write_allowed=false`, and normalize live-authority claims to false.
- Read-surface physical readback capture planning is operator-required and non-executing: it may name the next hardware proof target and expected artifact contracts, but must preserve `execution_allowed_by_plan=false`, `live_write=false`, `live_authority_granted=false`, and `full_autonomous_ready=false`.
- Raw mixer-state physical readback artifact intake may accept a valid operator-captured JSON artifact as a review candidate, but it must keep `fixture_closure_allowed_by_intake=false`, `physical_fixture_gap_remains=true`, `live_authority_granted=false`, and `full_autonomous_ready=false`.
- Channel identity/basic-control physical readback artifact intake may accept operator-captured channel fader/trim/mute/pan evidence only as a manual-review candidate; it must keep `fixture_closure_allowed_by_intake=false`, `physical_fixture_gap_remains=true`, `live_authority_granted=false`, and `full_autonomous_ready=false`.
- Channel EQ/dynamics/gate physical readback artifact intake may accept operator-captured HPF/EQ/compressor/gate evidence only as a manual-review candidate; it must keep `fixture_closure_allowed_by_intake=false`, `physical_fixture_gap_remains=true`, `live_authority_granted=false`, and `full_autonomous_ready=false`.
- Bus/group/main/DCA physical readback artifact intake may accept operator-captured bus, main, DCA, assignment, and metering evidence only as a manual-review candidate; it must keep `fixture_closure_allowed_by_intake=false`, `physical_fixture_gap_remains=true`, `live_authority_granted=false`, and `full_autonomous_ready=false`.
- Matrix-state physical readback artifact intake may accept operator-captured matrix controls, EQ, dynamics, inserts, and metering evidence only as a manual-review candidate; it must keep `fixture_closure_allowed_by_intake=false`, `physical_fixture_gap_remains=true`, `live_authority_granted=false`, and `full_autonomous_ready=false`.
- FX slots/inserts physical readback artifact intake may accept operator-captured FX slot, parameter schema, and insert assignment evidence only as a manual-review candidate; it must keep `fixture_closure_allowed_by_intake=false`, `physical_fixture_gap_remains=true`, `live_authority_granted=false`, and `full_autonomous_ready=false`.
- Routing/sends physical readback artifact intake may accept operator-captured static route, bus send, main send, matrix send, and output-route evidence only as a manual-review candidate; it must keep `fixture_closure_allowed_by_intake=false`, `physical_fixture_gap_remains=true`, `live_authority_granted=false`, and `full_autonomous_ready=false`.
- Snapshots/scenes physical readback artifact intake may accept operator-captured scene catalog, active-scene, scoped diff, and rollback evidence only as a manual-review candidate; it must keep `fixture_closure_allowed_by_intake=false`, `physical_fixture_gap_remains=true`, `scene_load_allowed=false`, `scene_save_allowed=false`, `live_authority_granted=false`, and `full_autonomous_ready=false`.
- User-presets physical readback artifact intake may accept operator-captured preset catalog, scope, target, content fingerprint, scoped diff, and rollback evidence only as a manual-review candidate; it must keep `fixture_closure_allowed_by_intake=false`, `physical_fixture_gap_remains=true`, `preset_recall_allowed=false`, `preset_store_allowed=false`, `live_authority_granted=false`, and `full_autonomous_ready=false`.
- dLive-state physical readback artifact intake may accept operator-captured cached dLive state, passive MIDI/TCP state counts, and WING parity evidence only as a manual-review candidate; it must keep `fixture_closure_allowed_by_intake=false`, `physical_fixture_gap_remains=true`, `midi_send_allowed=false`, `tcp_send_allowed=false`, `dlive_write_allowed=false`, `dlive_apply_allowed=false`, `live_authority_granted=false`, and `full_autonomous_ready=false`.

## Memory Pipeline Lessons

- Agent Core, Project Layer, and Pipeline files under `.ai/` are the first-read
  memory layer for Codex, Kimi, Paperclip, and humans.
- Memory updates are documentation or append-only records; they do not grant
  runtime authority.
- Use `tools/memory/memory_lint.py` to verify required memory files, sections,
  and safety wording after changes.
