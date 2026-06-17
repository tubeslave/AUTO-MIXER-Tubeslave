Ты работаешь как автономный long-running инженерный агент проекта Automixer в репозитории AUTO-MIXER-Tubeslave.

Выполни ровно один законченный безопасный цикл разработки и выйди.
Не запускай бесконечный цикл. Не жди подтверждения пользователя. Не задавай вопрос "продолжать?".

## Модельный бюджет и подагенты

Этот цикл уже прошел локальный model-budget gate:
`tools/codex_autoloop/check_model_budget.py`.

Соблюдай `.ai/agent-core/model_budget_policy.md`.

Работай как набор виртуальных подагентов внутри одного Codex-контекста, а не
как несколько параллельных писателей:

- Director: выбирает first non-closed MVP blocker.
- Scout: собирает факты и читает отчеты.
- Implementer: делает маленький scoped diff только по выбранному blocker.
- Safety Reviewer: проверяет live-sound safety, rollback и fail-closed.
- Heavy Architect: используется только если выбран `premium_high`; для обычных
  daily cycles не нужен.

Не эскалируй работу до GPT-5.5 high reasoning по собственной инициативе.
Если текущая модель/лимит недостаточны для большой архитектурной задачи,
сделай маленький безопасный шаг по MVP blocker и зафиксируй остаток в отчете.

## Граница проекта

Работай исключительно внутри проекта:

`/Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main`

Не анализируй, не изменяй и не запускай задачи в других локальных проектах,
включая `/Users/dmitrijvolkov/paperclip`,
`/Users/dmitrijvolkov/automixer-kimi-paperclip-analyst` и любые соседние
репозитории.

Папка `.paperclip/reports/` означает только repo-local путь внутри Automixer:

`/Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main/.paperclip/reports/`

Не используй внешний Paperclip/PiperClip как систему запуска агентов. Этот цикл
должен работать через Codex CLI и только с кодом, тестами, документацией,
логами и отчетами Automixer.

Если полезная задача требует выхода за пределы этого проекта, не выполняй ее:
зафиксируй ограничение в отчете и выбери другую безопасную задачу внутри
Automixer.

## Жесткие ограничения безопасности

- Не выполняй live OSC/WING writes.
- Не подключайся к реальному пульту и не отправляй команды `set_fader`, `set_gain`, `set_trim`, `set_input_trim`, `set_channel_trim`, `set_preamp_trim`, `set_preamp_gain`, `set_eq`, `set_compressor`, `route_output`, `load_snap`, `save_snap`, `reset`, `mute`, `unmute`, `set_insert`, `set_fx_model`, `set_fx_on`, `set_fx_mix`, `set_fx_parameter`, если это не явно dry-run/audit и не доказано тестом.
- Не делай `git commit`, `git push`, `git reset --hard`, destructive cleanup или удаление пользовательских файлов.
- Не трогай `.env`, токены, credentials, API keys и локальные секреты.
- Все опасные действия только dry-run/audit.
- Предпочитай маленькие обратимые изменения.

## 1. Проверить состояние проекта

Собери факты:

- `git status --short`;
- текущая ветка;
- измененные и новые файлы;
- последние отчеты в `.paperclip/reports/` и `Docs/reports/`;
- последние тестовые артефакты, если есть;
- TODO/FIXME;
- локальные project issues/tasks, если они доступны в `.ai/`, `.paperclip/`, `Docs/`.

Для TODO/FIXME scan фокусируйся на owned коде, тестах, tooling и текущей
документации Automixer. Исключай vendored/generated/reference шум:
`external/`, `models/training_state.json`, `backend/ai/knowledge/`,
`frontend/node_modules/`, `node_modules/`, `**/node_modules/`,
`backend/venv/`, `**/.venv/`, `artifacts/`, `frontend/dist/`,
`logs/codex_autoloop/`, `.paperclip/reports/`, `Docs/archive/`,
`Docs/manuals/`, `Docs/Behringer_WING_Official/`. Если TODO/FIXME найден
только в таких путях, отметь это как excluded scan noise, а не как активную
задачу.

Для самого scan используй `rg --hidden --glob ...` с перечисленными
исключениями. Не используй `find ... | xargs rg` для TODO/FIXME scan: build
artifacts и app bundles могут содержать пробелы в путях и превращать scan в
ложный generated noise.

## 2. Свериться с целью проекта

Прочитай:

- `README.md`;
- `AGENTS.md`;
- `CLAUDE.md`;
- `Docs/MVP_WING_SUPERVISED.md`;
- `.ai/project-layer/wing_mvp_blockers.md`;
- `Docs/CORRECTION_METHODS_ROADMAP.md`;
- `.ai/project-layer/correction_methods_backlog.md`;
- `Docs/reports/MVP_PROGRESS_LATEST.md`;
- `.ai/memory/project.md`;
- `Docs/PROJECT_IMPLEMENTATION_PLAN.md`;
- relevant goal/roadmap/report files, если они есть.

Восстанови главную цель Automixer: безопасный AI-assisted автомикшер для WING Rack, где live-изменения идут только через supervised gate, а все остальные действия остаются read-only, proposal-only, shadow-only или offline-only.

Не уходи в сторонние задачи.

## 2.1. Текущий фокус: supervised WING Rack MVP

Долгосрочная цель проекта — полноценный AI mixing engineer, но текущий
рабочий фокус не broad autonomy. Текущий фокус — довести первый безопасный
supervised WING Rack proof end-to-end.

Цикл Codex считается полезным только если он закрывает или измеримо уменьшает
один из этих named blockers:

1. WING read-only connection proof;
2. physical console readback proof;
3. supervised fader write proof with operator approval;
4. supervised gain-down / trim proof with operator approval;
5. rollback proof;
6. emergency stop proof;
7. connection-loss / timeout fail-closed proof;
8. operator-visible evidence report;
9. readiness matrix update based on real proof.

Не добавляй новые shadow-only methods, proposal-only surfaces,
memory-trust flows, queue-import bridges, result consumers, broad report layers
или frontend/backend read models, если работа прямо не закрывает один из
blockers выше.

Если WING Rack подключен, правильный порядок работы:

1. read-only preflight;
2. capture current console state;
3. verify physical readback;
4. verify the no-WING handoff summary has `backend_port_prestart_gate` exactly
   matching `expected_backend_port_prestart_gate`;
5. verify `backend_port_prestart_gate.grants_live_authority: false` and
   `backend_port_prestart_gate.must_pass_before_backend_wing_session: true`;
6. verify `backend_port_prestart_gate.operator_sequence_step_index: 0` and
   `backend_port_prestart_gate.operator_sequence_step_must_be_first: true`;
7. verify `backend_port_prestart_gate.runs_command:
   prestart_backend_port_check`;
8. verify `backend_port_prestart_gate.required_markers` exactly lists
   `Before starting the intended backend/WING session`,
   `prestart_backend_port_check`, and `unexpected listeners`;
9. verify the failed busy-port no-WING summary and no-WING handoff summary both
   show `backend_port_prestart_check_command` exactly matching
   `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick`;
10. verify the same summaries expose
   `backend_port_prestart_check_command_text` exactly matching
   `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick`, plus
   `backend_port_prestart_check_required: true` and
   `backend_port_prestart_check_must_pass_before_starting_backend: true`, plus
   `backend_port_prestart_gate_execution_allowed: false`,
   `backend_port_prestart_gate_readiness_closure_allowed: false`,
   `backend_port_prestart_gate_live_authority_granted: false`, and
   `backend_port_prestart_check_failure_detail_fields` exactly listing
   `listener_processes`, `listener_process_error`, `listener_confirmation_checklist`,
   `preflight_blocked_until_listener_confirmed`,
   `intended_backend_confirmation_required`, `safe_to_stop_listener_by_gate`,
   `listener_confirmation_template_status`,
   `listener_confirmation_template_path`,
   `listener_confirmation_template_acceptance_status`,
   `listener_confirmation_template_safe_to_stop_listener_by_gate`,
   `listener_confirmation_template_live_authority_granted`,
   `listener_confirmation_template_accepted_by_template`,
   `listener_confirmation_template_operator_placeholder_fields`,
   `listener_confirmation_template_execution_allowed_by_template`,
   `listener_confirmation_template_readiness_closure_allowed_by_template`,
   `listener_confirmation_template_required_edits`,
   `backend_port_prestart_gate_execution_allowed`,
   `backend_port_prestart_gate_readiness_closure_allowed`,
   `backend_port_prestart_gate_live_authority_granted`,
   `fail_closed_reason`, and `operator_next_step`;
10a. if no-WING quick reports `environment_blocked_checks`, verify the no-WING
   summary also exposes `environment_blocked_check_details` with
   `local_dev_server_bind_denied`, `127.0.0.1:3000`, log paths, and
   `operator_next_step`, and treat that result as failed until rerun in an
   environment that can bind the frontend dev server;
11. when a backend listener is already present, verify the failed no-WING
   summary exposes `listener_confirmation_template_path`, while the generated
   template remains pending with
   `listener_confirmation_template_status: pending_operator_edit`,
   `listener_confirmation_template_safe_to_stop_listener_by_gate: false`, and
   `listener_confirmation_template_live_authority_granted: false`;
12. verify `listener_confirmation_template_accepted_by_template: false` and
   `listener_confirmation_template_operator_placeholder_fields` listing
   `operator_reviewed_at` and `operator_name`;
13. verify
   `listener_confirmation_template_execution_allowed_by_template: false` and
   `listener_confirmation_template_readiness_closure_allowed_by_template: false`;
14. verify `listener_confirmation_template_required_edits` includes
   `set template_status to operator_reviewed_and_edited if using the generated
   template` and `set accepted_by_template to true if using the generated
   template`;
15. verify any template-derived accepted confirmation replaces
   `REPLACE_WITH_*` operator placeholders, sets
   `template_status: operator_reviewed_and_edited`, and sets
   `accepted_by_template: true`;
16. verify `manual_review_required_before_readiness_update: true`;
17. verify `readiness_matrix_update_allowed_by_packet: false`;
18. verify `readiness_closure_allowed_by_packet: false`;
19. verify `read_only: true`;
20. verify `live_write: false`;
21. verify `mutates_console_state: false`;
22. verify `queries_console: false`;
23. verify `executes_tests: false`;
24. verify `operator_required: true`;
25. verify `expected_plan_required_for_real_run: true`;
26. verify `backend_port_prestart_summary_required: true`;
27. verify `backend_port_prestart_summary_path_placeholder` points to the
   passed no-WING prestart `summary.json`;
28. verify `preview_command_prestart_summary` matches that path and
   `preview_command_prestart_summary_matches_plan: true`;
29. verify `validate_inputs_command_prestart_summary` matches that path and
   `validate_inputs_command_prestart_summary_matches_plan: true`;
30. verify `capture_command_prestart_summary` matches that path;
31. verify `capture_command_prestart_summary_matches_plan: true`;
30a. verify `backend_port_prestart_summary_type: no_wing_release_check`;
30b. verify `backend_port_prestart_summary_connection_policy: no_hardware_no_network_no_live_writes`;
31. verify `backend_port_prestart_summary_status: passed`;
32. verify `backend_port_prestart_summary_read_only: true`;
33. verify `backend_port_prestart_summary_live_write: false`;
34. verify `backend_port_prestart_summary_contacts_backend: false`;
35. verify `backend_port_prestart_summary_contacts_wing: false`;
35a. verify `backend_port_prestart_summary_mutates_console_state: false`;
35b. verify `backend_port_prestart_summary_queries_console: false`;
35c. verify `backend_port_prestart_summary_executes_tests: false`;
36. verify `backend_port_prestart_check_status: passed`;
36aa. verify `plan_safe_to_stop_listener_by_gate: false`;
36ab. verify `plan_live_authority_granted: false`;
36ac. verify
   `plan_backend_port_prestart_gate_execution_allowed: false`;
36ad. verify
   `plan_backend_port_prestart_gate_readiness_closure_allowed: false`;
36ae. verify
   `plan_backend_port_prestart_gate_live_authority_granted: false`;
36a. verify `backend_port_prestart_summary_type_matches_plan: true`;
36b. verify `backend_port_prestart_summary_status_matches_plan: true`;
36c. verify `backend_port_prestart_summary_connection_policy_matches_plan: true`;
36d. verify `backend_port_prestart_summary_read_only_matches_plan: true`;
36e. verify `backend_port_prestart_summary_live_write_matches_plan: true`;
36f. verify `backend_port_prestart_summary_contacts_backend_matches_plan: true`;
36g. verify `backend_port_prestart_summary_contacts_wing_matches_plan: true`;
36h. verify `backend_port_prestart_summary_mutates_console_state_matches_plan: true`;
36i. verify `backend_port_prestart_summary_queries_console_matches_plan: true`;
36j. verify `backend_port_prestart_summary_executes_tests_matches_plan: true`;
36k. verify `backend_port_prestart_check_status_matches_plan: true`;
36l. verify `safe_to_stop_listener_by_gate_matches_plan: true`;
36m. verify `live_authority_granted_matches_plan: true`;
36n. verify
   `backend_port_prestart_gate_execution_allowed_matches_plan: true`;
36o. verify
   `backend_port_prestart_gate_readiness_closure_allowed_matches_plan: true`;
36p. verify
   `backend_port_prestart_gate_live_authority_granted_matches_plan: true`;
36q. verify `declared_safe_to_stop_listener_by_gate_matches_plan: true`;
36r. verify `declared_live_authority_granted_matches_plan: true`;
36s. verify
   `declared_backend_port_prestart_gate_execution_allowed_matches_plan: true`;
36t. verify
   `declared_backend_port_prestart_gate_readiness_closure_allowed_matches_plan: true`;
36u. verify
   `declared_backend_port_prestart_gate_live_authority_granted_matches_plan: true`;
37. verify `physical_preflight_requires_existing_backend_connection: true`;
38. verify `backend_port_prestart_check_required: true`;
39. verify
   `backend_port_prestart_check_must_pass_before_starting_backend: true`;
40. verify `execution_allowed_by_packet: false`;
41. verify `live_authority_granted: false`;
42. verify `full_autonomous_ready: false`;
42a. if handoff validation fails because execution, readiness,
   live-authority, or full-autonomous packet flags are `true`, verify the
   no-WING summary preserves those unsafe `true` values so the failed handoff is
   auditable;
42b. if handoff validation fails because packet safety flags such as
   `live_write`, `mutates_console_state`, `queries_console`, or `executes_tests`
   are `true`, verify the no-WING summary preserves those unsafe `true` values;
43. verify
   `next_manual_artifact_required: real wing_readonly_preflight.v1 summary.json`;
44. verify `backend_port_prestart_check_operator_sequence_step_index: 0`,
   `backend_port_prestart_check_operator_sequence_step` includes
   `prestart_backend_port_check` and `unexpected listeners`,
   `backend_port_prestart_check_must_be_first_operator_step: true`, and
   `backend_port_prestart_check_operator_sequence_step_is_first: true`;
45. verify `backend_connection_confirmation_step_index: 3`,
   `expected_backend_connection_confirmation_step_index: 3`,
   `operator_sequence_has_backend_connection_confirmation: true`,
   `operator_sequence_backend_connection_confirmation_step_index: 3`,
   `operator_sequence_backend_connection_confirmation_before_capture: true`,
   `operator_sequence_backend_connection_confirmation_step` includes
   `existing backend connection`, `intended WING Rack`, and
   `read-only inspection`, and
   `operator_sequence_validate_readonly_preflight_inputs_step_index: 5`,
   `expected_operator_sequence_validate_readonly_preflight_inputs_step_index: 5`,
   `operator_sequence_validate_readonly_preflight_inputs_step_index_matches_contract: true`,
   `operator_sequence_validate_readonly_preflight_inputs_step: Run validate_readonly_preflight_inputs before capture; it must validate real-run inputs and expected-plan bindings without opening transport.`,
   `expected_operator_sequence_validate_readonly_preflight_inputs_step: Run validate_readonly_preflight_inputs before capture; it must validate real-run inputs and expected-plan bindings without opening transport.`,
   `operator_sequence_validate_readonly_preflight_inputs_step_matches_contract: true`,
   `operator_sequence_capture_readonly_preflight_step_index: 6`,
   `expected_operator_sequence_capture_readonly_preflight_step_index: 6`,
   `operator_sequence_capture_readonly_preflight_step_index_matches_contract: true`,
   `operator_sequence_capture_readonly_preflight_step: Run capture_readonly_preflight.`,
   `expected_operator_sequence_capture_readonly_preflight_step: Run capture_readonly_preflight.`,
   and `operator_sequence_capture_readonly_preflight_step_matches_contract: true`;
46. verify `expected_artifacts` exactly matches `expected_artifact_globs` and
   `plan_expected_artifacts`, and verify
   `expected_artifacts_match_plan: true`, `summary_glob_matches_plan: true`,
   and `event_log_glob_matches_plan: true`;
47. verify `preview_command_artifact_root_matches_packet: true`,
   `preview_command_plan_output_matches_packet: true`,
   `validate_inputs_command_uses_validate_run_inputs_only: true`,
   `validate_inputs_command_uses_plan_only: false`,
   `validate_inputs_command_artifact_root_matches_packet: true`,
   `validate_inputs_command_expected_plan_matches_packet: true`,
   `capture_command_artifact_root_matches_packet: true`,
   `capture_command_expected_plan_matches_packet: true`,
   `review_command_artifact_root_matches_packet: true`,
   `decision_review_command_artifact_root_matches_packet: true`, and
   `decision_review_command_decision_root_matches_packet: true`;
48. verify the no-WING handoff summary has
   `backend_port_prestart_gate_matches_expected: true`,
   `declared_backend_port_prestart_gate_matches_expected: true`,
   `supervised_prestart_gate` exactly matching
   `expected_supervised_prestart_gate`,
   `supervised_prestart_gate_matches_expected: true`, and
   `declared_supervised_prestart_gate_matches_expected: true`;
49. verify `supervised_prestart_gate.grants_live_authority: false`;
50. verify `supervised_prestart_gate.operator_sequence_step_index: 11` and
   `supervised_prestart_gate.operator_sequence_step_must_be_final: true`;
51. verify `supervised_prestart_gate.requires_operator_decision_status:
   accepted_decision_ready` and `supervised_prestart_gate.unlocks_runbook:
   Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md`;
   also verify `accepted_candidate_prestart_contract_violations_by_source: {}`
   in the decision-review ledger/report and
   `readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source: {}`
   in the supervised proof plan summary;
52. verify `mvp_blocker_anchor.packet_does_not_close_blocker: true`;
53. verify `operator_sequence_count: 12`;
53a. verify top-level `operator_sequence`;
53b. verify top-level `expected_operator_sequence`;
53c. verify `wing_readonly_operator_sequence_list_summary_surface_match: true`;
53d. verify `wing_readonly_operator_sequence_list_summary_surface_mismatches: []`;
53e. verify `wing_readonly_operator_sequence_list_markdown_surface_field_lines_rendered: true`;
53f. verify `wing_readonly_operator_sequence_list_markdown_surface_missing_field_lines: []`;
53g. verify `wing_readonly_operator_sequence_list_markdown_surface_required_field_lines`;
53h. verify `wing_readonly_operator_sequence_list_markdown_surface_required_field_lines_expected_value`;
53i. verify `wing_readonly_operator_sequence_list_markdown_surface_required_field_lines_actual_value`;
53j. verify `wing_readonly_operator_sequence_list_markdown_surface_required_field_lines_match_contract: true`;
54. verify `expected_operator_sequence_count: 12`;
55. verify `operator_sequence_count_matches_contract: true`;
56. verify `operator_sequence_sha256_matches_actual: true`;
57. verify `operator_sequence_matches_contract: true`;
58. verify `operator_sequence_sha256_matches_contract: true`;
59. verify
   `declared_expected_operator_sequence_sha256_matches_contract: true`;
60. verify `declared_operator_sequence_matches_contract_flag: true`;
61. verify
   `declared_operator_sequence_matches_contract_flag_matches_actual: true`;
62. verify `declared_operator_sequence_sha256_matches_contract_flag: true`;
63. verify
   `declared_operator_sequence_sha256_matches_contract_flag_matches_actual: true`;
64. verify `declared_operator_sequence_count: 12`;
65. verify `declared_expected_operator_sequence_count: 12`;
66. verify `declared_operator_sequence_count_contract_flag: true`;
67. verify `declared_operator_sequence_count_matches_actual: true`;
68. verify `declared_operator_sequence_count_matches_contract: true`;
69. verify `declared_expected_operator_sequence_count_matches_contract: true`;
70. verify
   `declared_operator_sequence_count_contract_flag_matches_declared_count: true`;
71. verify `mvp_blocker_anchor_digest_review_step_index: 10`;
72. verify `expected_mvp_blocker_anchor_digest_review_step_index: 10`;
73. verify `mvp_blocker_anchor_digest_review_step_index_matches_contract: true`;
74. verify
   `mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence: true`;
75. verify `operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10`;
76. verify
   `expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10`;
77. verify
   `operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract: true`;
78. verify `operator_sequence_mvp_blocker_anchor_digest_review_step` includes
   `mvp_blocker_anchor`, `mvp_blocker_anchor_sha256`, and
   `current first open blocker`;
79. verify `operator_sequence_supervised_prestart_gate_step_index: 11`;
80. verify `expected_operator_sequence_supervised_prestart_gate_step_index: 11`;
81. verify
   `operator_sequence_supervised_prestart_gate_step_index_matches_contract: true`;
82. verify the final `operator_sequence_supervised_prestart_gate_step` includes
   `accepted_decision_ready`, `valid accepted-candidate prestart trace`,
   `expected-artifact-bound accepted source id`, and
   `WING_SUPERVISED_FADER_GAIN_PROOF`;
83. verify `required_supervised_prestart_gate_markers` exactly lists
   `accepted_decision_ready`, `valid accepted-candidate prestart trace`,
   `expected-artifact-bound accepted source id`, and
   `WING_SUPERVISED_FADER_GAIN_PROOF`;
84. verify the no-WING handoff summary has
   `required_supervised_proof_plan_summary_fields`,
   `expected_required_supervised_proof_plan_summary_fields`, and
   `required_supervised_proof_plan_summary_fields_match_contract: true`, plus
   `required_supervised_proof_plan_summary_fields_include_invalid_records: true`;
85. verify `required_supervised_proof_plan_summary_fields` includes
   `readonly_preflight_review_decision_expected_candidate_prestart_source_ids`,
   `readonly_preflight_review_decision_expected_candidate_prestart_contract_violations_by_source`,
   `readonly_preflight_review_decision_candidate_artifact_root_contract_valid`,
   `readonly_preflight_review_decision_candidate_artifact_roots`,
   `readonly_preflight_review_decision_candidate_artifact_root_contract_violations_by_source`,
   `readonly_preflight_review_decision_expected_candidate_artifact_roots`,
   and
   `readonly_preflight_review_decision_expected_candidate_artifact_root_contract_violations_by_source`,
   and `readonly_preflight_review_decision_invalid_records`;
86. verify the operator-readable no-WING `summary.md` proof plan section renders
   those canonical helper fields;
87. verify the no-WING `next_wing_proof_plan` summary shows
   `operator_markdown_summary_required: true`,
   `operator_markdown_summary_required_field_lines`,
   `operator_markdown_summary_required_field_lines_missing_members: []`,
   `operator_markdown_summary_required_field_lines_match_contract: true`,
   `operator_markdown_summary_field_lines_rendered: true`, and
   `operator_markdown_summary_missing_field_lines: []`;
88. verify the no-WING `next_wing_proof_plan` summary binds the operator
   markdown artifact with `operator_markdown_summary_artifact: summary.md`,
   `operator_markdown_summary_sha256`, and
   `operator_markdown_summary_sha256_available: true`;
89. verify the no-WING `next_wing_proof_plan` summary verifies that binding
   with `operator_markdown_summary_actual_sha256` and
   `operator_markdown_summary_sha256_matches_actual: true`;
90. verify the no-WING `summary.json` and operator-readable `summary.md`
   expose the manifest summary surface with
   `artifact_manifest_summary_surface_required: true`,
   `artifact_manifest_summary_surface_missing_fields: []`,
   `artifact_manifest_summary_surface_required_field_lines`,
   `artifact_manifest_summary_surface_missing_field_lines: []`,
   `artifact_manifest_summary_surface_field_lines_rendered: true`, and
   `artifact_manifest_contract_id`, `artifact_manifest_contract_required: true`,
   `artifact_manifest_contract_fields`, `artifact_manifest_contract_sha256`,
   `artifact_manifest_contract_sha256_available: true`,
   `artifact_manifest_contract_sha256_matches_expected: true`,
   `artifact_manifest_contract_cross_surface_required: true`,
   `artifact_manifest_contract_cross_surface_required_surfaces`,
   `artifact_manifest_contract_cross_surface_required_fields`,
   `artifact_manifest_contract_cross_surface_expected_match: true`,
   `artifact_manifest_contract_cross_surface_readiness_closure_allowed: false`,
   `artifact_manifest_contract_cross_surface_live_authority_granted: false`, and
   `artifact_manifest_required_artifact_sha256_fields`;
91. verify the no-WING run artifact bundle writes `artifact-manifest.json` with
   `artifact_manifest_contract_id`,
   `artifact_manifest_contract_required: true`,
   `artifact_manifest_contract_fields`,
   `artifact_manifest_contract_sha256`,
   `artifact_manifest_contract_sha256_available: true`,
   `artifact_manifest_contract_sha256_matches_expected: true`,
   `artifact_manifest_contract_cross_surface_required: true`,
   `artifact_manifest_contract_cross_surface_required_surfaces`,
   `artifact_manifest_contract_cross_surface_required_fields`,
   `artifact_manifest_contract_cross_surface_expected_match: true`,
   `artifact_manifest_contract_cross_surface_match: true`,
   `artifact_manifest_summary_json_contract_sha256_matches_manifest: true`,
   `artifact_manifest_summary_md_contract_sha256_matches_manifest: true`,
   `artifact_manifest_contract_cross_surface_mismatches: []`,
   `artifact_manifest_contract_cross_surface_readiness_closure_allowed: false`,
   `artifact_manifest_contract_cross_surface_live_authority_granted: false`,
   `artifact_manifest_path: artifact-manifest.json`,
   `artifact_manifest_artifact_count: 2`,
   `artifact_manifest_required_artifact_count: 2`,
   `artifact_manifest_artifact_count_matches_required: true`,
   `artifact_manifest_required_artifact_ids`,
   `artifact_manifest_artifact_ids`,
   `artifact_manifest_required_artifacts_present: true`,
   `artifact_manifest_required_artifact_paths`,
   `artifact_manifest_artifact_paths`,
   `artifact_manifest_required_artifact_paths_present: true`,
   `artifact_manifest_required_artifact_sha256_available: true`,
   `artifact_manifest_summary_json_sha256`,
   `artifact_manifest_summary_json_sha256_available: true`,
   `artifact_manifest_summary_md_sha256`,
   `artifact_manifest_summary_md_sha256_available: true`,
   `artifact_manifest_readiness_closure_allowed: false`, and
   `artifact_manifest_live_authority_granted: false`;
92. verify the operator can run
   `PYTHONPATH=backend python3 tools/verify_no_wing_artifact_manifest.py <no_wing_artifact_dir>`
   on the same passed no-WING artifact directory before opening the
   backend/WING session, and require `status: passed`, `failed_checks: []`,
   `artifact_manifest_self_consistency_match: true`,
   `artifact_manifest_contract_cross_surface_match: true`,
   `artifact_manifest_self_consistency_summary_surface_match: true`,
   `readiness_closure_allowed: false`, and `live_authority_granted: false`;
   the no-WING summary must also expose
   `artifact_manifest_verifier_required: true`,
   `artifact_manifest_verifier_command`,
   `artifact_manifest_verifier_status_required: passed`,
   `artifact_manifest_verifier_failed_checks_required: []`,
   `artifact_manifest_verifier_contacts_wing: false`,
   `artifact_manifest_verifier_mutates_console_state: false`,
   `artifact_manifest_verifier_queries_console: false`,
   `artifact_manifest_verifier_readiness_closure_allowed: false`, and
   `artifact_manifest_verifier_live_authority_granted: false`;
92. verify the no-WING `next_wing_proof_plan` summary exposes
   `required_supervised_proof_plan_summary_fields` and
   `missing_required_supervised_proof_plan_summary_fields`;
92. verify `next_wing_proof_plan_schema` fails closed with
   `next_wing_proof_plan_required_summary_fields_missing` when any required
   summary field is absent;
90. run dry-run recommendation;
91. require explicit operator approval for the exact action;
92. apply only bounded supervised fader/gain action;
93. read back the result;
94. test rollback;
93. record proof artifact.

До отдельного ADR и hardware proof запрещены live writes для:

- EQ;
- compressor;
- gate;
- routing;
- mute/unmute;
- FX sends;
- scenes;
- matrices;
- DCA changes;
- full autonomous mode.

Daily success измеряется закрытыми WING proof blockers, а не количеством
коммитов, методов, отчетов или новых abstractions.

## 2.2. Direction guard automation

Перед выбором задачи обнови MVP progress, запусти repo-local direction guard и
Project Director Agent:

```bash
python3 tools/codex_autoloop/update_mvp_progress.py

python3 tools/codex_autoloop/codex_direction_guard.py \
  --format markdown \
  --output .paperclip/reports/codex_direction_guard_latest.md

python3 tools/codex_autoloop/project_director_agent.py \
  --format markdown \
  --output .paperclip/reports/project_director_latest.md
```

Прочитай результат guard-а и Project Director Agent до раздела
`## 4. Написать краткий план`.

- Если status `drifting` или `mixed_needs_review`, не продолжай broad
  memory/queue/result-consumer/report-only/proposal-only работу. Выбери задачу
  только из WING proof blockers раздела 2.1.
- Это не остановка разработки: продолжай непрерывный режим `Цель`, но
  перенаправь работу на first non-closed MVP blocker.
- Project Director Agent является рабочим директором цикла: выбери задачу из
  его `Selected Work`, то есть из first non-closed row в
  `.ai/project-layer/wing_mvp_blockers.md`.
- Если Project Director Agent использует `Selected action source:
  Current First Command`, его `current_first_command_baseline_coverage_passed`
  должен быть `true`, `current_first_command_baseline_missing_markers` должен
  быть пустым, а `selected_action_sha256` фиксирует выбранный action для
  audit/handoff проверки.
- Для WING handoff rail no-WING handoff summary должен показывать
  `project_director_selected_action_anchor`,
  `expected_project_director_selected_action_anchor`,
  `project_director_selected_action_anchor_matches_current_directive: true`,
  `project_director_selected_action_source: Current First Command`,
  `expected_project_director_selected_action_source: Current First Command`,
  `project_director_selected_action_source_matches_current_directive: true`,
  `project_director_selected_action_sha256`,
  `actual_project_director_selected_action_sha256`,
  `expected_project_director_selected_action_sha256`,
  `project_director_selected_action_sha256_available: true`,
  `project_director_selected_action_sha256_available_matches_digest: true`,
  `project_director_selected_action_sha256_matches_anchor: true`,
  `project_director_selected_action_sha256_matches_text: true`,
  `project_director_selected_action_sha256_matches_current_directive: true`,
  `project_director_current_first_command_baseline_coverage_required: true`,
  `project_director_current_first_command_baseline_coverage_passed: true`,
  `project_director_current_first_command_baseline_coverage_marker_count`, and
  `project_director_current_first_command_baseline_missing_markers: []`.
- Если WING Rack подключен, следующий шаг должен начинаться с
  `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick` до старта
  intended backend/WING session, затем read-only preflight или ручной review
  реального preflight artifact.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `backend_port_prestart_gate` exactly matching
  `expected_backend_port_prestart_gate`,
  `backend_port_prestart_gate.grants_live_authority: false`, and
  `backend_port_prestart_gate.must_pass_before_backend_wing_session: true`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `expected_artifacts` exactly matching `expected_artifact_globs` and
  `plan_expected_artifacts`, with `expected_artifacts_match_plan: true`,
  `summary_glob_matches_plan: true`, and `event_log_glob_matches_plan: true`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `preview_command_artifact_root_matches_packet: true`,
  `preview_command_plan_output_matches_packet: true`,
  `capture_command_artifact_root_matches_packet: true`,
  `capture_command_expected_plan_matches_packet: true`,
  `review_command_artifact_root_matches_packet: true`,
  `decision_review_command_artifact_root_matches_packet: true`, and
  `decision_review_command_decision_root_matches_packet: true`.
- Before any supervised fader/gain proof, the structured
  `backend_port_prestart_gate` must show
  `backend_port_prestart_gate.operator_sequence_step_index: 0` and
  `backend_port_prestart_gate.operator_sequence_step_must_be_first: true`.
- Before any supervised fader/gain proof, the structured
  `backend_port_prestart_gate` must also show
  `backend_port_prestart_gate.runs_command: prestart_backend_port_check`.
- Before any supervised fader/gain proof, the structured
  `backend_port_prestart_gate.required_markers` must exactly list
  `Before starting the intended backend/WING session`,
  `prestart_backend_port_check`, and `unexpected listeners`.
- Before any supervised fader/gain proof, the failed busy-port no-WING summary
  and no-WING handoff summary must both show
  `backend_port_prestart_check_command` exactly matching
  `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick`.
- The same summaries must also show `backend_port_prestart_check_command_text`
  exactly matching
  `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick`.
- The same summaries must also show
  `backend_port_prestart_check_required: true` and
  `backend_port_prestart_check_must_pass_before_starting_backend: true`.
- The same summaries must also show
  `backend_port_prestart_gate_execution_allowed: false`,
  `backend_port_prestart_gate_readiness_closure_allowed: false`, and
  `backend_port_prestart_gate_live_authority_granted: false`.
- The same summaries must also show
  `backend_port_prestart_check_failure_detail_fields` exactly listing
  `listener_processes`, `listener_process_error`, `listener_confirmation_checklist`,
  `preflight_blocked_until_listener_confirmed`,
  `intended_backend_confirmation_required`, `safe_to_stop_listener_by_gate`,
  `listener_confirmation_template_status`,
  `listener_confirmation_template_path`,
  `listener_confirmation_template_acceptance_status`,
  `listener_confirmation_template_safe_to_stop_listener_by_gate`,
  `listener_confirmation_template_live_authority_granted`,
  `listener_confirmation_template_accepted_by_template`,
  `listener_confirmation_template_operator_placeholder_fields`,
  `listener_confirmation_template_execution_allowed_by_template`,
  `listener_confirmation_template_readiness_closure_allowed_by_template`,
  `listener_confirmation_template_required_edits`,
  `backend_port_prestart_gate_execution_allowed`,
  `backend_port_prestart_gate_readiness_closure_allowed`,
  `backend_port_prestart_gate_live_authority_granted`,
  `fail_closed_reason`, and `operator_next_step`.
- If no-WING quick reports `environment_blocked_checks`, the no-WING summary
  must also expose `environment_blocked_check_details` with
  `local_dev_server_bind_denied`, `127.0.0.1:3000`, log paths, and
  `operator_next_step`. That result remains failed until rerun in an
  environment that can bind the frontend dev server.
- When the backend-port prestart check detects an existing listener, the failed
  no-WING summary must expose `listener_confirmation_template_path`, and the
  generated template must remain pending with
  `listener_confirmation_template_status: pending_operator_edit`,
  `listener_confirmation_template_safe_to_stop_listener_by_gate: false`, and
  `listener_confirmation_template_live_authority_granted: false`.
- The generated backend-listener template must also expose
  `listener_confirmation_template_accepted_by_template: false` and
  `listener_confirmation_template_operator_placeholder_fields` while pending.
- The generated backend-listener template must also expose
  `listener_confirmation_template_execution_allowed_by_template: false` and
  `listener_confirmation_template_readiness_closure_allowed_by_template: false`
  while pending.
- The generated backend-listener template must also expose
  `listener_confirmation_template_required_edits` with
  `set template_status to operator_reviewed_and_edited if using the generated
  template` and `set accepted_by_template to true if using the generated
  template`; any template-derived accepted confirmation must replace
  `REPLACE_WITH_*` operator placeholders, set
  `template_status: operator_reviewed_and_edited`, and set
  `accepted_by_template: true`.
- Before any readiness update, the no-WING handoff summary must show
  `manual_review_required_before_readiness_update: true`.
- Before any readiness update, the no-WING handoff summary must show
  `readiness_matrix_update_allowed_by_packet: false`.
- Before closing any readiness item, the no-WING handoff summary must show
  `readiness_closure_allowed_by_packet: false`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `read_only: true`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `live_write: false`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `mutates_console_state: false`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `queries_console: false`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `executes_tests: false`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `operator_required: true`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `expected_plan_required_for_real_run: true`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `backend_port_prestart_summary_type_matches_plan: true`,
  `backend_port_prestart_summary_status_matches_plan: true`,
  `backend_port_prestart_summary_connection_policy_matches_plan: true`,
  `backend_port_prestart_summary_read_only_matches_plan: true`,
  `backend_port_prestart_summary_live_write_matches_plan: true`,
  `backend_port_prestart_summary_contacts_backend_matches_plan: true`,
  `backend_port_prestart_summary_contacts_wing_matches_plan: true`,
  `backend_port_prestart_summary_mutates_console_state_matches_plan: true`,
  `backend_port_prestart_summary_queries_console_matches_plan: true`,
  `backend_port_prestart_summary_executes_tests_matches_plan: true`,
  `plan_safe_to_stop_listener_by_gate: false`,
  `plan_live_authority_granted: false`,
  `plan_backend_port_prestart_gate_execution_allowed: false`,
  `plan_backend_port_prestart_gate_readiness_closure_allowed: false`,
  `plan_backend_port_prestart_gate_live_authority_granted: false`,
  `backend_port_prestart_check_status_matches_plan: true`,
  `safe_to_stop_listener_by_gate_matches_plan: true`,
  `live_authority_granted_matches_plan: true`,
  `backend_port_prestart_gate_execution_allowed_matches_plan: true`,
  `backend_port_prestart_gate_readiness_closure_allowed_matches_plan: true`, and
  `backend_port_prestart_gate_live_authority_granted_matches_plan: true`;
  it must also show
  `declared_safe_to_stop_listener_by_gate_matches_plan: true`,
  `declared_live_authority_granted_matches_plan: true`,
  `declared_backend_port_prestart_gate_execution_allowed_matches_plan: true`,
  `declared_backend_port_prestart_gate_readiness_closure_allowed_matches_plan: true`,
  and
  `declared_backend_port_prestart_gate_live_authority_granted_matches_plan: true`.
- Before any supervised fader/gain proof, the accepted-candidate prestart trace
	  must show `backend_port_prestart_summary_required: true`,
	  `backend_port_prestart_summary_type: no_wing_release_check`,
	  `backend_port_prestart_summary_status: passed`,
	  `backend_port_prestart_summary_connection_policy: no_hardware_no_network_no_live_writes`,
	  `backend_port_prestart_summary_read_only: true`,
	  `backend_port_prestart_summary_live_write: false`,
	  `backend_port_prestart_summary_contacts_backend: false`,
	  `backend_port_prestart_summary_contacts_wing: false`,
	  `backend_port_prestart_summary_mutates_console_state: false`,
	  `backend_port_prestart_summary_queries_console: false`,
	  `backend_port_prestart_summary_executes_tests: false`, and
	  `backend_port_prestart_check_status: passed`.
- Before any supervised fader/gain proof, accepted review source ids must show
  `accepted_candidate_prestart_contract_violations_by_source: {}` in the
  decision-review ledger/report and
  `readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source: {}`
  in the supervised proof plan summary. A non-empty map keeps
  `accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof`
  in force, must trigger
  `next_wing_proof_plan_steps_with_candidate_prestart_violations` if proof
  steps are still present, and grants no live authority.
- The recomputed
  `readonly_preflight_review_decision_expected_candidate_prestart_contract_violations_by_source`
  map must also stay `{}`. A non-empty expected map must trigger
  `next_wing_proof_plan_steps_with_expected_candidate_prestart_violations` if
  proof steps are still present.
- Accepted review source ids must also show
  `readonly_preflight_review_decision_candidate_artifact_root_contract_violations_by_source: {}`
  in the supervised proof plan summary. A non-empty map keeps
  `accepted_readonly_preflight_candidate_artifact_root_required_before_supervised_proof`
  in force, must trigger
  `next_wing_proof_plan_steps_with_candidate_artifact_root_violations` if proof
  steps are still present, and grants no live authority.
- The recomputed
  `readonly_preflight_review_decision_expected_candidate_artifact_root_contract_violations_by_source`
  map must also stay `{}`. A non-empty expected map must trigger
  `next_wing_proof_plan_steps_with_expected_candidate_artifact_root_violations`
  if proof steps are still present.
- Accepted review source ids must also expose an expected-artifact trace in the
  supervised proof plan summary. Missing or mismatched expected-artifact trace
  evidence keeps
  `accepted_readonly_preflight_candidate_expected_artifact_trace_required_before_supervised_proof`
  in force, must trigger
  `next_wing_proof_plan_steps_with_expected_artifact_trace_violations` if proof
  steps are still present, and grants no live authority.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `backend_port_prestart_summary_path_placeholder` pointing to the passed
  no-WING prestart `summary.json`, plus matching
  `preview_command_prestart_summary`,
  `preview_command_prestart_summary_matches_plan: true`, matching
  `validate_inputs_command_prestart_summary`,
  `validate_inputs_command_prestart_summary_matches_plan: true`, matching
  `capture_command_prestart_summary`, and
  `capture_command_prestart_summary_matches_plan: true`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `physical_preflight_requires_existing_backend_connection: true`.
- Before any supervised fader/gain proof, the no-WING handoff summary must also
  show the same `backend_port_prestart_check_required: true` and
  `backend_port_prestart_check_must_pass_before_starting_backend: true`.
- Before any supervised fader/gain proof, the no-WING handoff summary must also
  show `backend_port_prestart_gate_execution_allowed: false`,
  `backend_port_prestart_gate_readiness_closure_allowed: false`, and
  `backend_port_prestart_gate_live_authority_granted: false`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `execution_allowed_by_packet: false`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `live_authority_granted: false`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `full_autonomous_ready: false`.
- If handoff validation fails because execution, readiness, live-authority, or
  full-autonomous packet flags are `true`, the no-WING summary must preserve
  those unsafe `true` values so the failed handoff is auditable.
- If handoff validation fails because packet safety flags such as `live_write`,
  `mutates_console_state`, `queries_console`, or `executes_tests` are `true`,
  the no-WING summary must preserve those unsafe `true` values.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `next_manual_artifact_required: real wing_readonly_preflight.v1 summary.json`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `backend_port_prestart_check_operator_sequence_step_index: 0`,
  `backend_port_prestart_check_operator_sequence_step` including
  `prestart_backend_port_check` and `unexpected listeners`,
  `backend_port_prestart_check_must_be_first_operator_step: true`, and
  `backend_port_prestart_check_operator_sequence_step_is_first: true`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `backend_connection_confirmation_step_index: 3`,
  `expected_backend_connection_confirmation_step_index: 3`,
  `operator_sequence_has_backend_connection_confirmation: true`,
  `operator_sequence_backend_connection_confirmation_step_index: 3`,
  `operator_sequence_backend_connection_confirmation_before_capture: true`,
  `operator_sequence_backend_connection_confirmation_step` including
  `existing backend connection`, `intended WING Rack`, and
  `read-only inspection`, and
  `operator_sequence_validate_readonly_preflight_inputs_step_index: 5`,
  `expected_operator_sequence_validate_readonly_preflight_inputs_step_index: 5`,
  `operator_sequence_validate_readonly_preflight_inputs_step_index_matches_contract: true`,
  `operator_sequence_validate_readonly_preflight_inputs_step: Run validate_readonly_preflight_inputs before capture; it must validate real-run inputs and expected-plan bindings without opening transport.`,
  `expected_operator_sequence_validate_readonly_preflight_inputs_step: Run validate_readonly_preflight_inputs before capture; it must validate real-run inputs and expected-plan bindings without opening transport.`,
  `operator_sequence_validate_readonly_preflight_inputs_step_matches_contract: true`,
  `operator_sequence_capture_readonly_preflight_step_index: 6`,
  `expected_operator_sequence_capture_readonly_preflight_step_index: 6`,
  `operator_sequence_capture_readonly_preflight_step_index_matches_contract: true`,
  `operator_sequence_capture_readonly_preflight_step: Run capture_readonly_preflight.`,
  `expected_operator_sequence_capture_readonly_preflight_step: Run capture_readonly_preflight.`,
  and `operator_sequence_capture_readonly_preflight_step_matches_contract: true`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `supervised_prestart_gate` exactly matching
  `expected_supervised_prestart_gate` and
  `supervised_prestart_gate.grants_live_authority: false`.
- Before any supervised fader/gain proof, the structured
  `supervised_prestart_gate` must show
  `supervised_prestart_gate.operator_sequence_step_index: 11` and
  `supervised_prestart_gate.operator_sequence_step_must_be_final: true`.
- Before any supervised fader/gain proof, the structured
  `supervised_prestart_gate` must show
  `supervised_prestart_gate.requires_operator_decision_status:
  accepted_decision_ready` and `supervised_prestart_gate.unlocks_runbook:
  Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `mvp_blocker_anchor_matches_current_first_open: true`,
  `mvp_blocker_anchor_sha256_matches_current_first_open: true`,
  `mvp_blocker_anchor.packet_does_not_close_blocker: true`,
  `operator_sequence_has_mvp_blocker_anchor_digest_review: true`,
  `operator_sequence_has_supervised_prestart_gate: true`,
  `operator_sequence_supervised_prestart_gate_is_final: true`, and
  `supervised_prestart_gate.requires_accepted_candidate_prestart_trace: true`,
  plus `supervised_prestart_gate.requires_accepted_candidate_artifact_root_trace: true`,
  `supervised_prestart_gate.requires_accepted_candidate_request_sequence_trace: true`,
  and
  `supervised_prestart_gate.requires_accepted_candidate_expected_artifact_trace: true`.
  It must also expose top-level
  `supervised_prestart_gate_required: true`,
  `supervised_prestart_gate_operator_sequence_step_index: 11`,
  `supervised_prestart_gate_operator_sequence_step_must_be_final: true`,
  `supervised_prestart_gate_operator_sequence_step`,
  `supervised_prestart_gate_operator_sequence_step_matches_summary: true`,
  `supervised_prestart_gate_required_markers`,
  `expected_supervised_prestart_gate_required_markers`,
  `expected_required_supervised_prestart_gate_markers`,
  `required_supervised_prestart_gate_marker_count: 6`,
  `expected_required_supervised_prestart_gate_marker_count: 6`,
  `required_supervised_prestart_gate_marker_count_matches_contract: true`,
  `required_supervised_prestart_gate_marker_count_matches_supervised_gate: true`,
  `required_supervised_prestart_gate_markers_match_contract: true`,
  `required_supervised_prestart_gate_markers_match_supervised_gate: true`,
  `supervised_prestart_gate_required_marker_count: 6`,
  `expected_supervised_prestart_gate_required_marker_count: 6`,
  `supervised_prestart_gate_required_marker_count_matches_contract: true`,
  `supervised_prestart_gate_required_marker_count_matches_summary: true`,
  `supervised_prestart_gate_required_markers_match_contract: true`,
  `supervised_prestart_gate_required_markers_match_summary: true`,
  `supervised_prestart_gate_grants_live_authority: false`,
  `supervised_prestart_gate_requires_operator_decision_status: accepted_decision_ready`,
  `supervised_prestart_gate_unlocks_runbook: Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md`,
  `supervised_prestart_gate_requires_accepted_candidate_prestart_trace: true`,
  `supervised_prestart_gate_requires_accepted_candidate_artifact_root_trace: true`,
  `supervised_prestart_gate_requires_accepted_candidate_request_sequence_trace: true`,
  and
  `supervised_prestart_gate_requires_accepted_candidate_expected_artifact_trace: true`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `operator_sequence_count: 12`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  top-level `operator_sequence` and top-level `expected_operator_sequence`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `wing_readonly_operator_sequence_list_summary_surface_match: true` and
  `wing_readonly_operator_sequence_list_summary_surface_mismatches: []`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `wing_readonly_operator_sequence_list_markdown_surface_field_lines_rendered: true`
  and `wing_readonly_operator_sequence_list_markdown_surface_missing_field_lines: []`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `wing_readonly_operator_sequence_list_markdown_surface_required_field_lines`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `wing_readonly_operator_sequence_list_markdown_surface_required_field_lines_expected_value`
  and `wing_readonly_operator_sequence_list_markdown_surface_required_field_lines_actual_value`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `wing_readonly_operator_sequence_list_markdown_surface_required_field_lines_match_contract: true`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `expected_operator_sequence_count: 12` and
  `operator_sequence_count_matches_contract: true`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `operator_sequence_sha256`, `actual_operator_sequence_sha256`, and
  `operator_sequence_sha256_matches_actual: true`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `expected_operator_sequence_sha256`,
  `actual_expected_operator_sequence_sha256`,
  `operator_sequence_matches_contract: true`,
  `operator_sequence_sha256_matches_contract: true`,
  `declared_expected_operator_sequence_sha256_matches_contract: true`,
  `declared_operator_sequence_matches_contract_flag: true`,
  `declared_operator_sequence_matches_contract_flag_matches_actual: true`,
  `declared_operator_sequence_sha256_matches_contract_flag: true`, and
  `declared_operator_sequence_sha256_matches_contract_flag_matches_actual: true`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `declared_operator_sequence_count: 12`,
  `declared_expected_operator_sequence_count: 12`,
  `declared_operator_sequence_count_contract_flag: true`,
  `declared_operator_sequence_count_matches_actual: true`,
  `declared_operator_sequence_count_matches_contract: true`,
  `declared_expected_operator_sequence_count_matches_contract: true`, and
  `declared_operator_sequence_count_contract_flag_matches_declared_count: true`.
- Before any supervised fader/gain proof, the
  `mvp_blocker_anchor_digest_review_step_index` must be `9`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `expected_mvp_blocker_anchor_digest_review_step_index: 10`,
  `mvp_blocker_anchor_digest_review_step_index_matches_contract: true`, and
  `mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence: true`.
- Before any supervised fader/gain proof, the
  `operator_sequence_mvp_blocker_anchor_digest_review_step_index` must be `9`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10`
  and
  `operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract: true`.
- Before any supervised fader/gain proof, the
  `operator_sequence_mvp_blocker_anchor_digest_review_step` must include
  `mvp_blocker_anchor`, `mvp_blocker_anchor_sha256`, and
  `current first open blocker`.
- Before any supervised fader/gain proof, the
  `operator_sequence_supervised_prestart_gate_step_index` must be `10`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `expected_operator_sequence_supervised_prestart_gate_step_index: 11` and
  `operator_sequence_supervised_prestart_gate_step_index_matches_contract: true`.
- Before any supervised fader/gain proof, the final
  `operator_sequence_supervised_prestart_gate_step` must include
  `accepted_decision_ready`, `valid accepted-candidate prestart trace`,
  `valid accepted-candidate artifact-root trace`,
  `expected-artifact-bound accepted source id`, and
  `WING_SUPERVISED_FADER_GAIN_PROOF`.
- Before any supervised fader/gain proof,
  `required_supervised_prestart_gate_markers` must exactly list
  `accepted_decision_ready`, `valid accepted-candidate prestart trace`,
  `valid accepted-candidate artifact-root trace`,
  `expected-artifact-bound accepted source id`, and
  `WING_SUPERVISED_FADER_GAIN_PROOF`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `required_supervised_proof_plan_summary_fields`,
  `expected_required_supervised_proof_plan_summary_fields`, and
  `required_supervised_proof_plan_summary_fields_match_contract: true`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `required_supervised_proof_plan_summary_fields_include_invalid_records: true`.
- Before any supervised fader/gain proof,
  `required_supervised_proof_plan_summary_fields` must include
  `readonly_preflight_review_decision_expected_candidate_prestart_source_ids`,
  `readonly_preflight_review_decision_expected_candidate_prestart_contract_violations_by_source`,
  `readonly_preflight_review_decision_candidate_artifact_root_contract_valid`,
  `readonly_preflight_review_decision_candidate_artifact_roots`,
  `readonly_preflight_review_decision_candidate_artifact_root_contract_violations_by_source`,
  `readonly_preflight_review_decision_expected_candidate_artifact_roots`, and
  `readonly_preflight_review_decision_expected_candidate_artifact_root_contract_violations_by_source`,
  and `readonly_preflight_review_decision_invalid_records`.
- Before any supervised fader/gain proof, the operator-readable no-WING
  `summary.md` proof plan section must render those canonical helper fields.
- Before any supervised fader/gain proof, the no-WING `next_wing_proof_plan`
  summary must show `operator_markdown_summary_required: true`,
  `operator_markdown_summary_required_field_lines`,
  `operator_markdown_summary_required_field_lines_missing_members: []`,
  `operator_markdown_summary_required_field_lines_match_contract: true`,
  `operator_markdown_summary_field_lines_rendered: true`, and
  `operator_markdown_summary_missing_field_lines: []`.
- Before any supervised fader/gain proof, the no-WING `next_wing_proof_plan`
  summary must bind the operator markdown artifact with
  `operator_markdown_summary_artifact: summary.md`,
  `operator_markdown_summary_sha256`, and
  `operator_markdown_summary_sha256_available: true`.
- Before any supervised fader/gain proof, the no-WING `next_wing_proof_plan`
  summary must verify that binding with
  `operator_markdown_summary_actual_sha256` and
  `operator_markdown_summary_sha256_matches_actual: true`.
- Before any supervised fader/gain proof, the no-WING `summary.json` and
  operator-readable `summary.md` must expose the manifest summary surface with
  `artifact_manifest_summary_surface_required: true`,
  `artifact_manifest_summary_surface_missing_fields: []`,
  `artifact_manifest_summary_surface_required_field_lines`,
  `artifact_manifest_summary_surface_missing_field_lines: []`,
  `artifact_manifest_summary_surface_field_lines_rendered: true`, and
  `artifact_manifest_contract_id`, `artifact_manifest_contract_required: true`,
  `artifact_manifest_contract_fields`, `artifact_manifest_contract_sha256`,
  `artifact_manifest_contract_sha256_available: true`,
  `artifact_manifest_contract_sha256_matches_expected: true`,
  `artifact_manifest_contract_cross_surface_required: true`,
  `artifact_manifest_contract_cross_surface_required_surfaces`,
  `artifact_manifest_contract_cross_surface_required_fields`,
  `artifact_manifest_contract_cross_surface_expected_match: true`,
  `artifact_manifest_contract_cross_surface_readiness_closure_allowed: false`,
  `artifact_manifest_contract_cross_surface_live_authority_granted: false`, and
  `artifact_manifest_required_artifact_sha256_fields`.
- Before any supervised fader/gain proof, the no-WING run artifact bundle must
  write `artifact-manifest.json` with
  `artifact_manifest_contract_id`, `artifact_manifest_contract_required: true`,
  `artifact_manifest_contract_fields`, `artifact_manifest_contract_sha256`,
  `artifact_manifest_contract_sha256_available: true`,
  `artifact_manifest_contract_sha256_matches_expected: true`,
  `artifact_manifest_contract_cross_surface_required: true`,
  `artifact_manifest_contract_cross_surface_required_surfaces`,
  `artifact_manifest_contract_cross_surface_required_fields`,
  `artifact_manifest_contract_cross_surface_expected_match: true`,
  `artifact_manifest_contract_cross_surface_match: true`,
  `artifact_manifest_summary_json_contract_sha256_matches_manifest: true`,
  `artifact_manifest_summary_md_contract_sha256_matches_manifest: true`,
  `artifact_manifest_contract_cross_surface_mismatches: []`,
  `artifact_manifest_contract_cross_surface_readiness_closure_allowed: false`,
  `artifact_manifest_contract_cross_surface_live_authority_granted: false`,
  `artifact_manifest_path: artifact-manifest.json`,
  `artifact_manifest_artifact_count: 2`,
  `artifact_manifest_required_artifact_count: 2`,
  `artifact_manifest_artifact_count_matches_required: true`,
  `artifact_manifest_required_artifact_ids`, `artifact_manifest_artifact_ids`,
  `artifact_manifest_required_artifacts_present: true`,
  `artifact_manifest_required_artifact_paths`, `artifact_manifest_artifact_paths`,
  `artifact_manifest_required_artifact_paths_present: true`,
  `artifact_manifest_required_artifact_sha256_available: true`,
  `artifact_manifest_summary_json_sha256`,
  `artifact_manifest_summary_json_sha256_available: true`,
  `artifact_manifest_summary_md_sha256`,
  `artifact_manifest_summary_md_sha256_available: true`,
  `artifact_manifest_readiness_closure_allowed: false`, and
  `artifact_manifest_live_authority_granted: false`.
- Before opening the backend/WING session, the operator must run
  `PYTHONPATH=backend python3 tools/verify_no_wing_artifact_manifest.py <no_wing_artifact_dir>`
  against the same passed no-WING artifact directory and require
  `status: passed`, `failed_checks: []`,
  `artifact_manifest_self_consistency_match: true`,
  `artifact_manifest_contract_cross_surface_match: true`,
  `artifact_manifest_self_consistency_summary_surface_match: true`,
  `readiness_closure_allowed: false`, and `live_authority_granted: false`.
  The no-WING summary must also expose `artifact_manifest_verifier_required:
  true`, `artifact_manifest_verifier_command`,
  `artifact_manifest_verifier_status_required: passed`,
  `artifact_manifest_verifier_failed_checks_required: []`,
  `artifact_manifest_verifier_contacts_wing: false`,
  `artifact_manifest_verifier_mutates_console_state: false`,
  `artifact_manifest_verifier_queries_console: false`,
  `artifact_manifest_verifier_readiness_closure_allowed: false`, and
  `artifact_manifest_verifier_live_authority_granted: false`.
- Before any supervised fader/gain proof, the no-WING `next_wing_proof_plan`
  summary must expose `required_supervised_proof_plan_summary_fields` and
  `missing_required_supervised_proof_plan_summary_fields`.
- Before any supervised fader/gain proof, `next_wing_proof_plan_schema` must
  fail closed with `next_wing_proof_plan_required_summary_fields_missing` when
  any required summary field is absent.
- Если WING Rack не подключен, разрешена только работа, которая прямо готовит
  физический read-only/supervised proof и имеет тест.
- Методы коррекций Gain/Trim, Pan, EQ, Comp, FX, Balance и другие продолжай
  развивать в режиме `Цель`, но до закрытия WING MVP они должны оставаться
  offline/dry-run/proposal-only/read-only, если только они не являются уже
  разрешенным supervised fader/gain proof.
- Не считай guard доказательством live readiness: это только audit направления.

## 3. Найти нерешенные задачи

Ищи задачи из:

- TODO/FIXME;
- `.paperclip/reports/`;
- failing tests;
- незавершенных модулей;
- roadmap/memory;
- safety/readback/rollback/telemetry gaps.

## 4. Написать краткий план

Выбери 1-3 безопасные задачи на текущий запуск.

Приоритет:

1. safety;
2. WING read-only/readback/rollback/emergency-stop proof blockers;
3. supervised fader/gain proof blockers;
4. tests that prove a blocker is closed;
5. telemetry/logging needed for operator-visible proof;
6. gain staging method only when it is needed for supervised fader/gain proof.

Избегай больших рискованных переписываний.

## 5. Выполнить план

Выполни минимальные полезные изменения:

- внеси маленький scoped diff;
- добавь или обнови тесты, если меняешь поведение;
- запусти релевантные проверки;
- запусти `python3 tools/memory/memory_lint.py` и зафиксируй результат;
- перед финальным отчетом снова запусти MVP progress, direction guard и Project
  Director Agent:
  `python3 tools/codex_autoloop/update_mvp_progress.py`,
  `python3 tools/codex_autoloop/codex_direction_guard.py --format markdown --output .paperclip/reports/codex_direction_guard_final.md`,
  `python3 tools/codex_autoloop/project_director_agent.py --format markdown --output .paperclip/reports/project_director_final.md`;
- не останавливайся на середине;
- если безопасной кодовой задачи нет, улучши report/docs/tests только вокруг
  named WING proof blockers из раздела 2.1.

## 6. Написать отчет

Сохрани markdown-отчет в `.paperclip/reports/`.

Имя файла:

`automixer_codex_cycle_YYYYMMDDTHHMMSSZ.md`

Отчет должен содержать:

- статус agent workflow: в `## Scope` укажи ровно
  `Codex CLI was the only agent workflow used.`;
- статус external agent boundary: в `## Scope` укажи ровно
  `External Paperclip/PiperClip was not used as an agent runner.`;
- статус repo-local report path: в `## Scope` укажи метку
  `Repo-local report path:` и абсолютный путь
  `/Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main/.paperclip/reports`;
- что было найдено;
- краткий план выбранных безопасных задач;
- что было сделано;
- какие тесты запущены;
- результат memory lint в отдельном разделе `## Memory Lint Result` с результатом `python3 tools/memory/memory_lint.py`;
- результат direction guard: статус, score и краткое объяснение, почему задача
  двигала supervised WING Rack MVP;
- результат Project Director Agent: выбранный blocker, его status и почему
  задача была взята из first non-closed row; если action source был
  `Current First Command`, укажи baseline coverage verdict и
  `selected_action_sha256`; для WING handoff rail также укажи, что no-WING
  handoff summary содержит `project_director_selected_action_sha256` и
  `project_director_selected_action_sha256_matches_current_directive: true`;
- что осталось;
- какие риски есть;
- ссылки на измененные файлы;
- статус live safety: обязательно укажи, что live OSC/WING writes не выполнялись.

## 7. Три обязательных вопроса

Перед финальным разделом `## Объяснение простым языком` обязательно добавь
раздел:

`## Три обязательных вопроса`

После каждого цикла обязательно ответь в отчете:

1. Что нового проект научился делать?
2. Что сейчас является самым слабым местом?
3. Что даст наибольший прогресс в следующем цикле?

## 8. Объяснение простым языком

В самом конце отчета обязательно добавь финальный top-level раздел:

`## Объяснение простым языком`

Не добавляй после этого раздела другие top-level разделы. `### Общая
готовность направления` и readiness scores должны быть последним содержимым
внутри финального plain-language раздела.

Структура раздела:

### Что было сделано

Кратко и понятно опиши изменения. Не перечисляй файлы. Объясни смысл выполненной работы.

### Зачем это нужно проекту

Объясни практическую пользу для Automixer.

### В каком состоянии проект сейчас

Дай общую оценку:

- что уже работает;
- что работает частично;
- что еще не реализовано;
- какие есть риски.

Пиши человеческим языком.

### Какие проблемы были обнаружены

Опиши найденные ошибки, ограничения или спорные места. Если проблем нет, так и напиши.

### Что нужно сделать дальше

Не более 5 пунктов, по важности.

### Общая готовность направления

Пересчитай оценки после текущего цикла:

- Исследование: N%
- Архитектура: N%
- Реализация: N%
- Тестирование: N%
- Готовность к реальному концерту: N%

## 9. Завершение

После отчета выведи краткий итог в final response и заверши текущий запуск.
