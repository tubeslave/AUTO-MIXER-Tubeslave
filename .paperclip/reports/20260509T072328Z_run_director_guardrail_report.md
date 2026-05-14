# Add Paperclip Connect run_director MCP tool

## Timestamp

2026-05-09T07:23:28Z

## Original GPT Prompt

```
Задача: добавить в Paperclip Connect новый tool `run_director`, который НЕ использует Codex как исполнителя.

Проект: `/Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main`

Цель: GPT Chat должен отправлять задание директору Paperclip, директор должен сам раздать подзадачи внутренним ролям, собрать результат, записать task/report и вернуть полный отчёт обратно в тот же GPT Chat.

Рабочая схема: GPT Chat -> Paperclip Connect -> MCP gateway -> run_director -> Paperclip Director -> локальные роли -> report -> ответ в GPT Chat.

Важно: Codex НЕ должен запускаться как исполнитель. Shell-команды для изменения кода запрещены. Git commit/checkout запрещены. Исходный код проекта не менять. Разрешены записи только в:
- `.paperclip/tasks`
- `.paperclip/reports`

Нужно сделать:
1. В `tools/mcp_remote_gateway/server.js` добавить MCP tool `run_director`.
2. Tool `run_director` должен принимать:
   - `title: string`
   - `prompt: string`
   - `mode: "analysis" | "planning" | "implementation_plan"`
   - `dryRun: boolean`
3. Независимо от входных данных:
   - `dryRun` всегда принудительно true
   - `dryRunEnforced` всегда true
   - `sourceCodeChangesAllowed` всегда false
4. `run_director` должен создать task JSON в `.paperclip/tasks`.
5. `run_director` должен создать Markdown report в `.paperclip/reports`.
6. Директор должен распределять задачу между внутренними ролями:
   - Director
   - DSP Agent
   - ML Agent
   - Refactor Agent
   - Analyzer Agent
   - Safety Agent
7. Внутренние роли не должны вызывать внешние LLM. Они должны генерировать структурированный план/анализ на основе prompt и текущего git status.
8. Report должен содержать:
   - title
   - timestamp
   - original GPT prompt
   - director decision
   - task breakdown
   - role assignments
   - risk assessment
   - execution plan
   - next steps
   - created task file path
   - created report file path
   - dryRunEnforced: true
   - sourceCodeChangesAllowed: false
   - git branch
   - git status
9. Ответ MCP tool должен вернуть в GPT Chat НЕ только путь к файлу, но и сам краткий отчёт:
   - ok: true
   - taskFile
   - reportFile
   - summary
   - directorReport
10. Добавить `run_director` в `tools/list`.
11. Добавить тест в `test.sh`, который вызывает `run_director` и проверяет:
   - task создан
   - report создан
   - dryRunEnforced true
   - sourceCodeChangesAllowed false
   - ответ содержит directorReport
12. Прогнать:
   - `npm install`
   - `node --check server.js`
   - `./test.sh`
13. В финальном ответе вывести:
   - изменённые файлы
   - результат тестов
   - пример команды для GPT Chat:

Use Paperclip Connect run_director. Title: Проверка директора Paperclip. Prompt: Проверь связь GPT Chat -> Paperclip Connect -> MCP gateway -> Paperclip Director. Раздай задачу внутренним ролям и верни отчёт в этот же чат. Исходный код не изменять. Mode: analysis. dryRun: true.

Главное: `run_director` должен возвращать отчёт прямо в GPT Chat, а не только создавать файл.
```

## Director Decision

Blocked under the active write policy. The requested feature requires editing:

- `tools/mcp_remote_gateway/server.js`
- `tools/mcp_remote_gateway/test.sh`

The same request also states that source code must not be changed and that writes are allowed only under:

- `.paperclip/tasks`
- `.paperclip/reports`

Therefore I created only this Paperclip task/report record and did not modify source code, did not run Codex as an executor, did not run git commit/checkout, and did not run `npm install`.

## Task Breakdown

1. Define `runDirectorInput` and `runDirectorOutput` schemas in `tools/mcp_remote_gateway/server.js`.
2. Implement a deterministic in-process director function that:
   - forces `dryRun: true`;
   - forces `dryRunEnforced: true`;
   - forces `sourceCodeChangesAllowed: false`;
   - reads git branch/status through the existing allowlisted git helpers;
   - creates a task JSON in `.paperclip/tasks`;
   - creates a report Markdown file in `.paperclip/reports`;
   - returns `summary` and `directorReport` directly in MCP structuredContent.
3. Add the MCP tool registration for `run_director`.
4. Ensure `tools/list` includes `run_director`.
5. Extend `tools/mcp_remote_gateway/test.sh` to call `run_director` and assert task/report/directorReport policy behavior.
6. Run checks only after source-edit permission is explicitly granted.

## Role Assignments

| Role | Assignment |
| --- | --- |
| Director | Enforce policy, coordinate roles, compile final task/report, return `directorReport` to GPT Chat. |
| DSP Agent | Produce deterministic DSP/audio implications from the prompt and git status only. |
| ML Agent | Produce deterministic ML/data/evaluation implications from the prompt and git status only. |
| Refactor Agent | Produce implementation boundaries for gateway/test edits without changing runtime behavior outside the new tool. |
| Analyzer Agent | Capture branch/status and identify existing gateway patterns for schemas, reports, and output fields. |
| Safety Agent | Verify no Codex execution, no external LLM calls, no source writes at runtime, dry-run enforcement, and artifact-only writes. |

## Risk Assessment

- Main blocker: adding the tool cannot be completed without source edits.
- `npm install` is not compatible with the current "writes only under .paperclip" rule because it may write `node_modules` and/or lockfiles.
- Existing `run_agent` in `tools/mcp_director/server.js` can invoke Codex when `dryRun` is false, so `run_director` should not delegate to it unless the path is provably dry-run-only. Safer implementation is a new deterministic in-process director inside the remote gateway.
- The current working tree is already dirty with many modified/untracked files, so any implementation should stay tightly scoped and avoid git checkout/reset/commit.

## Execution Plan

After explicit approval to edit `tools/mcp_remote_gateway/server.js` and `tools/mcp_remote_gateway/test.sh`:

1. Add constants for the six internal roles.
2. Add `runDirectorInput` with `title`, `prompt`, `mode`, and `dryRun`.
3. Add `runDirectorOutput` with `ok`, `taskFile`, `reportFile`, `summary`, `directorReport`, `dryRun`, `dryRunEnforced`, and `sourceCodeChangesAllowed`.
4. Add deterministic role-analysis helpers that use only input prompt/mode and git snapshot text.
5. Write task JSON via existing `writeJsonFile(tasksDir, ...)`.
6. Write Markdown via existing `writeMarkdownReport(...)`.
7. Register `run_director` before `list_artifacts` or near `run_agent`.
8. Update test assertions to include `run_director` in both raw and SDK `tools/list`.
9. Add a test call that verifies the task file, report file, policy fields, and returned `directorReport`.

## Next Steps

1. Relax the write constraint to allow edits to `tools/mcp_remote_gateway/server.js` and `tools/mcp_remote_gateway/test.sh`.
2. Decide whether `npm install` is allowed to write outside `.paperclip`; if not, skip it and use the already present `node_modules`.
3. Re-run:
   - `node --check server.js`
   - `./test.sh`
4. Use the GPT Chat command after implementation is actually present in `tools/list`.

## Created Task File Path

`.paperclip/tasks/task_20260509T072328Z_run_director_request.json`

## Created Report File Path

`.paperclip/reports/20260509T072328Z_run_director_guardrail_report.md`

## Policy

- dryRunEnforced: true
- sourceCodeChangesAllowed: false
- dryRun: true
- codexAsExecutorAllowed: false
- externalLlmAllowedForInternalRoles: false
- writesAllowed: `.paperclip/tasks`, `.paperclip/reports`

## Git Branch

```
feature/mcp-director
```

## Git Status

```
## feature/mcp-director
 M Docs/ARRANGEMENT_AUTOMATION.md
 M README.md
 M automixer/production_mix_v1/__main__.py
 M automixer/production_mix_v1/arrangement_input_levels.py
 M automixer/production_mix_v1/ayaic_offline_pipeline.py
 M backend/arrangement_automation/__init__.py
 M backend/auto_soundcheck_engine.py
 M backend/handlers/gain_staging_handlers.py
 M backend/handlers/mixer_handlers.py
 M backend/handlers/soundcheck_handlers.py
 M backend/lufs_gain_staging.py
 M backend/server.py
 M backend/signal_metrics.py
 M backend/wing_client.py
 M config/automixer.yaml
 M config/default_config.json
 M docs/ARCHITECTURE.md
 M frontend/package-lock.json
 M frontend/src/components/GainStagingTab.css
 M frontend/src/components/GainStagingTab.js
 M frontend/src/services/websocket.js
 M tests/test_arrangement_automation.py
 M tests/test_ayaic_offline_pipeline.py
 M tests/test_server.py
 M tests/test_wing_client.py
?? .ai/personas.md
?? .ai/rules.md
?? .ai/state.json
?? .ai/system_prompt.md
?? .ai/tasks.json
?? .paperclip/reports/20260508T143224Z_dsp_dsp-agent-repository-task.md
?? .paperclip/reports/20260508T144803Z_dsp_eq_decision_logic_dry_run_report.md
?? .paperclip/reports/20260508T152429Z_dsp_report.md
?? .paperclip/reports/20260508T170329Z_b391e36c_git_report.md
?? .paperclip/reports/20260508T170329Z_ef099ea9_smoke_report.md
?? .paperclip/reports/20260508T181626Z_8d80ccbb_git_report.md
?? .paperclip/reports/20260508T181626Z_f331a5fe_smoke_report.md
?? .paperclip/reports/20260508T182713Z_913b8059_git_report.md
?? .paperclip/reports/20260508T182713Z_cb52424b_smoke_report.md
?? .paperclip/reports/20260508T182839Z_11de4262_git_report.md
?? .paperclip/reports/20260508T182839Z_7a257858_smoke_report.md
?? .paperclip/reports/20260508T182908Z_005186c5_smoke_report.md
?? .paperclip/reports/20260508T182908Z_13c3e87f_git_report.md
?? .paperclip/reports/20260508T183327Z_a39133fb_smoke_report.md
?? .paperclip/reports/20260508T183327Z_b5af4e49_git_report.md
?? .paperclip/reports/20260508T184353Z_5cc1f9e5_git_report.md
?? .paperclip/reports/20260508T184353Z_b98c197d_smoke_report.md
?? .paperclip/reports/20260508T184418Z_1293c6c6_git_report.md
?? .paperclip/reports/20260508T184418Z_397dcb8c_smoke_report.md
?? .paperclip/reports/20260508T191812Z_0fc3d406_git_report.md
?? .paperclip/reports/20260508T191812Z_3e50abad_smoke_report.md
?? .paperclip/reports/20260508T192611Z_9dd326ca_smoke_report.md
?? .paperclip/reports/20260508T192611Z_a4405df3_git_report.md
?? .paperclip/reports/20260509T070231Z_a17daf95_director_report.md
?? .paperclip/tasks/task_20260508T143224Z_d3a621be_dsp-agent-repository-task.json
?? .paperclip/tasks/task_20260508T152429Z_dsp.json
?? .paperclip/tasks/task_20260508T170328Z_57ed2fbb_smoke.json
?? .paperclip/tasks/task_20260508T170328Z_b75cec3c_automixer.json
?? .paperclip/tasks/task_20260508T181626Z_27d30e9a_automixer.json
?? .paperclip/tasks/task_20260508T181626Z_c4900bdb_smoke.json
?? .paperclip/tasks/task_20260508T182713Z_1fdcdc84_smoke.json
?? .paperclip/tasks/task_20260508T182713Z_b6e4aab0_automixer.json
?? .paperclip/tasks/task_20260508T182839Z_464bd183_smoke.json
?? .paperclip/tasks/task_20260508T182839Z_88ae3f41_automixer.json
?? .paperclip/tasks/task_20260508T182908Z_da020252_automixer.json
?? .paperclip/tasks/task_20260508T182908Z_f139169a_smoke.json
?? .paperclip/tasks/task_20260508T183327Z_14ad4ca5_smoke.json
?? .paperclip/tasks/task_20260508T183327Z_755a9eec_automixer.json
?? .paperclip/tasks/task_20260508T184353Z_377db649_smoke.json
?? .paperclip/tasks/task_20260508T184353Z_de0ef95c_automixer.json
?? .paperclip/tasks/task_20260508T184418Z_4d728682_automixer.json
?? .paperclip/tasks/task_20260508T184418Z_7c0747ba_smoke.json
?? .paperclip/tasks/task_20260508T190421Z_066e06f4_automixer.json
?? .paperclip/tasks/task_20260508T191811Z_32edb25a_smoke.json
?? .paperclip/tasks/task_20260508T191811Z_eb6a8319_automixer.json
?? .paperclip/tasks/task_20260508T192611Z_216aec3d_smoke.json
?? .paperclip/tasks/task_20260508T192611Z_c23ec2ba_automixer.json
?? .paperclip/tasks/task_20260509T064644Z_9dc527f3_automixer.json
?? .paperclip/tasks/task_20260509T070231Z_eb0d6654_director.json
?? AGENTS.md
?? Docs/CONTEXTUAL_COMPRESSION.md
?? Docs/GLOBAL_PHASE_ALIGNMENT.md
?? Docs/MUSICAL_OUTPUT_BALANCE.md
?? Docs/MUSICAL_PANNING.md
?? Docs/console_write_safety_policy.md
?? Docs/current_architecture_map.md
?? Docs/runtime_ownership_matrix.md
?? ai/
?? automixer/production_mix_v1/contextual_compression.py
?? automixer/production_mix_v1/musical_output_balance.py
?? automixer/production_mix_v1/musical_panning.py
?? automixer/production_mix_v1/musical_panorama_width.py
?? automixer/production_mix_v1/snare_pair_coherence.py
?? automixer/production_mix_v1/snare_top_bottom_balance.py
?? backend/arrangement_automation/live_input_trim_controller.py
?? backend/arrangement_automation/wing_meter_reader.py
?? backend/autogain_contract.py
?? backend/live_apply.py
?? frontend/src/TUB-61_OPERATOR_SUPERVISION_SPEC.md
?? frontend/src/components/AutoGainSupervisionCard.js
?? frontend/src/components/autoGainSupervision.js
?? frontend/src/components/autoGainSupervision.test.js
?? offline_test_input/
?? offline_test_output/
?? project_docs/
?? scripts/
?? tests/test_auto_soundcheck_engine_safety.py
?? tests/test_autogain_contract.py
?? tests/test_gain_staging_handlers.py
?? tests/test_live_apply_contract.py
?? tests/test_signal_metrics.py
?? tests/test_soundcheck_handlers.py
?? tools/mcp_remote_gateway/
?? tools/mcp_remote_gateway_old/
?? tools/telegram_codex_bridge.py
```

## GPT Chat Command After Implementation

```
Use Paperclip Connect run_director. Title: Проверка директора Paperclip. Prompt: Проверь связь GPT Chat -> Paperclip Connect -> MCP gateway -> Paperclip Director. Раздай задачу внутренним ролям и верни отчёт в этот же чат. Исходный код не изменять. Mode: analysis. dryRun: true.
```
