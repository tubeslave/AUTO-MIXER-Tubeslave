# Kimi Proposal: System Inspector Phase 0 and Snapshot Replay

## 1. Short thesis

Первый срез Inspector должен быть чистой функцией над неизменяемыми входными
байтами: snapshot bundle + profile → findings + deterministic JSON. Начать с
одного доказательного route detector и закрепить read-only границу AST-тестом.

## 2. What I understood

- Phase 0 не имеет live-контакта по определению.
- `backend/wing_client.py` смешивает запросы, `send()` и десятки `set_*`,
  поэтому Inspector не должен его импортировать.
- Dante в Phase 0 — импортированный файл, а не Dante Controller API.
- `UNKNOWN` означает отсутствие достаточного evidence, а не OK и не
  наблюдаемый отказ.
- `config/show_profile.yaml`, `backend/system_inspector/` и
  `tools/automixer.py` пока отсутствуют.

## 3. Proposed solution

- Новый пакет с frozen dataclasses, строгой versioned schema, safe YAML
  profile loader, normalization, одним playback routing detector и JSON CLI.
- Сортировать коллекции и исключить неявное текущее время из канонического
  отчёта; два одинаковых входа должны давать одинаковый SHA-256.
- Каждый check декларирует required evidence; при его отсутствии выдаёт
  `UNKNOWN` и `missing_evidence`.
- Не вводить скалярный risk score в первой вертикали; показывать counts и
  coverage.
- Доказывать dependency boundary через AST: запрет транспортных импортов,
  `.send()` и `set_*`.

## 4. Likely files to touch

- `backend/system_inspector/**`
- `config/show_profile.yaml`
- `tests/fixtures/system_inspector/**`
- `tests/test_system_inspector_*.py`
- ADR и Council decision

## 5. Alternatives considered

- Generic graph immediately: слишком рано для одного detector.
- Reuse `WingClient`: нарушает dependency boundary.
- Numeric readiness in v1: создаёт ложную точность при частичном evidence.
- Runtime guard as the only protection: weaker than removing write
  dependencies and proving that statically.

## 6. Risks

- Fixture schema may drift from future real exports.
- Token-based static scans can flag their own test strings; use AST.
- Dict/set ordering can silently break deterministic output.
- A new show profile must stay separate from existing role override semantics.

## 7. Test plan

- Healthy replay is byte-for-byte deterministic.
- Duplicate-right emits exactly one evidence-backed finding.
- Missing route leg is `UNKNOWN`.
- Oversized, invalid, duplicate-ID and unsafe YAML inputs fail closed.
- AST scan proves the package has no live/write imports or calls.
- Input hashes remain unchanged before and after replay.

## 8. Where the other agent may disagree

Codex may retain the explicitly requested runtime guard/security log, top-level
CLI wrapper and readiness score. If retained, they need independent tests and
must not weaken the structural boundary or overstate incomplete coverage.

