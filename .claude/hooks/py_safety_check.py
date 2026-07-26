#!/usr/bin/env python3
"""Safety-линтер для AUTO-MIXER: PostToolUse hook + sweep-режим для /verify.

Hook-режим (без аргументов): читает JSON события Claude Code со stdin,
проверяет отредактированный .py файл. Находки уходят в stderr с exit 2 —
Claude видит их как обратную связь сразу после правки.

Sweep-режим: py_safety_check.py --sweep [file ...] — проверяет указанные
файлы, либо все изменённые/новые .py по git. Используется командой /verify.

Проверки выведены из раздела «Зафиксированные правила» CLAUDE.md:
  - синтаксис (ruff при наличии, иначе py_compile)
  - fader ceiling: литерал > 0 dBFS в set_fader/set_channel_fader
  - json.dumps в файле с numpy без convert_numpy_types
  - вызов gcc_phat: напоминание о порядке аргументов (повторявшийся баг)
  - вызов load_snap*: загрузка снапшота перезаписывает состояние пульта
  - похожие на секреты строки (ключи/токены в литералах)
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

FADER_CALL = re.compile(
    r"\b(?:set_fader|set_channel_fader)\s*\([^,()]+,\s*\+?(\d+(?:\.\d+)?)"
)
GCC_CALL = re.compile(r"(?<!def )\b_?gcc_phat\s*\(")
LOAD_SNAP_CALL = re.compile(r"(?<!def )\bload_snap(?:_by_index)?\s*\(")
SECRET = re.compile(
    r"sk-[A-Za-z0-9]{16,}"
    r"|(?:api_key|apikey|token|secret|password)\s*=\s*[\"'][A-Za-z0-9_\-]{12,}[\"']",
    re.IGNORECASE,
)
SECRET_PLACEHOLDER = re.compile(r"YOUR_|XXX|EXAMPLE|PLACEHOLDER|CHANGE_?ME", re.IGNORECASE)


def _matching_lines(text, pattern):
    """Номера строк и содержимое строк файла, где сработал паттерн."""
    hits = []
    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if pattern.search(line):
            hits.append((lineno, stripped))
    return hits


def check_syntax(path):
    """ruff (только F/E9 — реальные ошибки, не стиль), иначе py_compile."""
    findings = []
    if shutil.which("ruff"):
        proc = subprocess.run(
            ["ruff", "check", "--select", "F,E9", "--quiet", str(path)],
            capture_output=True, text=True,
        )
        out = (proc.stdout or "").strip()
        if proc.returncode != 0 and out:
            findings.extend(f"[ruff] {line}" for line in out.splitlines()[:10])
    else:
        proc = subprocess.run(
            [sys.executable, "-m", "py_compile", str(path)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            err = (proc.stderr or "").strip().splitlines()
            findings.append(f"[syntax] {err[-1] if err else 'py_compile failed'}")
    return findings


def check_content(path, text, scope_text=None):
    """Доменные проверки. scope_text — только добавленный текст правки:
    если задан, проверка срабатывает лишь когда правка сама вносит паттерн
    (защита от повторного шума на старом коде при каждом редактировании)."""
    findings = []
    rel = path
    try:
        rel = path.relative_to(REPO_ROOT)
    except ValueError:
        pass

    def scoped(pattern):
        if scope_text is not None and not pattern.search(scope_text):
            return []
        return _matching_lines(text, pattern)

    for lineno, line in scoped(FADER_CALL):
        m = FADER_CALL.search(line)
        if m and float(m.group(1)) > 0.0:
            findings.append(
                f"{rel}:{lineno}: [fader-ceiling] литерал > 0 dBFS в set_fader — "
                f"запрещено без явного запроса оператора: {line}"
            )

    for lineno, line in scoped(GCC_CALL):
        findings.append(
            f"{rel}:{lineno}: [gcc-phat] проверь порядок аргументов по сигнатуре "
            f"вызываемой функции — инверсия (reference, target) = фаза корректируется "
            f"в обратную сторону (повторявшийся баг): {line}"
        )

    for lineno, line in scoped(LOAD_SNAP_CALL):
        findings.append(
            f"{rel}:{lineno}: [snap-safety] load_snap перезаписывает настройки пульта — "
            f"для поиска по имени использовать только кэш имён (find_snap_by_name): {line}"
        )

    if "json.dumps" in text and "convert_numpy_types" not in text and re.search(
        r"^(?:import numpy|from numpy)", text, re.MULTILINE
    ):
        for lineno, line in _matching_lines(text, re.compile(r"json\.dumps\s*\(")):
            findings.append(
                f"{rel}:{lineno}: [numpy-json] json.dumps в файле с numpy без "
                f"convert_numpy_types — np.float64 не сериализуется: {line}"
            )

    for lineno, line in scoped(SECRET):
        if not SECRET_PLACEHOLDER.search(line):
            findings.append(
                f"{rel}:{lineno}: [secret] строка похожа на ключ/токен — "
                f"секреты только через env/конфиг вне git"
            )

    return findings


def check_file(path, scope_text=None):
    if path.suffix != ".py" or not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return check_syntax(path) + check_content(path, text, scope_text)


def added_text_from_event(event):
    """Текст, который правка добавила (new_string/content/edits[])."""
    ti = event.get("tool_input") or {}
    parts = []
    for key in ("new_string", "content"):
        if isinstance(ti.get(key), str):
            parts.append(ti[key])
    for edit in ti.get("edits") or []:
        if isinstance(edit, dict) and isinstance(edit.get("new_string"), str):
            parts.append(edit["new_string"])
    return "\n".join(parts)


def run_hook():
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0
    file_path = (event.get("tool_input") or {}).get("file_path")
    if not file_path:
        return 0
    findings = check_file(Path(file_path), scope_text=added_text_from_event(event))
    if findings:
        print("[py_safety_check] Проверь находки:", file=sys.stderr)
        for f in findings[:20]:
            print(f"  {f}", file=sys.stderr)
        return 2
    return 0


def run_sweep(args):
    if args:
        files = [Path(a) for a in args]
    else:
        proc = subprocess.run(
            ["git", "diff", "--name-only", "HEAD", "--", "*.py"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "*.py"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        names = set(proc.stdout.split()) | set(untracked.stdout.split())
        files = [REPO_ROOT / n for n in sorted(names)]

    all_findings = []
    for path in files:
        all_findings.extend(check_file(path))

    if all_findings:
        print(f"[py_safety_check] Находок: {len(all_findings)}")
        for f in all_findings:
            print(f"  {f}")
        return 1
    print("[py_safety_check] OK — находок нет"
          + ("" if files else " (нет изменённых .py файлов)"))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--sweep":
        sys.exit(run_sweep(sys.argv[2:]))
    sys.exit(run_hook())
