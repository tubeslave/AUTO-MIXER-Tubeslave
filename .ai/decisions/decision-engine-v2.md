# Decision Note: Decision Engine v2

Codex implements v2 as a disabled-by-default layer under `automixer/`.

The existing AutoFOH and mix-agent paths remain intact. v2 is enabled only by explicit CLI flags or direct module calls. The first version intentionally favors explainability, dry-run reporting, and testability over aggressive live automation.

Independent Kimi proposal/review was not executed in this worktree during this implementation pass; no second agent wrote code here.
