# AUTO-MIXER Post-Cleanup Health Baseline

Date: 2026-05-18
Scope: local repository health after generated-artifact cleanup, cache cleanup,
and quarantine decision. No live WING/OSC actions were performed.

## Verdict

The project is locally usable after cleanup.

- Backend test baseline: green.
- Frontend dependency install: green.
- Frontend production build: green.
- Frontend dependency security audit: red.
- Git working tree before this report: clean.
- Branch state before this report: `main...origin/main [ahead 66, behind 26]`.
- Live-console readiness: unchanged; no hardware proof was run in this step.

## Verified Commands

| Area | Command | Result |
| --- | --- | --- |
| Backend full suite | `PYTHONPATH=backend python -m pytest tests/ -q` | `742 passed, 6 skipped, 2 warnings in 5.11s` |
| Frontend dependencies | `npm --prefix frontend ci` | Passed; installed 1520 packages |
| Frontend build | `npm --prefix frontend run build` | Passed; production bundle compiled |
| Frontend audit | `npm --prefix frontend audit --audit-level=high --omit=dev` | Failed; 40 vulnerabilities |
| Git hygiene | `git status --short --branch --untracked-files=all` | Clean tree; branch diverged from origin |

Frontend build output:

- JavaScript bundle gzip size: `74.41 kB`.
- CSS bundle gzip size: `8.44 kB`.
- Build warning under Node `v24.15.0`: `DEP0176 fs.F_OK is deprecated`.

## Known Risks

### 1. Frontend audit is red

`npm audit --audit-level=high --omit=dev` reports:

- 40 total vulnerabilities;
- 23 high severity vulnerabilities;
- several issues are chained through the legacy `react-scripts` stack.

Important vulnerable dependency families include:

- `react-scripts` transitive chain;
- `nth-check`;
- `postcss`;
- `serialize-javascript`;
- `webpack-dev-server`;
- `lodash`;
- `node-forge`;
- `minimatch`;
- `picomatch`;
- `rollup`;
- `jsonpath`.

Do not run `npm audit fix --force` as a blind cleanup step. The audit output
shows that the force path can install `react-scripts@0.0.0`, which is a breaking
change and would likely damage the frontend toolchain.

### 2. Branch divergence must be handled deliberately

Local `main` is far ahead of and behind `origin/main`. Pushing or merging without
a separate sync plan can mix the cleanup/product-layer work with older local
history and remote changes.

Recommended next step before push or PR:

- create a dedicated integration branch from the current local state; or
- create a clean branch from `origin/main` and cherry-pick the verified cleanup
  commits intentionally.

### 3. Live-console readiness did not change

This health baseline proves local code health only. It does not prove physical
WING readiness. Real-console readiness still requires the supervised path:

- explicit operator approval;
- one parameter;
- readback;
- rollback;
- cooldown;
- emergency-stop proof;
- preserved telemetry artifacts.

## Current Cleanup State

Completed:

- Generated dependency folders were removed and later restored where needed:
  `frontend/node_modules/` and `frontend/build/` now exist locally and are
  ignored by git.
- Python caches, pytest caches, and `.DS_Store` files were cleaned once.
- `.gitignore` no longer hides source files under `automixer/logs/`.
- Orphan `automixer/logs/` files were quarantined by decision record instead of
  restored as broken code.
- Product Layer proposal workflow has focused tests and server integration tests.

Not completed:

- Frontend dependency modernization.
- Branch synchronization with `origin/main`.
- Phase 2 backend root de-noising.
- Physical WING supervised proof.

## Recommended Next Work

1. Decide the branch synchronization strategy before any push.
2. Start Phase 2 backend root de-noising: move experimental entrypoints and
   scripts into a documented archive or `tools/` structure without changing
   runtime behavior.
3. Treat frontend dependency modernization as a separate task, because the audit
   cannot be fixed safely with automatic force updates.
