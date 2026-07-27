# System Inspector Threat Model

Status: Phase 0 snapshot replay

## Security objective

Inspector may read validated evidence and write local reports/security events.
It must not mutate WING, Dante, macOS network configuration, Ableton,
CoreAudio devices or Automixer live authority.

## Protected assets

- physical mixer and PA state;
- Dante subscriptions and clocking;
- macOS routes, adapters and permissions;
- show profile and known-good baseline;
- operator trust in findings and readiness;
- existing live safety gates and evidence journals;
- privacy of device names, addresses and local configuration.

## Trust boundaries

1. Untrusted profile/snapshot files → bounded parser.
2. Imported evidence → normalized graph.
3. Graph/findings → readiness and reports.
4. Future live collectors → OS/network/device APIs.
5. Inspector → Automixer promotion gate.

Only boundaries 1–3 exist in Phase 0.

## Threats and controls

| Threat | Impact | Phase 0 controls | Residual risk |
|---|---|---|---|
| Malicious YAML object construction | Local code execution | `yaml.safe_load`, type validation, regression test | Parser/library defects |
| Oversized input | Memory/disk exhaustion | 1 MiB per file, 4 MiB bundle | Many repeated invocations |
| Symlink input/output | Read/write outside intended path | Reject bundle/input symlinks and report/security-log symlinks | Parent-directory substitution outside current single process |
| Duplicate/spoofed stable IDs | Wrong topology joins | Unique-ID validation and explicit sources | A valid but dishonest import |
| Missing evidence shown as OK | False readiness | `UNKNOWN`, missing-evidence list, visible coverage | Operator may ignore warnings |
| Imported Dante record treated as live health | False audio assurance | Evidence ladder and limitation on every route item | Stale import |
| Risk score false precision | Unsafe confidence | Integer penalties, minimum coverage, confidence and unknowns | Weight policy remains provisional |
| Inspector imports write transport | Accidental device mutation | AST dependency test, no `WingClient`/socket/subprocess/audio imports | Future code must keep the rail |
| Write attempt through future code | Device mutation | Compile-time read-only invariant, no disable flag, runtime exception and redacted audit | A completely separate process bypassing Inspector |
| Secrets in logs | Credential disclosure | Security event logs only redacted action class; no payload/address/value | Other application logs are out of scope |
| Raw device string in HTML | XSS | No HTML/UI in Phase 0; future renderer must escape by default | Future implementation |
| Baseline auto-update after drift | Poisoned known-good state | Phase 0 baseline is read-only; inputs hashes unchanged in tests | Create/version workflow not yet implemented |
| Active discovery harms show network | Availability issue | No live/network collector imports | Later checkpoint must add rate limits and opt-out |

## Abuse cases

### Attempt to disable read-only

`INSPECTOR_READ_ONLY=false` fails startup. There is no `--write`, `--apply`,
`--fix`, `--no-read-only` or equivalent argument.

### Attempt to invoke a write

Future code must call the final guard before any mutation. `prohibit_write()`
logs a redacted `inspector_write_blocked` event and raises before the action.
The package currently has no transport capable of carrying the request.

### Stale or partial snapshots

The report retains captured timestamps and source-specific freshness. Missing
route evidence produces `UNKNOWN`; imported topology never becomes
`AUDIO_SIGNAL_CONFIRMED`.

## Manual security checkpoints

Stop and produce an action/risk/permission/rollback/read-only proof packet
before:

- real WING connection;
- active network discovery;
- Dante Controller configuration read;
- Dante Managed API or other licensed SDK;
- macOS elevated permission/helper;
- CoreAudio capture;
- network configuration change;
- any write client or OSC value-bearing command.

## Security validation

Current executable rails:

- read-only cannot be disabled;
- write attempt raises and creates redacted audit evidence;
- AST scan rejects live/write dependencies;
- unsafe YAML fails closed;
- duplicate IDs fail closed;
- replay inputs remain byte-identical.

