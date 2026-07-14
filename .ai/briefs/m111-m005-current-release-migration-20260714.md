# M111: truthful M005 current-release migration

## Owner request

The owner states that M005 is closed and authorizes M006. The physical WING is
powered and receiving signal, but the required 48x48/48 kHz CoreAudio device is
not currently visible to this Mac, so M110-HW01 and every live write remain
blocked before start.

## Confirmed starting point

- Historical M005 evidence is immutable:
  `artifacts/wing_timeout_disconnect_proof/20260711T131505Z_wing_m005_disconnect_timeout_stale_proof/summary.json`
  and its sibling `events.jsonl`.
- The old evidence proves physical baseline readback, operator-confirmed cable
  loss, unavailable stale readback, a safely blocked apply request, and
  unchanged physical reconnect readback.
- The current M006 packet verifier rejects that evidence with 43 failed checks:
  15 are legacy shape/sequence mismatches and 28 are current provenance checks.
- The historical run cannot truthfully acquire current-source prestart
  provenance, a proof-owned backend PID/nonce/runtime-identity event, an
  initial current-contract write-capable connection, or an owner console
  challenge after the fact.
- Current documents explicitly allow rebind or migration without repeating the
  cable pull unless the owner requests it. The runtime-identity-v2 ADR forbids
  relabelling the historical artifact itself as current provenance.

## Required result

Create a separate fail-closed migration candidate which combines, without
rewriting the historical artifact:

1. byte identities and independently extracted facts from the historical
   hardware evidence;
2. a clean-current-source compatibility replay through the canonical current
   apply path proving that stale/disconnected state cannot emit a console
   write;
3. a fresh owner console identity attestation for WING Rack firmware 3.0.5,
   including non-empty serial and explicit same-console confirmation;
4. exact current source/runtime/verifier identities; and
5. authority/contact/write denial flags.

The M006 packet and decision verifiers may accept either a native current M005
proof or a fully verified migration. They must expose which route was used and
must never report missing historical prestart/runtime facts as passed.

## Safety invariants

- Do not modify, copy over, or augment the historical `summary.json` or
  `events.jsonl`.
- No WING/backend/network/audio contact is needed to build or test the
  migration implementation.
- Compatibility replay is synthetic current-code evidence only; it cannot
  replace the historical physical facts.
- An owner attestation is not a hardware query and must be labelled as such.
- Missing or mismatched source identity, historical hash, replay evidence,
  owner attestation, or denial flag blocks migration.
- No live authority, readiness closure, direct write, gain-up, fader boost, or
  family widening follows from migration or M006 technical acceptance.
- EQ, dynamics, gate, pan, FX, mute, routing, master and scene authority remain
  blocked until their own M110 hardware evidence is accepted.

## Likely files

- `tools/migrate_m005_current_release_provenance.py`
- `tools/verify_m006_owner_readiness_packet.py`
- `tools/verify_m006_owner_readiness_decision.py`
- focused tests for those tools
- `Docs/adr/m111-m005-current-release-migration-20260714.md`
- M110/autopilot truth surfaces only after implementation and validation

## Acceptance evidence

- mutation tests for every bound hash/path/source/attestation field;
- a canonical golden legacy extractor test over copied fixture bytes;
- a current-path stale/disconnect replay proving zero console send and zero
  fader/gain writer calls;
- packet and decision verifier tests for native, migrated, missing, mutated and
  widened-scope cases;
- focused M005/M006/WING rail tests, full backend suite, frontend checks and the
  five-store no-WING gate;
- independent safety review before commit;
- current CP05 marker rebind after all tracked edits.

## Explicit blocker

Implementation can prepare the migration and produce a blocked template, but
technical migration cannot become accepted until a fresh owner console
identity attestation supplies the actual serial and same-console confirmation.
This blocker must remain visible and must not be guessed from IP, channel names
or historical state.
