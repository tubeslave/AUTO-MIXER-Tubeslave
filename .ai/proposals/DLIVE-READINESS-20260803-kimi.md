# Kimi proposal/critique — DLIVE-READINESS-20260803

## 1. Strongest part of the Codex proposal

Reachable wrong bytes are worse than no bytes. The fail-closed bounded rewrite,
explicit preamp-socket mapping and byte-vector/running-status/stale-cache tests
match the safety problem well.

## 2. Possible mistakes or blind spots

- Pin the official v2.0 source and auditable byte vectors; tests derived only by
  the implementer prove consistency, not independent correctness.
- Fresh GET attribution on an unsolicited MIDI stream needs an explicit design,
  not just a timestamp check.
- Do not expand into discovery changes unless the brief requires them.

## 3. Complexity, fragility or cost

- Put a deny-by-default capability gate in place before widening the protocol
  rewrite.
- Keep the first supported slice to fader, mute and name.
- Validate and report TLS inputs in this cycle; do not treat unproved TLS
  negotiation as ready.
- Record Mixing Station as a possible interim attended path rather than
  dismissing it.

## 4. What to keep

- unsupported families return `False` without bytes;
- Surface/MixRack endpoint and base-channel validation;
- explicit input-to-preamp mapping and 48V denied by default;
- official golden vectors and running-status parser tests;
- no WING milestone or dLive live-authority change.

## 5. Revised recommendation

Deliver two bounded layers: a safety gate/no-socket preflight first, then the
documented fader/mute/name protocol core against pinned official vectors. Leave
pan, dynamics, gate, delay, polarity and all unproved TLS/hardware behavior
blocked pending attended evidence.

Kimi caveat: the final critique was based primarily on the brief/proposal and
did not independently verify every code-level claim.
