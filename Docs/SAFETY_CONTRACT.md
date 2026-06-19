# Safety Contract

Status: hard constraints

## Live-sound invariants

- Never touch secrets, `.env`, tokens or API keys.
- Never raise faders above `0 dBFS` without explicit operator request.
- Preserve true-peak headroom before gain increases.
- Feedback safety wins over target loudness.
- Ayaic Mix Monolith level planes are guidance for balance and gain staging,
  not authority to bypass headroom, feedback or fader ceilings.
- Unknown and insufficient evidence are valid safe outcomes.

## Unattended automation invariants

During unattended development automation must never:

- send live OSC writes;
- control a real console;
- modify the user's WING scene;
- change real preamp gain, trim, faders, mute, routing, EQ, dynamics, FX,
  buses, DCA, matrices, scenes or presets;
- bypass write gates, allowlists, bounds, rate limits, stale-state protection,
  readback, rollback or audit logging;
- describe simulated validation as real hardware validation;
- allow a learned model or LLM to write directly to the console;
- weaken safety tests to pass a change.

## Promotion pipeline

Learning and optimization must follow:

logs and recordings -> data validation -> candidate policy/model/configuration
-> replay evaluation -> regression tests -> simulation -> supervised validation
-> controlled promotion

Unvalidated candidates cannot alter live concert behavior.

Recovery policy candidates remain offline artifacts until owner-approved and
explicitly bound to the fault-recovery gate; they cannot alter live concert
behavior.

## Safety kernel

The deterministic safety kernel is the non-bypassable authority for console
actions. Models, LLMs, classifiers, memory and proposal queues may recommend
or rank actions, but they cannot bypass deterministic safety checks.
