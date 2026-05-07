# Drum Phase Time Alignment Rule

## Task

Add the proposed source-grounded rule for phase/polarity and delay use on
multi-miked drums.

## Constraints

- Do not change live OSC behavior by default.
- Keep the rule advisory/shadow-first.
- Preserve fader/headroom safety.
- Use authoritative source metadata rather than unsourced folklore.

## Decision

Encode a new source-knowledge rule that logs phase-alignment candidates for
offline/shadow evaluation. The rule should prefer polarity checks and bounded
small-delay candidates, require mono/full-mix auditioning, and reject changes
that improve a solo drum while worsening bleed, cymbals, room tone, or mono
translation.

## Test Plan

- Validate source registry references.
- Verify the rule is retrievable by phase/drum/delay search.
- Run the project pytest command.
