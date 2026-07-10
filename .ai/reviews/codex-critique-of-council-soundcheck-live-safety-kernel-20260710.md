# Codex critique of council fallback proposal

## 1. Strongest part

The proposal correctly identifies capability ownership, rather than more DSP
heuristics, as the release blocker. Runtime readiness must govern writes.

## 2. Possible mistakes or blind spots

Creating a new broker immediately would duplicate the existing manual gate and
increase live risk before behavior is proven. The current gate already contains
channel/range validation, cooldown, audit and proof-run integration.

## 3. Where it is too complex or costly

A full transport-capability refactor in one change would touch many controllers
and make regression attribution difficult in a dirty live checkout.

## 4. What should be kept

Structural proposal-only enforcement, remote authentication, observed-state
separation, system stop, CAS rollback, matrix-derived allowlist and disabled
means blocked.

## 5. Revised recommendation

First make the existing supervised gate the sole executable WING authority and
prove it. Then extract that proven behavior into `ConsoleWriteBroker` as a
separate checkpoint without changing semantics.
