# ADR: Fail closed at unauthenticated control and evidence boundaries

## 1. Context

Completed standard scan `b6b64252-e684-44a9-abb6-c7d4360fc105` reported 13
findings: one high, eight medium, and four low. The repository controls live
sound, is heavily dirty, and currently has no physical-console proof authority.

## 2. Options considered

1. Preserve existing behavior and add warnings.
2. Invent application nonces over unauthenticated UDP protocols.
3. Fail closed where identity cannot be proven, while retaining bounded offline
   and observational components.
4. Build a new authenticated gateway before fixing the findings.

## 3. Decision

Choose option 3. Source-tuple and loopback observations cannot satisfy physical
or application identity. Tokenless control-plane sessions are denied. Transport
lifecycle is requester-owned. Allocation limits apply before parsing and
accumulation. Offline audit fields describe simulation and failures literally.

## 4. Why this won

It closes the actual trust and resource boundaries with the smallest auditable
change and does not manufacture new live authority. It also remains consistent
with the current externally gated physical-proof milestone.

## 5. Rejected alternatives

Warnings leave exploitable paths open. Local nonces without an authenticated
peer can be forged or observed and do not prove mixer identity. A new gateway is
a separate architecture project and should not block fail-closed remediation.

## 6. Implementation plan

Patch network/ownership boundaries, resource/evidence boundaries, then service
hardening. Add regressions next to each root cause and preserve unrelated hunks.

## 7. Test plan

Focused negative and positive controls, owning module suites, no-WING quick,
standard full pytest, syntax/import checks, and `git diff --check`. No physical
network or audio-hardware interaction.

## 8. Risks and rollback

Tokenless source clients and unauthenticated mixer integrations will be denied.
Rollback is per exact remediation hunk, but only after reopening the associated
finding. A green software suite is not WING/dLive or acoustic validation.
