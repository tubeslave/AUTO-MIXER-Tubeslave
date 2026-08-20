# Proposal — M124 pipeline separation and WING lab MCP

## 1. Short thesis

Add an explicit entrypoint catalog without moving compatibility scripts, and
replace the insecure legacy MCP with a fixed-command local stdio lab bridge.

## 2. What I understood

The test phase needs visibility into every accepted correction on WING Rack
without repeated confirmations. It must remain a temporary, opt-in lab surface,
not a production authority expansion.

## 3. Proposed solution

- Validate working and experimental entrypoints from a versioned JSON catalog.
- Record duplicate families and canonical replacements.
- Require `AUTOMIXER_WING_LAB_AUTOWRITE=1` plus one time/count-bounded arm.
- Compile only known WING OSC address families with bounded values.
- Read preimage, WAL, send, double-readback, retain rollback stack.
- Restore in reverse order on stop, mismatch, expiry or emergency.
- Expose only fixed MCP tools through a fixed Python bridge command.

## 4. Likely files to touch

`backend/pipeline_entrypoint_catalog.py`, `backend/wing_lab_autowrite.py`,
`tools/wing_lab_mcp_bridge.py`, `tools/mcp_director/`, tests and docs.

## 5. Alternatives considered

- Move legacy scripts: rejected because it would break paths and imports.
- Restore old MCP Director: rejected because it exposed commands, Git and tokens.
- Disable all safeguards globally: rejected because it would alter production.

## 6. Risks

OSC readback is unauthenticated UDP evidence. FX parameters remain model-specific,
so every generic FX value requires caller-declared bounds and fresh rollback.

## 7. Test plan

Catalog validation, mapping/bounds tests, fake-WING readback and rollback tests,
MCP smoke test, security quarantine test, no-WING release gate and full suite.

## 8. Where another reviewer may disagree

An independent reviewer may require the MCP to accept only proposal DTOs rather
than the narrow decision DTO. That can be added without broadening transport.
