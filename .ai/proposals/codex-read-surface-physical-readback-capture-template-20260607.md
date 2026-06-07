# Proposal: Read-Surface Physical Readback Capture Template

## 1. Short thesis

Add a fail-closed pending artifact template between the capture packet and
artifact review queue.

## 2. What I understood

Full console autonomy needs physical readback proof for every read surface, but
the repo must keep hardware-facing work behind operator action and no-WING gates.
The previous packet manifest shows what remains, yet it does not give the
operator a structured JSON shape to fill with real capture data.

## 3. Proposed solution

- Add `tools/build_read_surface_physical_readback_capture_template.py`.
- Build pending records from `remaining_capture_targets`.
- Keep placeholder values for `artifact_source_id`, `captured_at`, `captured_by`,
  `source`, `console_model`, `capture_scope`, and
  `physical_readback_payload`.
- Keep all execution, closure, live authority, and autonomous readiness flags
  false.
- Add a no-WING artifact gate that rejects unsafe flags and rejects already
  filled proof payloads under the pending-template schema.
- Insert the command between capture packet generation and artifact-review queue
  generation.

## 4. Likely files to touch

- `tools/build_read_surface_physical_readback_capture_template.py`
- `tools/no_wing_release_check.py`
- `tests/test_read_surface_physical_readback_capture_template.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/read-surface-physical-readback-capture-template-20260607.md`
- `Docs/adr/read-surface-physical-readback-capture-template-20260607.md`
- `Docs/reports/AUTOMIXER_READ_SURFACE_PHYSICAL_READBACK_CAPTURE_TEMPLATE_20260607.md`
- `Docs/INDEX.md`

## 5. Alternatives considered

- Put the pending artifact fields directly inside the capture packet.
- Wait until a real physical readback artifact exists.
- Let the review queue imply the needed shape.

## 6. Risks

The main risk is mistaking a pending template for physical proof. The validator
must require placeholders to remain placeholders and must reject candidate,
fixture-closure, live-authority, and full-autonomous claims.

## 7. Test plan

- py_compile for the new tool and no-WING runner.
- Focused builder tests.
- Focused no-WING artifact and command-order tests.
- Broader read-surface/no-WING pytest.
- no-WING quick run.
- Memory lint and diff check.
- Full backend pytest if no-WING release boundary changes remain broad.

## 8. Where the other agent may disagree

Kimi may prefer waiting for a real operator-captured artifact before adding
another generated file. The counterargument is that a safe pending template
shrinks operator ambiguity while still refusing to become evidence until the
real capture passes contract intake and manual review.
