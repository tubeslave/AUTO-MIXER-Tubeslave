# Enable Full MERT

## Context

The operator requested full MERT during live WING Rack testing. The current
perceptual config requests `backend: mert`, but the backend environment lacks
`transformers`, so the evaluator falls back to the lightweight embedding backend.

## Goal

Install and declare the dependencies needed to load `m-a-p/MERT-v1-95M`, and
make the live configuration fail visibly instead of silently falling back when
MERT cannot be loaded.

## Safety Constraints

- This change must not send mixer commands by itself.
- MERT remains shadow evaluation only; it informs decisions/logs but does not
  bypass headroom, feedback, or channel safety gates.
- Dependency additions need a written reason and rollback path.

