# Pipeline entrypoints

The machine-readable catalog is `configs/pipelines/catalog.json`.

- `working` contains the canonical runtime, the attended WING runner and the
  supported offline render entrypoints.
- `experimental` contains retained compatibility and research scripts. Their
  existing paths are preserved, but they are not canonical release entrypoints.
- `duplicate_families` records which experiments overlap a supported pipeline.

This separation is additive: imports, command paths and the production runtime
remain unchanged. Promotion from `experimental` requires an ADR, focused tests,
and the appropriate software or attended hardware evidence.

The processing-method profiles in `backend/processing_registry.py` remain the
authority for method-level status. This catalog covers executable entrypoints,
which the method registry does not otherwise make explicit.
