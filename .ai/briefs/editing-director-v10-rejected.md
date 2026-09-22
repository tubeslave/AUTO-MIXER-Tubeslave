# Editing Cleanup v0.9/v0.10 — REJECTED

Human listening result: the cleanup created more audible defects than it removed.

Do not use these approaches in production:
- floating/hysteresis gates for de-bleed;
- offline region/clip-gain masks that are effectively gates;
- aggressive spectral masks driven only by local level;
- fixed noise subtraction when it audibly damages musical timbre.

All v0.9/v0.10 cleanup renders are development artifacts only. The production pipeline rolls back to the pre-cleanup raw/phase-aligned sources until a model-based cleanup stage passes listening validation.
