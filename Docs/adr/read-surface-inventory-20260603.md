# read-surface-inventory-20260603

## 1. Context

The long-term goal requires the Automixer to read all available console
parameters and understand channels, groups, matrices, buses, FX, scenes, and
presets before autonomous control is credible. Existing client methods and
Product Layer snapshots provide partial read coverage, but the project lacked a
single audited read-surface contract.

## 2. Options considered

- Leave read coverage implicit in mixer clients and handlers.
- Combine read coverage with the write-surface inventory.
- Add a separate read-only console surface inventory.

## 3. Decision

Add `backend/read_surface_inventory.py` as a declarative read-only inventory.
It records each console read surface, current status, entrypoints, read fields,
blockers, and next action. It is connected to the project-goal capability map
for `console_read_parameters`.

## 4. Why this won

Read completeness and write safety are different proof problems. A separate
inventory makes it clear that reading is inventoried but still incomplete, while
preserving the current fail-closed live-write boundary.

## 5. Rejected alternatives

Implicit read coverage is too easy to overstate. A dynamic live discovery pass
would require mixer transport and is not appropriate as a safe architecture
increment.

## 6. Implementation plan

1. Add read-surface rows for raw state, channels, processing, buses/DCA/main,
   matrices, FX/inserts, routing/sends, snapshots/scenes, presets, and dLive.
2. Mark partial/missing coverage explicitly.
3. Update `console_read_parameters` in the project-goal matrix to reference the
   inventory.
4. Expose the inventory inside the processing pipeline snapshot for Product
   Layer visibility.
5. Add tests proving the inventory is read-only and transport-free.

## 7. Test plan

- `PYTHONPATH=backend python -m pytest tests/test_read_surface_inventory.py -q`
- Existing capability/registry/de-esser contract tests.

## 8. Risks and rollback

The main risk is stale read metadata. Tests keep the surface list unique and
fail if the module starts importing transport or write methods. Rollback is a
clean removal of the module, tests, and docs plus reverting the project-goal
support path update.
