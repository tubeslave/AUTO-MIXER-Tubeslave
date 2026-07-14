# Codex Critique of the Independent M112 Proposal

## 1. Strongest part

The proposal correctly distinguishes a validated product/runtime baseline from
the exact future tree identity and makes the two-commit CP05 sequence explicit.

## 2. Possible mistakes or blind spots

It treats the progress generator defect as secondary. Leaving it unfixed would
allow the next routine regeneration to restore false decision status and
requirement-count text, so M112 should close it now.

## 3. Where it is too complex, fragile, or costly

Adding new STATE fields for exact current marker hashes would create another
update cycle because the evidence hash changes with the marker content. Dynamic
marker verification is enough; no duplicate hashes should enter non-marker
truth files.

## 4. What should be kept

Keep the baseline-vs-current-tree distinction, exactly three capsule files,
unchanged launch-only provenance and strict zero-authority language.

## 5. Revised recommendation

Adopt the proposal with the generator fix included, avoid redundant marker
hashes in STATE, and require no tracked changes after marker-only commit.
