# Independent Critique of Codex Proposal: M117

Codex read-only reviewer substitute; this is not a Kimi opinion.

## 1. Strongest part of the other proposal

The proposal preserves the narrow safety boundary: M116 may truthfully record
operator-attended read-only contact without implying live authority. It also
requires the extra capture observations to be labelled `operator-observed` and
`not independently machine-verified`, avoids a production authority redesign,
and anticipates the marker-only CP05 rebind.

## 2. Possible mistakes or blind spots

The existing M116 branch runs only while M116 is the latest ledger row. After
M117 appends a row, a durable test must find the unique M116 event anywhere in
history and pin its contact facts, zero writes/authority, evidence hashes and
provenance. The proposal also underlists STATE, append-only ledger, exact
blocker mirrors, heartbeat/report, M001 plan, progress report and project-layer
surfaces. The M116 plan still marks CP04-CP06 incomplete, and ROADMAP calls the
now fail-closed M113 identity gate a `fail-open`.

## 3. Where the solution is too complex, fragile, or costly

The proposal is under-specified rather than too complex. Name STATE as the
canonical blocker source and list every tested mirror. Do not rewrite every
historical mention or M116 ledger row 302; restrict changes to current sections,
present-tense contracts, the completed M116 plan and one appended M117 row.

## 4. What should be kept from the other proposal

Keep the truth-only scope, exact evidence-class qualifier, unchanged score and
authority, no backend/CoreAudio/OSC/WING contact, focused and full validation,
and separate marker-only CP05 rebind.

## 5. My revised recommendation

Append an offline M117 truth row, synchronize the complete current mirror set,
make M116 ledger coverage historical rather than latest-only, reject
unqualified current capture prose and absolute no-contact rules, close the
M116 plan status, correct the ROADMAP gate wording, then validate and rebind
CP05 without touching production policy.
