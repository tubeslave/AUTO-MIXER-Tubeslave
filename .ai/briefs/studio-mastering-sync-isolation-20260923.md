# STUDIO mastering integration and live-boundary audit

## Request
Wire the accepted mastering meter into the user's STUDIO pipeline and verify that live development has not replaced or damaged the studio work.

## Inspection
- Workbench baseline: `audio-workbench-mcp-v0.1` at `c66a2d418ac4aeb704de72b861c0b708fed07cc1`, still in draft PR #99.
- Its last Tests run `35901285869` and Stem Offline Test `35901285979` both passed when inspected.
- Master contains the meter and AutoMaster integration from PRs #112/#113 at `a5b472fcbcf5290510667a304e9da3774fee54b2`; integration Tests `35913849473` and Stem Offline Test `35913849318` passed.
- Workbench did NOT yet contain `backend/studio_mastering_metrics.py`. Its AutoMaster blob `0ab9f2fc825164fa0f3a682bf0947e6f12fb205e` was exactly the pre-integration master version.
- `backend/lufs_gain_staging.py` is identical in both branches (blob `d9321408b7cde5b5763d0cec4c3073dbd13ad55a`).
- Workbench's Automixer bridge calls `validate_or_apply(..., apply=False)`; its Director explicitly excludes editing, pitch/timing and mastering loudness maximization from live control.

## Bounded change
Copy only the exact accepted meter, AutoMaster and two regression-test blobs from master. Do not merge master into Workbench. Add subprocess regressions that block network/OSC calls before imports, render built-in/fallback/safety-only paths, and assert unchanged inputs, preserved stereo ratio, full length and true-peak safety. Enable existing full CI for PRs targeting the Workbench branch.

The `audio_workbench/`, `mix_agent/`, source audio, mix recipes, cleanup decisions, WING/live-control implementation and shared gain-staging DSP are not edited. Preservation is code/tree scope, not a claim that every audible outcome has been approved.

## Acceptance
Status at implementation: validation_pending. Require the focused tests and full Python 3.10/3.11/3.12 suite on this actual Workbench-based candidate. Source-branch green CI alone is insufficient. Do not promote any user audio baseline. Record final CI evidence in the PR discussion/state before merge.

## Limitations
This wires the backend AutoMaster path; Workbench's separate MasteringDirector and its limiter budgets remain intact, not silently replaced. General loudness compliance certification and artistic quality are not established by these tests. Ptitsa CLEAN remains pending human A/B. Local repository clone was unavailable because the execution container could not resolve github.com; source reads and validation use the authorized GitHub connector and repository CI instead.
