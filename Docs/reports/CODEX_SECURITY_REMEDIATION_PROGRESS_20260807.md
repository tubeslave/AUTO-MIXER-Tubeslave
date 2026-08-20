# Codex Security Remediation Progress — 2026-08-07

Source scan: `2c00402a-b02e-4b80-a68f-c9c079eb70c4`

Source revision: `f8a33c44f0144bbc62d6cadc135493ed10cd496e`

This ledger tracks code remediation in the current collaborative checkout. A
row marked `fixed` means the finding's ordered local verification passed; it
does not close the sealed scan record and does not grant live-console or
concert readiness. A fresh Codex Security scan is still required to confirm
the remaining queue.

## Verified findings

### `occ_063bdb67b407d6b54b57a7a9` — high — fixed

- Finding: tokenless WebSocket mode exposed cached mixer state.
- Boundary: raw cached mixer state now requires an authenticated WebSocket
  boundary across direct `get_state`, registration, and mixer-connection
  broadcasts.
- Preserved behavior: authenticated clients still receive the existing state
  payload; explicit tokenless diagnostics remain available.
- Files: `backend/server.py`, `tests/test_server.py`.
- Validation:
  - `python -m py_compile backend/server.py tests/test_server.py` — passed.
  - focused raw-state and registration regressions — `5 passed`.
  - auth/get-state slice — `42 passed, 375 deselected`.
  - `PYTHONPATH=backend python -m pytest tests/test_server.py -q` —
    `417 passed`.
  - `git diff --check -- backend/server.py tests/test_server.py` — passed.
- Remaining uncertainty: the original Security workbench Apply record remains
  failed because its base-revision patch overlapped pre-existing local changes;
  the current checkout behavior was verified independently through the real
  server boundary. A fresh scan must reclassify it.

### `occ_06ccc21df1e6769cf3324810` — high — no_change

- Finding: candidate filenames could escape the sandbox render directory.
- Current classification: the current checkout already contained a bounded
  path resolver before this cycle; no additional source change was needed.
- Boundary: absolute paths, `.`/`..` components, traversal fallbacks and
  symlink targets outside the resolved render root are rejected before
  `write_audio`.
- Preserved behavior: ordinary initial and candidate WAV filenames still
  render below the configured peak ceiling.
- Files inspected: `ai_mixing_pipeline/sandbox_renderer/renderer.py`,
  `tests/test_sandbox_renderer.py`.
- Validation:
  - `python -m py_compile ...` — passed.
  - traversal/symlink/legitimate focused slice — `6 passed, 2 deselected`.
  - `PYTHONPATH=. python -m pytest tests/test_sandbox_renderer.py -q` —
    `8 passed`.
  - `git diff --check -- ...` — passed.
- Remaining uncertainty: a fresh Security scan must reclassify the sealed
  base-revision finding against the current working-tree snapshot.

### `occ_39f791a9315e63b1ec04af1a` — high — fixed

- Finding: a fader-only promotion result widened the static WING supervised
  write scope to include unproven gain/preamp/trim writes.
- Boundary: the transition gate now intersects static policy with the exact
  `action_family` of every currently proved promotion target. Unknown or
  missing families fail closed.
- Preserved behavior: fader-only evidence permits only fader; combined fader
  and gain evidence still permits the existing bounded fader+gain scope.
- Files: `backend/autonomy_mode_transition_gate.py`,
  `tests/test_autonomy_mode_transition_gate.py`.
- Validation:
  - pre-fix reproducer returned `['fader', 'gain']` and
    `['preamp', 'trim', 'fader']` for fader-only evidence.
  - post-fix reproducer returns only `['fader']`/`['fader']`; unknown family
    blocks runtime entry.
  - transition-gate tests — `8 passed`.
  - operator policy, promotion readiness and evidence overlay — `33 passed`.
  - syntax and `git diff --check` — passed.
- Remaining uncertainty: no console was contacted; a fresh Security scan must
  reclassify the sealed finding against the current checkout.

### `occ_f42a3c13180143db3d66d86f` — high — fixed

- Finding: a fetched training manifest could redirect trained-model output to
  an arbitrary process-writable filesystem path.
- Boundary: remote manifest target fields can no longer override the locally
  configured `output_path`; other supported manifest tuning fields remain
  usable.
- Preserved behavior: the legitimate remote `n_epochs` override still reaches
  the trainer while the trusted local model path is retained.
- Files: `backend/ml/agent_training_service.py`,
  `tests/test_ml/test_agent_training_archive_loaders.py`.
- Validation:
  - pre-fix `_run_once` reproducer selected an arbitrary absolute path supplied
    by the manifest.
  - post-fix regression retained `models/channel_classifier.pt`, applied the
    legitimate `n_epochs=1`, and created no outside file.
  - owning archive/training service tests — `6 passed`.
  - full `tests/test_ml/` — `186 passed`.
  - syntax and `git diff --check` — passed.
- Remaining uncertainty: a fresh Security scan must reclassify the sealed
  finding against the current checkout.

### `occ_9fb8af1f23ca5de8a1ed3acc` — high — fixed

- Finding: authenticated WebSocket input could direct the backend to open a
  dLive TCP/TLS connection to an arbitrary host and port.
- Boundary: the direct WebSocket handler no longer consumes destination or TLS
  identity fields. The server independently permits only the exact dLive
  endpoint in `config/automixer.yaml`, requires an RFC1918 IPv4 unicast
  address, and validates the official endpoint/TLS port before constructing a
  client or disconnecting the current mixer. Auto-connect no longer probes a
  client-supplied `preferred_ip`.
- Preserved behavior: the configured MixRack endpoint and bounded MIDI base
  channel/user profile still flow through the existing dLive client. The
  packaged offline transport lock remains the first gate.
- Files: `backend/handlers/connection_handlers.py`, `backend/server.py`,
  `tests/test_dlive_connection_pipeline.py`, `tests/test_server.py`.
- Validation:
  - focused destination/connection regressions — `6 passed, 416 deselected`.
  - dLive client, connection pipeline and packaged-offline tests — `52 passed`.
  - full `tests/test_server.py` — `419 passed`.
  - syntax and `git diff --check` — passed.
  - negative regression proved the metadata-service address is rejected before
    `DLiveClient` construction and before the existing mixer is disconnected.
- Remaining uncertainty: validation was offline and contacted no console or
  network endpoint; a fresh Security scan must reclassify the sealed finding.

### `occ_99afa4a911984ffef9e61985` — high — fixed

- Finding: `ActionPlanExecutor` dispatched gain, EQ, compressor and pan writes
  without a final operation-specific live-write authority decision.
- Boundary: Decision Engine v2 is now proposal-only. Even when a caller asks
  for `dry_run=False`, Safety Gate is evaluated in dry-run mode, every allowed
  action is returned as a recommendation with
  `live_write_authority_unavailable`, and no mixer write method exists in the
  executor path. Live writes remain owned by canonical supervised transaction
  paths with their separate authority/readback/rollback contracts.
- Preserved behavior: ordinary dry-run planning continues to return the gated
  recommendations and `no_action` results.
- Files: `automixer/executor/plan_executor.py`, `tests/test_executor_v2.py`.
- Validation:
  - pre-fix real-boundary reproducer called `set_fader(1, -12.5)` and recorded
    one sent action;
  - post-fix same-boundary reproducer recorded zero sends and zero mixer calls;
  - focused executor tests — `2 passed`;
  - Decision Engine v2 / Safety Gate / executor regression slice —
    `23 passed`;
  - syntax, direct-sink scan, and scoped `git diff --check` — passed;
  - Ruff — unknown because the executable is not installed.
- Remaining uncertainty: the blocked full Deep Scan needs a managed read-only
  filesystem profile before it can reclassify the whole current checkout. No
  network, Ableton instance, console, or acoustic system was contacted.

### `occ_1be8a1e96551795319d39b5e` — high — no_change after preceding fix

- Finding: `ActionPlanExecutor` treated `None` and other non-`False` mixer
  results as successful writes.
- Current classification: the immediately preceding executor remediation
  removed every mixer dispatch helper and every `result is not False` success
  conversion from this component. The cited source-to-sink path no longer
  exists, so a second source patch would be speculative.
- Preserved behavior: proposal/dry-run recommendations remain covered by the
  same positive control.
- Validation:
  - static sink/pattern scan found no `_dispatch`, `result is not False`, or
    mixer setter in `automixer/executor/plan_executor.py`;
  - the write-capable four-family negative regression records zero calls;
  - the same Decision Engine v2 / Safety Gate / executor slice passed:
    `23 passed`.
- Remaining uncertainty: a fresh full scan is still required to reclassify the
  sealed occurrence against the current checkout.

### `occ_4fddb1bed449a80a8751308f` — high — fixed

- Finding: ambient speech could directly change non-WING faders without
  per-command operator authorization.
- Boundary: the STT callback now supplies server-owned `ambient_voice=True`
  provenance, and every ambient fader request is blocked before mixer access.

### `occ_ffecc071628771bc0bf3d359` — high — fixed

- Finding: ambient speech could directly change non-WING input gain without
  per-command operator authorization.
- Boundary: ambient gain requests hit the same immutable provenance gate and
  cannot call `set_channel_gain`.

### `occ_ee83a52876a8ee8a487d42aa` — high — fixed

- Finding: ambient speech could directly mute or unmute non-WING channels.
- Boundary: ambient mute requests are proposal-only and cannot call
  `set_channel_mute`.

### `occ_962cf4b0e4d4399f9cd66bb7` — high — fixed

- Finding: ambient volume-down could write a fader directly and could derive a
  target from a default zero-dB baseline when state was unavailable.
- Boundary: the ambient-source gate runs before every fader query, cached
  fallback, target calculation, and mixer write.

### `occ_f5033008c6c2959a4b3b57df` — high — fixed

- Finding: ambient volume-up could raise a non-WING fader directly and could
  derive a target from a default zero-dB baseline.
- Boundary: the same pre-read gate prevents both the direct raise and the
  fallback calculation.

Shared ambient-voice remediation evidence:

- The command payload cannot override provenance; a forged
  `ambient_voice:false` field remained blocked.
- Existing explicit internal non-ambient mute/pan behavior and the WING direct
  voice quarantine remain intact.
- Files: `backend/server.py`, `tests/test_server.py`.
- Pre-fix real scheduler reproducer called `set_channel_fader(1, -6.0)` and
  broadcast `voice_command_executed`; the post-fix reproducer made zero mixer
  calls and returned `ambient_voice_write_requires_operator_authorization`.
- Focused boundary/compatibility slice — `21 passed`.
- Full `tests/test_server.py` — `431 passed`.
- Adjacent voice runtime/handler/Mixing Station slice — `35 passed`.
- Syntax and scoped `git diff --check` — passed.
- No microphone, network endpoint, Ableton instance, mixer, or acoustic system
  was contacted. A new full Security scan remains required for sealed-record
  reclassification.

## Security checkpoint scan

A fresh repository checkpoint scan was completed and sealed after the first
five cycles:

- Scan ID: `59cb954a-42ab-4eae-b6d0-eba34941710c`.
- Snapshot digest:
  `codex-security-snapshot/v1:sha256:75a0eb05468fc72afa3c618137c6ae3d359d83ea6a09d3ac79dbcf8960a07e62`.
- Result: `12` reportable findings — `1 high`, `10 medium`, `1 low`.
- Coverage: partial first-party checkpoint, not a replacement full Deep Scan.
- The five exact remediation sinks above were confirmed closed. Adjacent
  sibling paths below were then remediated in the live collaborative checkout.
- No application, external network, console, Ableton, acoustic, or physical
  validation was performed by the scan.

## Verified checkpoint findings

### Transport and tokenless boundaries

- `occ_5f800a3ec6dc933092e73ed6` — high — fixed locally: every WING transport
  lifecycle operation now requires authentication; tokenless
  `observation_only` can no longer replace the active transport. Focused tests:
  `11 passed`; full `tests/test_server.py`: `421 passed` at that checkpoint.
- `occ_f8d2cec05c2a019c1c03f6f6` — medium — fixed locally: Mixing Station
  connection input ignores caller destinations, uses only the configured
  loopback endpoint, validates it before disconnect/client construction, and
  disables HTTP redirects. Relevant tests: `11 passed`; full server suite:
  `423 passed` at that checkpoint.
- `occ_b63fbd61f8b61efc6e4896f6` — medium — fixed locally: mixer-derived
  dashboard, normalized, and dLive snapshot routes require authentication.
- `occ_62bbfcc01268b9cf69de1fe9` — medium — fixed locally: physical parameter
  readback requires authentication.
- `occ_eae00dc67e975c956d12d217` — medium — fixed locally: mixer/subnet scans
  require authentication. The shared tokenless-boundary regression slice was
  `12 passed`; full server suite was `424 passed`.

### Training and filesystem boundaries

- `occ_d247a7bfd4119bff919c2cdd` — medium — fixed locally: WebSocket
  `start_training` no longer accepts a manifest URL; training enablement and
  manifest location are operator-owned local configuration.
- `occ_0cabbebb4bd42b869209b142` — medium — fixed locally: dataset retrieval is
  HTTPS-only, host-allowlisted, DNS/IP-class checked, credential-free,
  port-bounded, and redirect-disabled.
- `occ_e67b282cd8c48f05342c8e91` — medium — fixed locally: remote manifests can
  no longer enable training or replace local resource budgets. Training
  security slice: `9 passed`; full `tests/test_ml/`: `189 passed`.
- `occ_a7f5b8e6f2183a85aa0e182d` — medium — fixed locally: the memory writer
  ignores caller roots and appends beneath the server-owned repository root
  using containment and no-follow protections. New traversal/symlink tests:
  `2 passed`; writer slice: `3 passed`; related product-state suite:
  `232 passed`.

### Ableton state provenance and bounded UDP handling

- `occ_51ba6acace418bc876a198c7` — medium — fixed locally: setters and the
  compatibility shim return strict nested transport results and never publish
  requested values as observed state.
- `occ_671fe89aec0dbd136c67ca60` — medium — fixed locally: loopback is bound
  narrowly, sender IP and the documented AbletonOSC source port are checked,
  the handshake address is exact, and only fresh expected replies are
  promoted.
- `occ_24c0c237f37c6da548486ee1` — low — fixed locally: unknown/unsolicited OSC
  addresses never enter state or callbacks; supported track identifiers and
  pending replies are bounded.
- New Ableton security tests: `9 passed`; Ableton plus auto-fader regression
  slice: `35 passed`.

## Aggregate verification and open proof gaps

- Standard full-suite attempt reached `6365 passed, 3 skipped` before an
  unrelated existing test blocked at
  `tests/test_supervised_proof_capability.py:64`. That helper writes a packet
  synchronously to a pipe before starting its reader; the process was stopped
  after four minutes and about 6.6 GiB resident memory. This is not reported as
  a green full-suite result.
- A follow-on run after that file reached `715 passed` and then correctly
  failed because this remediation session's required
  `ops/autopilot/WRITER_LEASE/owner.json` still existed. The lease must be
  released before the final rail recheck.
- `git diff --check` and final focused tests remain the final local handoff
  gates.
- The original sealed Deep Scan contained `143` findings. This checkpoint was
  explicitly partial, so its locally fixed 12-item queue does not prove that
  every deferred Deep Scan finding is closed. The next safe cycle is a new
  current-checkout Security scan followed by remaining findings in severity
  order.
- No live mixer, Ableton instance, network endpoint, or acoustic system was
  contacted. These changes provide software/offline evidence only and do not
  grant concert readiness.

## Post-remediation Security checkpoint

A second targeted Security scan was completed after the 12-item checkpoint
queue was fixed:

- Scan ID: `e01f678c-e332-4688-b22c-2a9a9ab1d8e9`.
- Snapshot digest:
  `codex-security-snapshot/v1:sha256:c11deea606c2fb7750d77fd147f660f407036c56d0f4dd21df4d14786363ba20`.
- Result: `0` reportable residual findings across five reviewed remediation
  surfaces; the canonical artifacts are sealed and `report.md` is nonempty.
- All 12 occurrences from scan `59cb954a-42ab-4eae-b6d0-eba34941710c`
  were independently revalidated as closed.
- The review found five adjacent issues before sealing; all were fixed and
  revalidated in the same current-working-tree checkpoint:
  - Medium: decompressed ZIP/TAR/GZIP datasets could exceed the compressed
    download budget. Archive metadata and streamed output are now bounded by
    trusted local `max_dataset_bytes`.
  - Medium: study-resource HTTP redirects could escape the approved domain.
    Study fetches now use the same HTTPS/domain/DNS/IP-class policy and reject
    all redirects.
  - Medium sibling: authenticated WebSocket input could still select an
    arbitrary subnet. The handler now ignores the ingress subnet and derives
    full-scan subnets only from active local interfaces.
  - Low: a racing invalid UDP packet could abort the Ableton handshake. Invalid
    peers, malformed OSC, and wrong addresses are now ignored until the
    monotonic deadline or a valid `/live/test` reply.
  - Low: tokenless emergency-stop responses exposed operational and recovered
    readiness detail. Both emergency routes now return a constant minimal
    acknowledgement while preserving detailed internal audit; readiness
    broadcasts omit the recovered payload when authentication is disabled.
- New negative tests cover the three archive types, redirect rejection,
  caller-subnet removal, Ableton handshake races, and both tokenless emergency
  routes.
- Targeted combined regression gate: `641 passed`.
- Coverage remains explicitly `partial`: this does not replace a new full Deep
  Scan of the deferred 28,742-file inventory or prove physical readiness.

### `occ_7d2fac1df56024f42ef4b06c` — high — no_change after preceding fix

- Finding: a tokenless local WebSocket client could start and monopolize the
  shared audio-capture preview and read live channel meters.
- Current classification: the preceding tokenless-WebSocket remediation made
  the source-runtime boundary default-deny. Only static runtime identity and
  emergency stop remain unauthenticated; all audio-device, capture-status,
  channel-meter, and meter-subscription handlers require authentication before
  dispatch. A second source patch would duplicate the existing boundary.
- Concrete closure: real `handle_client_message` calls for
  `start_audio_capture_preview`, `get_audio_capture_status`,
  `get_channel_meters`, `subscribe_meters`, and `unsubscribe_meters` returned
  `authenticated_control_session_required` before any handler, capture sink,
  ownership assignment, or meter subscription. The registered-handler harness
  recorded zero capture calls and zero unauthenticated subscribers.
- Preserved behavior: with the configured authentication boundary enabled, the
  same message families reached their existing handlers; a mocked physical
  capture produced the established `audio_capture_status` and `channel_meters`
  payloads without changing their API.
- Files inspected: `backend/server.py`, `backend/handlers/audio_handlers.py`,
  `backend/handlers/agent_handlers.py`,
  `backend/handlers/meter_stream_handlers.py`, `tests/test_server.py`.
- Validation:
  - exact dispatch-boundary malicious/legitimate harness — passed for all five
    cited message types;
  - registered-handler boundary harness — passed with zero physical capture;
  - focused auth/default-deny tests — `3 passed, 428 deselected`;
  - full `tests/test_server.py` — `431 passed`;
  - audio, meter, agent-handler and capture owning modules — `35 passed`;
  - Python compilation of all four affected production modules — passed.
- Remaining uncertainty: no microphone, audio device, network endpoint,
  Ableton instance, mixer, or acoustic system was contacted. The sealed scan is
  base-revision evidence; the blocked full current-checkout Deep Scan still
  needs a managed read-only filesystem profile for reclassification.

### `occ_0901d08f0f0db3bff6de8aff` — high — no_change after preceding fix

- Finding: Ableton channel/master fader, mute, and pan setters could publish
  requested values into the observed-state cache even when UDP transmission
  failed.
- Current classification: a preceding Ableton peer/readback remediation
  already removed every setter-side state publication. Channel fader, mute,
  pan, name and master fader/pan now return strict transport booleans, while
  only fresh, correlated replies from the configured AbletonOSC peer may
  update observed state. No additional source patch was needed.
- Original sealed-base evidence: the four cited setters ignored `_send_osc`
  results and unconditionally assigned their requested values to `self.state`;
  cache-first getters could then return those values as though observed.
- Concrete closure: real setters retained the exact prior cache after
  `_send_osc` returned `False`, `None`, `0`, an arbitrary object, or raised an
  `OSError`. Accepted UDP submission returned `True` but still left observed
  state empty until a correlated reply, keeping transport acceptance separate
  from readback.
- Preserved behavior: all channel/master setters still emit the same OSC
  addresses and normalized values and report successful submission as `True`;
  a fresh correlated volume reply remains the only path that publishes the
  confirmed fader state.
- Generated Patch state: Security request
  `239e195c-9173-40bb-acc2-9041c70457e2` remains `generated` at version `2`.
  Its exact base-revision patch neither applies forward nor reverse over the
  broader current Ableton changes and would reintroduce setter-side cache
  publication after mere UDP acceptance. It was therefore not forced onto the
  dirty checkout; no Apply token is currently pending in Security.
- Files inspected: `backend/ableton_client.py`,
  `tests/test_ableton_client_security.py`, and the two recorded remediation
  patch artifacts.
- Validation:
  - current Ableton security module — `10 passed`;
  - exact `OSError` and non-`True` transport harness — passed for channel
    fader/mute/pan and master fader/pan;
  - accepted-send positive harness — five setters returned `True`, made five
    socket calls, and published zero unconfirmed cache entries;
  - Ableton security plus full server suite — `441 passed`;
  - Python compilation and scoped `git diff --check` — passed;
  - Ruff — unavailable in this checkout.
- Remaining uncertainty: no Ableton instance, UDP peer, audio device, network
  endpoint, mixer, or acoustic system was contacted. Canonical workbench
  reclassification still requires a new scan of the current checkout.

### `occ_515e8ac33669b73358bb3ba6` — high — no_change after preceding fix

- Finding: the Ableton OSC receiver accepted the first handshake datagram and
  subsequent state datagrams without checking their UDP origin, freshness, or
  relationship to a current query.
- Current classification: the preceding Ableton peer/readback remediation
  already rejects unexpected IPs and source ports before OSC parsing, accepts
  only a valid `/live/test` handshake from the resolved configured peer, and
  promotes state only from a supported, fresh, operation-correlated reply.
  The occurrence has `remediationState=idle`; no Generated Patch exists for it.
- Original sealed-base path: `recvfrom` discarded `addr`, `_handle_message`
  stored every raw OSC address, and translated unrequested fader/mute/name/pan
  values directly into mixer state and callbacks.
- Concrete closure: a real `_receiver_loop` harness rejected wrong-IP,
  wrong-port, unsolicited-known, unknown-address, malformed, and stale
  datagrams. Only the expected peer's current track-volume reply consumed the
  pending operation, produced one state key and emitted one callback.
- Preserved behavior: all seven supported query families — track volume, mute,
  name and pan, master volume, track-name list, and track count — still update
  state and callbacks after a matching fresh request. Pending replies and track
  identifiers are bounded; failed sends remove their pending entry.
- Independent Security evidence: sealed checkpoint scan
  `e01f678c-e332-4688-b22c-2a9a9ab1d8e9` completed with `0` reportable
  findings. Its explicit `surface_ableton` review covered UDP origin,
  correlation, cache bounds, handshake and result truth with disposition
  `no_issue_found`. Coverage remains partial outside those reviewed surfaces.
- Files inspected: `backend/ableton_client.py`,
  `tests/test_ableton_client_security.py`, and the sealed checkpoint manifest,
  coverage and report artifacts.
- Validation:
  - integrated real receiver malicious/positive harness — passed;
  - stale-reply receiver harness — passed;
  - seven supported correlated-reply positive controls — passed;
  - Ableton security tests — `10 passed`;
  - Ableton security plus full server suite — `441 passed`;
  - Python compilation and scoped `git diff --check` — passed;
  - Ruff — unavailable.
- Remaining uncertainty: AbletonOSC supplies no cryptographic application-level
  identity, so the software proof is exact configured UDP tuple plus fresh
  operation correlation, not cryptographic peer attestation. No real Ableton
  process or network was contacted, and no physical/readiness claim is made.

### `occ_06fcaff50949b719c354066f` — high — fixed locally

- Finding: an unterminated dLive SysEx frame could retain every later MIDI/TCP
  receive chunk in `_parse_buffer` without a hard bound.
- Applied boundary: a frame may retain at most 64 KiB. On overflow the parser
  clears retained bytes, ignores all remaining bytes through the next `F7`,
  resets running status, and then resumes ordinary parsing. Successful connect
  and disconnect also reset all stream-local parser state.
- Generated Patch: Security request
  `c7fb68d9-0420-45b4-b34f-a78401238fb6`, version `2`, artifact SHA-256
  `87685a267f2614c25c24a8c460b669d3e1eb446f30b3f1c811c5b4d0e999f4b9`.
  The digest matched, forward applicability passed before the local apply, and
  reverse applicability passed afterward, proving the exact patch is present
  in the current checkout.
- Negative proof: the pre-fix parser retained `65,537` bytes after the first
  unterminated frame and `98,305` after more input. Post-fix, an additional
  8 MiB remained at zero retained bytes; MIDI-like bytes embedded in the
  rejected frame were not dispatched.
- Positive proof: same-stream recovery after `F7`, fragmented legitimate
  SysEx (`Lead`), running-status NRPN, and reconnect recovery all passed.
- Validation:
  - focused security and compatibility tests — `6 passed`;
  - owning module — `49 passed`;
  - broader dLive client/connection/discovery/preflight/evidence/proof slice —
    `128 passed`;
  - Python compilation and scoped `git diff --check` — passed;
  - Ruff — unavailable.
- Files: `backend/dlive_client.py`, `tests/test_dlive_client.py`.
- Workbench truth: the request still reports `generated` because Security did
  not provide a pending Apply action token. No canonical workbench stage was
  fabricated; the exact generated artifact is locally applied and verified.
- Scope: all tests used in-memory inputs and fake sockets. No dLive console,
  network endpoint, audio device, Ableton instance, or acoustic system was
  contacted. A fresh full current-checkout scan remains required for canonical
  reclassification and complete-project closure.

### `occ_c42cc31cf52f32bdcabbbc76` — high — fixed locally

- Finding: `backend/auto_test_routing.py` tried six guessed routing schemas and
  sent twelve live OSC mutations through a caller-supplied `WingClient` without
  supervised authority, pre-state, readback, or rollback.
- Applied boundary: the legacy helper is now proposal-only. It returns the same
  six candidate methods and twelve address/value pairs as inert review data,
  with `live_authority_granted=false`, `live_write=false`, and
  `attempted_writes=0`. It contains no WING client import, connection, send, or
  disconnect sink; the CLI exits fail-closed with status `2`.
- Negative proof: the pre-fix real function boundary invoked a fake client's
  `send` method twelve times. Post-fix, a client object that raises on every
  attribute access remained untouched and the receipt recorded zero attempts.
- Positive proof: all original routing address candidates remain available for
  explicit operator review without transport access.
- Validation:
  - focused security tests — `3 passed`;
  - routing-management, transport-safety and supervised WING adapter slice —
    `94 passed`;
  - Python compilation, sink scan, and scoped `git diff --check` — passed;
  - Ruff — unavailable.
- Files: `backend/auto_test_routing.py`,
  `tests/test_auto_test_routing_security.py`.
- Workbench truth: the occurrence remains `idle`; Security has not produced a
  Generated Patch or pending Apply action token. The current local closure is
  therefore documented without fabricating a workbench stage.
- Scope: no WING console, socket, external network, audio device, Ableton
  instance, or acoustic system was contacted. Routing live apply remains
  unavailable until a canonical supervised authority/readback/rollback path is
  implemented and physically proven.

### `occ_bd9f34d7b5adb9fc30b97432` — high — fixed locally

- Finding: caller-controlled Ayaic `output_name` was interpolated into WAV and
  JSON filenames after joining with `output_dir`; `../escaped` wrote both
  generated files outside the trusted output directory.
- Applied boundary: output names must be non-empty filename stems with no
  absolute path, slash, backslash, NUL, or nested/traversal component. The
  output root is resolved once, every WAV/report/optional-MP3 destination is
  resolved and required to remain a direct child, and any pre-existing symlink
  at those destinations is rejected before audio loading or output writes.
- Negative proof: the pre-fix real pipeline wrote `escaped.wav` and
  `escaped_report.json` one directory above `output_dir`. Post-fix the same
  call raised `ValueError` before loading stems and created zero outside files;
  absolute, POSIX/Windows nested names, and WAV/MP3/report symlink cases also
  failed closed.
- Positive proof: a real temporary input WAV with `output_name=safe_mix`
  produced `safe_mix.wav` and `safe_mix_report.json` directly under the
  resolved output root.
- Validation:
  - focused negative and positive slice — `8 passed`;
  - full Ayaic owning module — `54 passed`;
  - Python compilation and scoped `git diff --check` — passed;
  - Ruff — unavailable.
- Files: `automixer/production_mix_v1/ayaic_offline_pipeline.py`,
  `tests/test_ayaic_offline_pipeline.py`.
- Workbench truth: the occurrence remains `idle`; Security has not generated a
  patch or issued a pending Apply action token, so no workbench stage was
  fabricated.
- Scope: temporary local audio only; no external network, console, live audio
  device, Ableton instance, or acoustic system was contacted.

### `occ_a1057d0d20a32406ba3e0976` — high — fixed locally

- Finding: the generic bulk-routing CLI and the WING-to-Dante CLI constructed
  `WingClient`, opened a console connection, and called the shared
  `route_multiple_outputs` mutation path without supervised authority,
  pre-state, readback, or rollback. Dante mode also sent four debug queries.
- Applied boundary: both modules are proposal-only and contain no `WingClient`,
  connect, send, bulk-route, or disconnect sink. Their callable APIs ignore any
  client, return `attempted_writes=0`, and preserve sequential output/source
  mappings plus candidate read addresses as inert review data. Both CLIs exit
  fail-closed with status `2` without opening a socket.
- Negative proof: pre-fix harnesses observed one bulk mutation in each module
  and four additional Dante query sends. Post-fix, clients that raise on any
  attribute access remained untouched for both entrypoints.
- Positive proof: generic MOD/CRD and Dante proposal rows preserve requested
  starting outputs, source channels, counts, and card name.
- Validation:
  - focused dual-CLI security tests — `4 passed`;
  - routing management, transport safety, and no-live-transport slice —
    `53 passed`;
  - Python compilation, direct sink scan, and scoped `git diff --check` —
    passed;
  - Ruff — unavailable.
- Files: `backend/route_channels.py`, `backend/route_dante_outputs.py`,
  `tests/test_bulk_routing_cli_security.py`.
- Workbench truth: the occurrence remains `idle`; no Generated Patch or Apply
  action token exists, so no canonical workbench transition was fabricated.
- Scope: no WING console, socket, network, audio device, Ableton instance, or
  acoustic system was contacted. Live output routing remains unavailable until
  a canonical supervised authority/readback/rollback implementation exists.

### `occ_2a75266e960cd95193db2217` — high — fixed locally

- Finding: authenticated WebSocket `set_mute`, `mute_channel`, and
  `unmute_channel` requests were quarantined for WING, but Mixing Station and
  other non-WING modes called `set_channel_mute` directly without canonical
  live authority, readback, WAL, or rollback.
- Applied boundary: valid non-WING mute requests now return
  `direct_mute_write_requires_supervised_authority` before any mixer access.
  WING retains its existing proposal quarantine, and malformed mute values
  retain the earlier `invalid_mute` response.
- Negative proof: the pre-fix real registered-handler boundary in
  `connection_mode=mixing_station` called `set_channel_mute(4, 1)` and emitted
  no block response. Post-fix all three aliases recorded zero mixer calls and
  returned the same fail-closed authority reason.
- Positive proof: WING proposal quarantine and invalid-value diagnostics remain
  unchanged and covered.
- Validation:
  - focused mute/alias/quarantine slice — `10 passed`;
  - full `tests/test_server.py` — `431 passed`;
  - Python compilation and scoped `git diff --check` — passed;
  - Ruff — unavailable.
- Files: `backend/handlers/mixer_handlers.py`, `tests/test_server.py`.
- Workbench truth: the occurrence remains `idle`; Security has not created a
  Generated Patch or pending Apply token, so no workbench stage was fabricated.
- Scope: in-memory handler fakes only; no Mixing Station process, socket,
  external network, console, audio device, Ableton instance, or acoustic system
  was contacted.

### `occ_db4567d2d5b933bf61d6ad00` — high — fixed locally

- Finding: dLive physical-readback artifact intake eagerly collected unlimited
  roots/files/payloads, decoded files without a byte limit, and passed
  arbitrarily deep or wide JSON into a recursive contract validator.
- Applied boundary: intake is capped at 16 roots and 128 artifacts; files are
  read through a 1 MiB per-file and 8 MiB aggregate budget. Encoded file depth
  is checked before `json.loads`; decoded/inline data is iteratively checked at
  depth 32, 50,000 nodes per artifact, 200,000 nodes aggregate, and bounded
  scalar-character totals. Limit breaches become explicit invalid
  `artifact_load_error` records and never grant fixture or live authority.
- Negative proof: before the fix, 257 inline payloads produced 257 records, a
  1,200-level inline mapping escaped intake as `RecursionError`, and a
  3,145,743-byte JSON file was fully loaded. Post-fix the same classes cap at
  128 records or return
  `artifact_load_error:DLiveStateArtifactResourceLimitError`; deep files are
  rejected before the recursive decoder. Cyclic mappings and wide node bombs
  also fail closed.
- Positive proof: the canonical one-channel operator artifact still becomes a
  `valid_operator_artifact_available` review candidate while fixture closure,
  MIDI/TCP sends, dLive writes, and live authority remain false.
- Validation:
  - focused malicious/legitimate slice — `6 passed`;
  - full owning module — `15 passed`;
  - intake, review queue, processing registry, and capability matrix —
    `87 passed`;
  - Python compilation and scoped `git diff --check` — passed;
  - Ruff — unavailable.
- Files: `backend/dlive_state_physical_readback_artifact_intake.py`,
  `tests/test_dlive_state_physical_readback_artifact_intake.py`.
- Workbench truth: the occurrence remains `idle`; no Generated Patch, pending
  Apply action, or action token exists, so no canonical stage was fabricated.
- Scope: filesystem fixtures and in-memory payloads only; no dLive console,
  network, audio device, Ableton instance, or acoustic system was contacted.

### `occ_a21ee9fb3507947a5009dcdf` — high — fixed locally

- Finding: a duck-typed in-process mixer could return a matching
  `status/address/confirmed_round_trips` mapping and have both apply and
  rollback persisted as verified physical WING proof. The fake dry-run runner
  intentionally exercised this path and labelled the resulting records as
  physically verified.
- Applied boundary: the receiver now rejects datagrams outside the configured
  WING IP:port before updating state or callbacks. Successful
  `WingClient.query_parameter` calls return a process-local, single-use receipt
  bound to that exact peer, two increasing receive sequences from the current
  query, and the canonical transport instance. The server consumes the receipt
  for every WING pre-read, CAS, post-write readback, and rollback read.
  Duck-typed clients, direct in-process callbacks, copied mappings, replaced
  query methods, wrong-peer data, stale/replayed receipts, and cross-transport
  receipts fail before they can establish proof.
- Artifact closure: the fake gain dry-run now treats rejection of fake physical
  proof, zero mixer writes, and no rollback attempt as its successful security
  outcome. It records `physical_proof_claimed=false` and can no longer produce
  `manual_write_applied` or `manual_write_rollback_applied` evidence from its
  in-process mixer.
- Negative proof: before the fix, the forged apply and forged rollback tests
  both passed as `verified`; afterward a forged apply is blocked before the
  setter, and swapping a forged query in after a canonical apply persists only
  `manual_write_rollback_blocked`, never a verified rollback event.
- Positive proof: canonical `WingClient` and `EnhancedOSCClient` receipts from
  the configured peer still verify exact values, preserve WING trim
  quantization tolerance, and support a canonical apply followed by rollback.
  Offline non-WING unit behavior remains outside the WING physical-proof
  boundary.
- Validation:
  - pre-fix vulnerable apply/rollback reproducer — `2 passed`;
  - combined WING client, wrapper, server, dry-run, evaluator, and no-WING
    suites — `963 passed`;
  - full no-WING release-check tests — `457 passed`;
  - fake dry-run security contract — `7 passed`;
  - real offline `tools/no_wing_release_check.py --quick` — `passed`, zero
    failed checks; generated proof row remained `dry_run_ready`,
    `promotion_allowed=false`, and `autonomous_ready=false`;
  - Python compilation and scoped `git diff --check` — passed;
  - Ruff — unavailable.
- Files: `backend/wing_client.py`, `backend/server.py`,
  `backend/supervised_proof_artifact_evaluator.py`,
  `tools/gain_proof_dry_run_runner.py`, `tools/no_wing_release_check.py`,
  `tests/test_wing_client.py`, `tests/test_enhanced_osc_client.py`,
  `tests/test_server.py`, `tests/test_gain_proof_dry_run_runner.py`,
  `tests/test_supervised_proof_artifact_evaluator.py`, and
  `tests/test_no_wing_release_check.py`.
- Workbench truth: the occurrence has no Generated Patch or pending Apply
  action token in the sealed scan artifacts, so no canonical workbench stage
  was fabricated.
- Scope: all verification used in-memory fakes, temporary artifacts, and
  unconnected local sockets. No WING console, external network, audio
  device, Ableton instance, or acoustic system was contacted. This closes the
  software trust boundary only; a fresh full current-checkout scan remains
  required for canonical reclassification.

### `occ_cb138b2638e0e80912f67137` — high — fixed locally

- Finding: feedback replay accepted coerced or non-finite source values,
  ignored its declared `max_q`, had no notch-depth limit, and persisted the
  resulting proposal as `applied` in trusted replay state.
- Applied boundary: feedback events now require an exact integer channel,
  allowlisted action, finite numeric frequency/gain/Q/confidence, exact boolean
  policy flags, and bounded acoustic ranges. Every proposal carries explicit
  `max_q` and `max_cut_db`; both declared limits are themselves capped at hard
  maxima. Restored/direct proposals must contain both limits and are checked
  again by the executor before any replay-state or incident-log mutation.
  The policy validator independently reports invalid feedback bounds even when
  no executor result is supplied.
- Negative proof: before the fix, Q `100` with `max_q=8`, gain `-1000 dB`, and
  `NaN` frequency all returned `status=applied`. After the fix, the builder
  rejects each attacker-controlled source event; an equivalent direct proposal
  is `blocked`, `applied=false`, has identical pre/post signatures, and leaves
  both feedback and incident state empty. Missing bounds, booleans used as
  numbers, string channels, infinities, out-of-range limits, and unknown
  actions also fail closed.
- Positive proof: a bounded notch at Q `8`, gain `-9 dB`, and explicit
  `max_q=8`/`max_cut_db=9` remains applied. The documented bounded
  `fader_reduce` fallback is preserved.
- Validation:
  - focused builder/executor/policy slice — `46 passed`;
  - all `tests/test_replay*.py` modules — `84 passed`;
  - related live-input/log/migration/offline/room/source/system/WING replay
    modules — `44 passed`;
  - real offline `tools/no_wing_release_check.py --quick` — `passed`, all
    `66/66` checks passed, no failed checks;
  - Python compilation and scoped `git diff --check` — passed;
  - Ruff — unavailable.
- Files: `ai_mixing_pipeline/decision_layer/replay_proposal_ranking.py`,
  `ai_mixing_pipeline/decision_layer/replay_executor.py`,
  `ai_mixing_pipeline/decision_layer/replay_policy_validator.py`,
  `tests/test_replay_proposal_ranking.py`, `tests/test_replay_executor.py`,
  `tests/test_replay_policy_validator.py`, and two deterministic replay
  fixtures whose hashes changed because `max_cut_db` is now explicit.
- Workbench truth: a current Security context read returned
  `remediationState.state=idle`; no Generated Patch, pending Apply request, or
  action token exists, so no canonical workbench transition was fabricated.
- Scope: replay simulation, local fixtures, and offline safety checks only. No
  WING/dLive console, external network, backend service, audio device, Ableton
  instance, or acoustic system was contacted; all authority/readiness flags in
  the no-WING artifact remain false.

### `occ_3f5b930c842d78c8f72b3444` — high — fixed locally

- Finding: a restored/direct gain proposal with no step limit received an
  effective zero through coercion, which disabled the conditional limit check.
  A `NaN` limit or target also made comparisons false and allowed non-finite or
  arbitrary gain state to be persisted as `applied`.
- Applied boundary: gain replay now requires an exact channel/action, finite
  current and requested trim in the WING `-18..+18 dB` range, and a present,
  finite, positive `max_step_db` no greater than `12 dB`. The builder validates
  source data and always emits an explicit bound; the executor independently
  revalidates restored/direct proposals and reads current state before testing
  the delta. The policy validator reports missing/malformed gain bounds even
  when no executor result is present.
- Negative proof: before the fix, a no-bound proposal and a `NaN`-bound
  proposal both applied `+60 dB`, while a builder-created `NaN` target was
  persisted. After the fix, missing/NaN limits return `blocked`, keep identical
  pre/post signatures and empty gain/incident state, and `NaN` source targets
  are rejected before proposal creation. Zero/infinite/oversized/string limits,
  string/bool channels and gains, non-finite/out-of-range trims, and unknown
  actions also fail closed.
- Positive proof: an explicit `3 dB` request with `max_step_db=3` still applies.
  A finite `36 dB` delta may remain an auditable proposal, but executor blocks
  it with `gain_step_exceeds_limit:36.000>12.000` before state mutation.
- Validation:
  - focused builder/executor/policy/failure-containment slice — `79 passed`;
  - all `tests/test_replay*.py` modules — `115 passed`;
  - related replay modules — `44 passed`;
  - full no-WING release-check test module — `457 passed`;
  - post-fix exploit harness, Python compilation, and scoped
    `git diff --check` — passed;
  - Ruff — unavailable.
- Files: `ai_mixing_pipeline/decision_layer/replay_proposal_ranking.py`,
  `ai_mixing_pipeline/decision_layer/replay_executor.py`,
  `ai_mixing_pipeline/decision_layer/replay_policy_validator.py`,
  `tests/test_replay_proposal_ranking.py`, `tests/test_replay_executor.py`, and
  `tests/test_replay_policy_validator.py`.
- Workbench truth: a current Security context read returned
  `remediationState.state=idle`; no Generated Patch, pending Apply request, or
  action token exists, so no canonical workbench transition was fabricated.
- Scope: local deterministic replay and offline tests only. No WING/dLive
  console, external network, backend service, audio device, Ableton instance,
  or acoustic system was contacted. This proves software closure, not physical
  console movement or concert readiness.

### `occ_f37b0d4cd886fbf9876eeb08` — high — fixed locally

- Finding: the gain/pan training path passed an attacker-influenced NPZ
  directly to `numpy.load` and then converted arrays to Python lists before
  enforcing any archive, shape, element, allocation, sample, or channel limit.
  A small compressed input could therefore request a very large decompression,
  NumPy allocation, Python-object expansion, or channel-padding allocation.
- Applied boundary: the loader opens the NPZ once, bounds the compressed file,
  ZIP member count, per-member and total uncompressed bytes, compression
  methods, names, duplicates, NPY header size/version, dimensions, each
  dimension, total elements, dtype, and array bytes before calling
  `numpy.load` on that same file descriptor. Object, structured, nonnumeric,
  encrypted, nested, malformed, and payload-size-mismatched members fail
  closed. Required gain/pan keys are checked before load.
- Allocation closure: Python-list amplification was removed. Dataset sample
  count, requested channel count, per-sample padded elements, and cumulative
  materialized elements are bounded before `zeros`, `vstack`, or model input
  construction.
- Negative proof: header-only allocation bombs are rejected before a
  monkeypatched `numpy.load` can run, including a zero-product shape with one
  enormous dimension. Tests also reject excessive member counts, compressed
  and uncompressed byte budgets, payload mismatches, unsafe member names,
  duplicate keys, object arrays, and unsafe channel-padding expansion without
  creating a real memory-pressure event.
- Positive proof: legitimate numeric NPZ gain/pan datasets still load and the
  predictor/agent-training archive behavior remains intact.
- Validation:
  - focused gain/pan loader, predictor, and agent-training archive slice —
    `32 passed`;
  - complete `tests/test_ml/` suite — `197 passed`;
  - Python compilation — passed;
  - Ruff — unavailable.
- Files: `backend/ml/npz_archive_safety.py`,
  `backend/ml/train_gain_predictor.py`, and
  `tests/test_ml/test_training_dataset_loaders.py`.
- Workbench truth: a current Security context read returned
  `remediationState.state=idle`; no Generated Patch, pending Apply request, or
  action token exists, so no canonical workbench transition was fabricated.
- Scope: local temporary archives and offline ML tests only. No model training
  job, external download, console, network service, audio device, Ableton
  instance, or acoustic system was contacted. A fresh full scan remains
  required for canonical reclassification.

### `occ_4560f7276c91dee738256692` — high — fixed locally

- Finding: the original service merged every remote gain/pan manifest field
  into the locally trusted target configuration, then coerced unbounded
  `n_epochs`, `batch_size`, `n_channels`, `n_samples`, and `lr` values directly
  into constructors and training loops. Large values could trigger excessive
  synthetic-data allocation, batch memory, model size, or training work.
- Source boundary: remote manifests may now supply dataset identity/location
  fields only. They cannot enable training, change model output, or override
  any gain/pan hyperparameter or local resource budget.
- Sink boundary: `train_gain_predictor` independently requires exact bounded
  integers, a finite bounded learning rate, and joint limits for synthetic
  elements, batch elements, dataset samples, and epoch-by-sample work before
  device, dataset, loader, or model creation. `SyntheticMixDataset` repeats
  allocation-critical validation for direct callers. External datasets are
  checked against the actual loaded length and maximum sample size before a
  `DataLoader` is built.
- Negative proof: a manifest containing `10**9` epochs/batch/channels/samples
  and `NaN` learning rate leaves the locally configured values unchanged.
  Direct huge, non-finite, excessive joint-allocation, batch, and work-factor
  requests raise before a monkeypatched dataset allocator can run.
- Positive proof: locally configured bounded gain/pan values remain selected,
  and a two-sample, one-channel synthetic dataset still constructs normally.
- Validation:
  - focused gain/pan, dataset-loader, and agent-training archive slice —
    `35 passed`;
  - complete `tests/test_ml/` suite — `200 passed`;
  - Python compilation and scoped `git diff --check` — passed;
  - Ruff — unavailable.
- Files: `backend/ml/agent_training_service.py`,
  `backend/ml/train_gain_predictor.py`,
  `tests/test_ml/test_agent_training_archive_loaders.py`, and
  `tests/test_ml/test_gain_pan_predictor.py`.
- Workbench truth: a current Security context read returned
  `remediationState.state=idle`; no Generated Patch, pending Apply request, or
  action token exists, so no canonical workbench transition was fabricated.
- Scope: configuration merge and offline ML tests only. No model training job,
  external download, console, network service, audio device, Ableton instance,
  or acoustic system was contacted. A fresh full scan remains required for
  canonical reclassification.

### `occ_165770202ca4a3c92908129b` — high — fixed locally

- Finding: the supposedly observational `list_all_snapshots` utility selected
  every index and sent `/$ctl/lib/$action = GO`, physically loading each scene,
  then sent another `GO` in an attempt to restore the original. Listing could
  therefore mutate the entire live-console state without mutation authority.
- Applied boundary: the utility no longer contains or sends the snapshot
  action endpoint, `GO`, or `LOAD`. It selects a bounded library cursor and
  queries the read-only `/$ctl/lib/$name` address already used by the canonical
  WING name lookup. It clears stale cached names, records duplicate names by
  their distinct indexes, and restores only the original library cursor in a
  `finally` block; the active scene/index are never changed.
- Negative proof: a recording fake confirms that neither the action endpoint
  nor `GO`/`LOAD` is emitted, active scene state remains identical, coerced or
  excessive scan ranges fail before any send, and `KeyboardInterrupt` still
  restores the cursor without loading a scene. Base-revision source evidence
  contains one `GO` per scanned index plus the restore `GO`; current source
  contains neither.
- Positive proof: bounded name queries still return all populated indexes,
  including equal names at separate indexes.
- Validation:
  - focused listing, maintenance CLI, and WING client slice — `54 passed`;
  - full no-WING release-check test module — `457 passed`;
  - wider no-WING/WING-rail run — `593 passed`, with only the expected active
    writer-lease assertion and pre-existing `m110_cp05_tracked_tree_not_clean`
    gate failing during the in-progress write;
  - Python compilation and scoped `git diff --check` — passed;
  - Ruff — unavailable.
- Files: `backend/list_all_snapshots.py` and
  `tests/test_list_all_snapshots_security.py`.
- Workbench truth: a current Security context read returned
  `remediationState.state=idle`; no Generated Patch, pending Apply request, or
  action token exists, so no canonical workbench transition was fabricated.
- Scope: recording fakes and offline tests only. No WING/dLive console,
  external network, backend service, audio device, Ableton instance, or
  acoustic system was contacted. This proves removal of the unsafe software
  path, not physical-console readiness.

### `occ_26962f8cedee5ccbab728054` — high — fixed locally

- Finding: the original unrestricted remote-target merge could control
  training work factors for channel-classifier, gain/pan, and mix-console.
  Their concrete sinks accepted independent large values that multiplied into
  unbounded dataset objects/elements, batches, model dimensions, scheduler
  duration, and epoch loops.
- Source boundary: remote manifests remain limited to dataset identity and
  location; enablement, output paths, train-without-dataset authority, and all
  hyperparameters remain locally configured.
- Sink boundary: each of the three trainers now independently requires exact
  bounded integers and finite bounded learning rates before device, dataset,
  loader, loss, optimizer, scheduler, or model construction. Joint budgets cap
  dataset samples/elements, per-batch elements where applicable, channel and
  feature dimensions, audio length/rate, and epoch-by-sample work. Synthetic
  constructors/generators repeat their allocation checks for direct callers,
  and actual external dataset sizes are rechecked before training.
- Negative proof: `10**9`, `NaN`, infinity, excessive cross-products, many
  small classifier samples, huge batches, and excessive epoch-by-sample work
  all raise before monkeypatched allocators run. The remote manifest regression
  verifies that the same malicious fields cannot replace bounded local values.
- Positive proof: small bounded synthetic classifier, gain/pan, and mix-console
  datasets still construct normally; prior predictor/loader behavior remains.
- Validation:
  - focused all-target resource/loader/manifest slice — `39 passed`;
  - complete `tests/test_ml/` suite — `204 passed`;
  - Python compilation and scoped `git diff --check` — passed;
  - Ruff — unavailable.
- Files: `backend/ml/agent_training_service.py`,
  `backend/ml/train_classifier.py`, `backend/ml/train_gain_predictor.py`,
  `backend/ml/train_mix_console.py`,
  `tests/test_ml/test_agent_training_archive_loaders.py`,
  `tests/test_ml/test_gain_pan_predictor.py`, and
  `tests/test_ml/test_training_resource_bounds.py`.
- Workbench truth: a current Security context read returned
  `remediationState.state=idle`; no Generated Patch, pending Apply request, or
  action token exists, so no canonical workbench transition was fabricated.
- Scope: local configuration and offline ML tests only. No real training job,
  model publication, download, network service, console, audio device, Ableton
  instance, or acoustic system was contacted. A fresh full scan remains
  required for canonical reclassification.

### `occ_7e9ce177ca899ab49ed65075` — high — fixed locally

- Finding: fader replay coerced channel/value/limit fields and used a truthy
  conditional for `max_step_db`; a missing or zero limit disabled the step
  check, while `NaN` made comparisons false. Direct/restored proposals could
  therefore persist unbounded or non-finite fader state as applied.
- Applied boundary: fader proposals now require an exact channel/action,
  finite current/requested values in `-144..0 dB`, and an explicit finite
  positive `max_step_db` capped at `12 dB`. The builder validates source
  confidence, deadband, and boolean policy fields and always emits a bound.
  The executor independently revalidates restored/direct proposals using
  current replay state before mutation. The policy validator reports malformed
  or missing fader bounds without relying on executor output.
- Negative proof: missing, zero, string, oversized, `NaN`, and infinite limits;
  string/bool channels; non-finite and out-of-range faders; above-unity faders;
  excessive deltas; malformed deadbands/confidence/booleans; and unknown
  actions all fail closed with identical pre/post state and empty incident log.
- Positive proof: a declared `12 dB` move from `-12` to `0 dB` applies, while
  `2.1 dB` against a `2 dB` bound is blocked before state mutation.
- Validation:
  - focused builder/executor/policy slice — `108 passed`;
  - all `tests/test_replay*.py` modules — `146 passed`;
  - full no-WING release-check test module — `457 passed`;
  - Python compilation and scoped `git diff --check` — passed;
  - Ruff — unavailable.
- Files: `ai_mixing_pipeline/decision_layer/replay_proposal_ranking.py`,
  `ai_mixing_pipeline/decision_layer/replay_executor.py`,
  `ai_mixing_pipeline/decision_layer/replay_policy_validator.py`,
  `tests/test_replay_proposal_ranking.py`, `tests/test_replay_executor.py`, and
  `tests/test_replay_policy_validator.py`.
- Workbench truth: a current Security context read returned
  `remediationState.state=idle`; no Generated Patch, pending Apply request, or
  action token exists, so no canonical workbench transition was fabricated.
- Scope: deterministic replay and offline tests only. No console, external
  network, backend service, audio device, Ableton instance, or acoustic system
  was contacted. This proves software closure, not physical movement.

### `occ_e9d34916f11339c2b60012b2` — high — fixed locally

- Finding: the mix-console loader called `numpy.load` before any archive or
  NPY-header budget and converted both arrays with `.tolist()`. A small
  compressed NPZ could therefore request large decompression/NumPy allocation,
  amplify into Python objects, and then trigger unbounded channel padding.
- Applied boundary: mix-console NPZ input now uses the shared same-file-handle
  validator before `numpy.load`, bounding compressed/archive/member/header,
  dtype, dimensions, elements, and uncompressed bytes. Required
  `multitracks`/`references` keys are verified pre-load. `.tolist()` was
  removed; sample count, exact channel range, per-sample padded elements,
  cumulative materialization, dimensionality, and reference length are checked
  before `zeros`/`vstack` or training handoff.
- Negative proof: a header-only allocation bomb is rejected before a
  monkeypatched `numpy.load` can execute. Separate cases reject 64-channel
  padding amplification and cumulative materialization overflow without
  inducing real memory pressure. Object-array rejection remains covered.
- Positive proof: a legitimate numeric two-sample multitrack/reference NPZ
  retains its original shapes and loads successfully.
- Validation:
  - focused loader/resource/archive slice — `26 passed`;
  - complete `tests/test_ml/` suite — `206 passed`;
  - Python compilation and scoped `git diff --check` — passed;
  - Ruff — unavailable.
- Files: `backend/ml/train_mix_console.py` and
  `tests/test_ml/test_training_dataset_loaders.py`; shared validation remains
  in `backend/ml/npz_archive_safety.py`.
- Workbench truth: a current Security context read returned
  `remediationState.state=idle`; no Generated Patch, pending Apply request, or
  action token exists, so no canonical workbench transition was fabricated.
- Scope: temporary local archives and offline tests only. No training job,
  model publication, download, network service, console, audio device, Ableton
  instance, or acoustic system was contacted. A fresh full scan remains
  required for canonical reclassification.

### `occ_087fe95011e490f352a29bea` — high — fixed locally by overlapping remediation

- Finding: the sealed base revision allowed remote mix-console manifest fields
  to flow into epochs, dimensions, sample count/rate, learning rate, allocation,
  and training work without a complete resource boundary.
- Current boundary: the shared remote-manifest merger now accepts only dataset
  identity/location fields. Compute hyperparameters remain local. The
  mix-console sink independently requires exact bounded integers, a finite
  bounded learning rate, bounded joint dataset elements, bounded
  epoch-by-sample work, and repeats actual-dataset checks before construction
  and training. These protections were already present from the preceding
  all-target resource remediations, so no speculative duplicate source patch
  was added for this occurrence.
- Negative proof: a remote manifest containing billion-scale epochs, channels,
  audio length, sample count/rate, and `NaN` learning rate cannot replace the
  bounded local mix-console configuration.
- Positive proof: the legitimate local values reach the training handoff
  unchanged while the remote dataset identity/location path remains supported.
- Validation:
  - focused manifest/resource/loader slice — `27 passed`;
  - complete `tests/test_ml/` suite — `207 passed`;
  - Python compilation and scoped `git diff --check` — passed;
  - Ruff — unavailable.
- Files: regression proof in
  `tests/test_ml/test_agent_training_archive_loaders.py`; overlapping controls
  in `backend/ml/agent_training_service.py` and
  `backend/ml/train_mix_console.py`.
- Workbench truth: the current occurrence is `generated` at version `2`, but
  has no `pendingAction`, delivered action token, or Apply request. Its stored
  base-revision patch was therefore not applied or represented as applied; the
  current checkout was verified directly.
- Scope: local configuration and offline ML tests only. No training job,
  model publication, download, external service, console, audio device,
  Ableton instance, or acoustic system was contacted. A fresh full scan remains
  required for canonical reclassification.

### `occ_9c7858666502503ace0f381e` — high — fixed locally

- Finding: mixer channel state and knowledge-base text were embedded in a
  prompt that could select the agentic `kimi_cli` backend, allowing
  attacker-influenced metadata to become instructions to a subprocess with
  filesystem/tool capabilities.
- Applied boundary: mixer recommendations now use a dedicated untrusted-data
  query path that permits only inference-only OpenAI, Ollama, and Perplexity
  transports. Explicit `kimi_cli` selection fails closed; auto fallback skips
  that backend entirely. The existing general query API remains available for
  separately trusted callers, and the mixer path still parses JSON data before
  it can become a typed recommendation.
- Negative proof: instruction-like channel metadata and KB context cannot
  invoke a mocked Kimi subprocess, including when `kimi_cli` is the configured
  backend or the first auto fallback.
- Positive proof: auto mode skips Kimi and accepts a legitimate JSON response
  from an inference-only Ollama backend; unavailable safe inference retains the
  existing deterministic fallback recommendation.
- Validation:
  - focused LLM/agent/handler slice — `42 passed`;
  - complete `tests/test_ai/` suite — `59 passed`;
  - full no-WING release-check module — `457 passed`;
  - Python compilation and scoped `git diff --check` — passed;
  - Ruff — unavailable.
- Files: `backend/ai/llm_client.py` and
  `tests/test_ai/test_llm_client.py`.
- Workbench truth: a current Security context read returned
  `remediationState.state=idle`; no Generated Patch, pending Apply request, or
  action token exists, so no canonical workbench transition was fabricated.
- Scope: mocked inference responses and offline tests only. No Kimi subprocess,
  external model service, console, backend service, audio device, Ableton
  instance, or acoustic system was contacted. This is software-path closure,
  not physical movement proof.

### `occ_a8170215755df404343d1c87` — high — fixed locally

- Finding: connection verification accepted the first UDP datagram regardless
  of source or OSC address, and the receiver cached state from any sender.
  Parameter queries also returned a pre-existing cache entry after a fixed
  sleep, allowing stale data to masquerade as current readback.
- Applied boundary: the configured OSC hostname is resolved once per
  connection into an exact IP+port allowlist and all sends use that resolved
  peer. Handshake verification ignores unexpected sources, malformed packets,
  and unrelated OSC addresses and requires a value-bearing reply for the
  current probe. The receiver rejects every datagram outside the bound peer.
  Query events are cleared before transmission and return a value only after a
  post-request state update; stale cache now returns `readback_unavailable`.
  Loopback discovery likewise verifies source IP+port and OSC framing.
- Negative proof: a correctly framed spoofed packet cannot establish a
  connection or mutate cached state; an expected-peer reply for another OSC
  address cannot satisfy the connection probe; and a stale cached trim value
  cannot satisfy a new parameter query.
- Positive proof: the current probe reply from the resolved peer establishes
  the mocked connection, updates the canonical state key, and a fresh matching
  trim reply remains available through `query_parameter`.
- Validation:
  - focused Mixing Station client module — `7 passed`;
  - expanded client/connection/offline-lock slice — `16 passed`;
  - server Mixing Station slice — `2 passed` (`430 deselected`);
  - full no-WING release-check module — `457 passed`;
  - Python compilation and scoped `git diff --check` — passed;
  - Ruff — unavailable.
- Files: `backend/mixing_station_client.py` and
  `tests/test_mixing_station_client.py`.
- Workbench truth: a current Security context read returned
  `remediationState.state=idle`; no Generated Patch, pending Apply request, or
  action token exists, so no canonical workbench transition was fabricated.
- Scope: synthetic OSC packets, fake sockets, and offline tests only. No UDP
  socket was opened to Mixing Station or a console; no external network,
  backend service, audio device, Ableton instance, or acoustic system was
  contacted. This proves software peer/readback binding, not hardware identity
  or physical movement.

### `occ_819907a1f3fdfea344f4f0ad` — high — fixed locally

- Finding: relative fader, pan, and compressor limits were applied only when a
  getter happened to return a value. Missing/failed readback fell back to
  `0 dB` or the requested target and still issued the write.
- Applied boundary: current values now come through a shared finite readback
  helper that prefers `query_parameter` and requires `status=received` for the
  exact operation. If an operation-specific API exists, timeout, unsupported,
  malformed, boolean, or non-finite results fail closed without falling back to
  a stale getter. Fader setters now return an application result and require a
  current value; pan and compressor writes likewise require current readback
  before applying their configured relative step limits.
- Negative proof: a mixer returning timeout for the exact fader, pan, and
  compressor queries cannot receive any of those writes, even though its
  legacy getters return plausible cached values. No action is marked applied.
- Positive proof: fresh exact readback bounds a requested fader move to `1 dB`,
  pan to `0.25`, and compressor threshold to `3 dB`, then records the action.
- Validation:
  - focused MixingAgent module — `23 passed`;
  - complete AI plus agent-handler slice — `68 passed`;
  - full no-WING release-check module — `457 passed`;
  - Python compilation and scoped `git diff --check` — passed;
  - Ruff — unavailable.
- Files: `backend/ai/agent.py` and `tests/test_ai/test_agent.py`.
- Workbench truth: a current Security context read returned
  `remediationState.state=idle`; no Generated Patch, pending Apply request, or
  action token exists, so no canonical workbench transition was fabricated.
- Scope: fake mixer objects and offline tests only. No live write, console,
  external network, backend service, audio device, Ableton instance, or
  acoustic system was contacted. This proves fail-closed software behavior,
  not current hardware readback or physical movement.

### `occ_1c766f48056bb846a3920401` — high — fixed locally

- Finding: `NaN`/infinite action, render, critic, configuration, weight, and
  score values flowed through `float` plus `min`/`max` comparisons. Because
  comparisons against `NaN` are false and clamp ordering could turn it into a
  high score, malformed candidates could pass safety or win selection.
- Applied boundary: a shared strict numeric helper accepts only real,
  non-boolean, finite values. Both offline Safety Governors now reject invalid
  action limits/values, render metrics, critic inputs, improvement evidence,
  and composed safety results with explicit reasons. Score/delta normalization
  maps invalid critic values to neutral zero; identity aggregates reject any
  malformed component; invalid/negative weights and decision configuration
  fail before selection; aggregate output is guaranteed finite. The decision
  engine separately requires finite safety evidence before treating a
  candidate as safe.
- Negative proof: `NaN`, positive infinity, and negative infinity are rejected
  for gain actions; non-finite peak/headroom/correlation render metrics and
  critic values fail safety; non-finite critic scores cannot beat no-change;
  non-finite safety evidence cannot authorize selection; and malformed config
  raises before a run.
- Positive proof: existing bounded action, safe-render, weighted-critic,
  no-change threshold, proxy weighting, sandbox rendering, and offline runner
  behavior remains green.
- Validation:
  - focused safety/decision modules — `23 passed`;
  - expanded safety/decision/runner/generator/renderer slice — `33 passed`;
  - full no-WING release-check module — `457 passed`;
  - Python compilation and scoped `git diff --check` — passed;
  - Ruff — unavailable.
- Files: `ai_mixing_pipeline/numeric_safety.py`,
  `ai_mixing_pipeline/decision_layer/score_normalizer.py`,
  `ai_mixing_pipeline/decision_layer/decision_engine.py`,
  `ai_mixing_pipeline/decision_layer/safety_governor.py`,
  `ai_mixing_pipeline/safety_governor/governor.py`,
  `tests/test_safety_governor.py`, and `tests/test_decision_engine.py`.
- Workbench truth: a current Security context read returned
  `remediationState.state=idle`; no Generated Patch, pending Apply request, or
  action token exists, so no canonical workbench transition was fabricated.
- Scope: synthetic candidates, patched metrics, temporary audio fixtures, and
  offline tests only. No live renderer, console, external network, backend
  service, audio device, Ableton instance, or acoustic system was contacted.
  This proves deterministic software rejection, not physical readiness.

### `occ_42fbffae3d8307edb383d349` — high — fixed locally

- Finding: LLM numeric fields were converted with `float` and clamped with
  `min`/`max`; `NaN`/infinity could normalize to a physical maximum and reach
  fader, pan, EQ, or compressor setters.
- Applied boundary: action numeric fields now require an exact non-boolean
  finite real value before clamping. Channel and EQ-band identifiers require
  bounded integers. A malformed EQ collection or any malformed gain, pan, EQ,
  threshold, ratio, attack, or release field rejects the whole recommendation
  and records a rejected audit entry. The LLM dispatch sink repeats the full
  normalization, and the fader setter independently rejects non-finite targets,
  so direct calls cannot bypass preparation.
- Negative proof: nine payload families covering `NaN` and both infinities in
  every physical-control group produce no mixer calls and no applied history,
  both through `_prepare_actions` and direct `_act` dispatch.
- Positive proof: a bounded combined fader/pan/EQ/compressor recommendation
  retains its supported values and remains eligible for the existing fresh-
  readback and relative-step controls.
- Validation:
  - focused MixingAgent module — `33 passed`;
  - complete AI plus agent-handler slice — `78 passed`;
  - full no-WING release-check module — `457 passed`;
  - Python compilation and scoped `git diff --check` — passed;
  - Ruff — unavailable.
- Files: `backend/ai/agent.py` and `tests/test_ai/test_agent.py`.
- Workbench truth: a current Security context read returned
  `remediationState.state=idle`; no Generated Patch, pending Apply request, or
  action token exists, so no canonical workbench transition was fabricated.
- Scope: fake mixer objects and offline tests only. No live write, console,
  external network, backend service, audio device, Ableton instance, or
  acoustic system was contacted. This proves software rejection before the
  sink, not physical readiness.

### `occ_6964b929ca619c4e7311c7fc` — high — fixed locally

- Finding: the normal configured dLive route selected the non-TLS MixRack port
  and passed no authentication password. Direct `DLiveClient.connect` also
  permitted non-loopback cleartext, while TLS loaded trust only after opening
  the TCP connection and accepted an empty password.
- Applied boundary: the configured route now requires TLS, the matching TLS
  port, an absolute CA path, an explicit certificate identity, and a validated
  environment-variable name for the password. The password value is resolved
  only inside the trusted server boundary and is never accepted from WebSocket
  input or stored in YAML. Missing configuration/secret fails before client
  construction. `DLiveClient` independently rejects non-loopback cleartext,
  missing CA/identity, and empty, oversized, delimiter/control-bearing
  passwords. Trust configuration is loaded before socket creation; the default
  SSL context performs CA and hostname verification.
- Negative proof: non-loopback cleartext, absent CA, absent identity, absent
  password, certificate failure, unconfigured destination, and missing env
  value all fail before an authorized transport or client is established.
- Positive proof: a fully configured mocked TLS route passes the dummy password
  from its named environment variable to `DLiveClient`; the TLS client uses the
  configured CA and hostname and sends the expected authenticated profile
  payload after wrapping.
- Validation:
  - focused connection/security slice — `22 passed` (`470 deselected`);
  - expanded dLive/offline transport slice — `92 passed`;
  - full no-WING release-check module — `457 passed`;
  - Python compilation and scoped `git diff --check` — passed;
  - Ruff — unavailable.
- Files: `backend/dlive_client.py`, `backend/server.py`,
  `config/automixer.yaml`, `tests/test_dlive_client.py`, and
  `tests/test_server.py`.
- Workbench truth: a current Security context read returned
  `remediationState.state=idle`; no Generated Patch, pending Apply request, or
  action token exists, so no canonical workbench transition was fabricated.
- Scope: fake sockets, dummy credentials, mocks, and offline tests only. No
  real secret was read, no network socket reached a console, and no console,
  audio device, Ableton instance, or acoustic system was contacted. Operators
  must still provision the CA path, certificate identity, and password env
  reference before the configured route becomes available.

### `occ_d455f3db3027a42c165c63b1` — high — fixed locally

- Finding: `requirements-ai-decision.txt`, the installer, and its documentation
  directed `pip` to execute `pymixconsole` from an unpinned mutable GitHub
  default branch.
- Applied boundary: the automated requirements and installer no longer contain
  or execute the VCS dependency. The installer explicitly reports that it will
  not download `pymixconsole`; documentation prohibits mutable VCS URLs and
  permits only an independently reviewed immutable artifact through the
  operator's normal supply-chain process. The existing adapter still uses a
  separately provisioned local module when present and otherwise performs the
  real dependency-light `fallback_virtual_mixer` path.
- Negative proof: executable requirement lines and installer text are tested
  to contain no `git+`/GitHub dependency, so the original default-branch sink
  is unreachable from the project installer.
- Positive proof: the adapter/fallback test renders an output successfully
  whether or not a reviewed local `pymixconsole` exists.
- Validation:
  - installer security plus virtual-mixer fallback — `5 passed`;
  - expanded decision-layer slice — `38 passed`;
  - shell syntax (`bash -n`) — passed;
  - full no-WING release-check module — `457 passed`;
  - scoped `git diff --check` — passed;
  - Ruff — unavailable.
- Files: `requirements-ai-decision.txt`,
  `scripts/install_ai_decision_layer.sh`,
  `Docs/AI_DECISION_CORRECTION_LAYER.md`, and
  `tests/test_ai_decision_installer_security.py`.
- Workbench truth: a current Security context read returned
  `remediationState.state=idle`; no Generated Patch, pending Apply request, or
  action token exists, so no canonical workbench transition was fabricated.
- Scope: static installer inspection and offline rendering only. The installer
  was not executed, no dependency/network/repository service was contacted,
  and no console, audio device, Ableton instance, or acoustic system was used.
  Existing independently installed code remains outside this proof and must be
  reviewed through its own provenance process.

### `occ_a106a293ea29a95e019c8075` — high — fixed locally

- Finding: global live queue mode ignored `OSCCommand.dry_run_only=True` and
  passed the command to `sender.send`.
- Fix: `OSCCommandQueue.flush` now rechecks every command immediately before
  the transport sink. Restricted commands produce no sender call or `sent`
  record and instead retain an auditable `blocked` record with
  `block_reason=command_dry_run_only` and a warning.
- Negative proof: global live mode with a command-level dry-run candidate
  produced zero replay sender incidents, zero sent commands, and one blocked
  command. Positive proof: ordinary dry-run reconstruction remains exact and
  transport-free.
- Validation: replay harness `12 passed`; production-mix plus replay `33
  passed`; full no-WING release check, Python compilation, and scoped
  `git diff --check` passed; Ruff unavailable.
- Files: `automixer/production_mix_v1/osc_queue.py` and
  `tests/test_replay_harness.py`.
- Workbench: current state is `idle`; no Generated Patch, Apply request, or
  action token existed, so no transition was fabricated.
- Scope: offline replay only; no live sender, console, external network,
  backend, audio device, Ableton instance, or acoustic system was contacted.

## 2026-08-19 — complete local disposition of the sealed 143-occurrence set

- The frozen ledger contains the exact sealed occurrence set: 143 rows
  (77 high, 59 medium, 7 low), occurrence-set SHA-256
  `80146278415ba2ef4108d7998f7eff70a061a27df1b7afadbcbb926c9c6a57fd`.
- All 143 rows are now `locally_verified_unsealed`. Every row keeps
  `closure_authority=fresh_full_scan`, `hardware_contacted=false`, and
  `live_authority_granted=false`.
- High-severity owning regression: 728 passed.
- Medium/Low owning and adjacent regression: 272 passed.
- Expanded server/network/ML regression: 544 passed.
- Replay integrity regression after canonical pre/post-state repair: 57 passed
  with zero validator findings.
- Aggregate repository modules passed in bounded pytest processes to preserve
  the production 512 MiB evidence-store reserve on the nearly full checkout
  volume. The lease-sensitive final package must be repeated after releasing
  `ops/autopilot/WRITER_LEASE/owner.json`.
- No WING, dLive, audio device, Ableton instance, Telegram endpoint, remote ML
  feed, or MCP service was contacted. No live authority or readiness closure
  was produced.
- A fresh full current-checkout Codex Security scan is still required for
  canonical reclassification and closure. The Security connector is currently
  unavailable, so this report must not be described as sealed closure.
