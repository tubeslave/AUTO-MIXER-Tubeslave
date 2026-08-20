# Kimi independent critique — DLIVE-READINESS-20260803 (Codex proposal)

**Role:** independent architect/critic only. No code edits, no hardware/network contact.
**Review format:** per AGENTS.md council critique format.

Sources reviewed: `AGENTS.md`, `.ai/briefs/DLIVE-READINESS-20260803.md`,
`.ai/proposals/DLIVE-READINESS-20260803-codex.md`, `backend/dlive_client.py`,
plus spot-checks of `backend/server.py` (`connect_dlive`, ~L13119),
`backend/mixer_client_base.py` (L135–146), `backend/user_config_store.py`
(L52–78), `backend/operator_product_state.py` (L2257+). Note: the official
A&H dLive MIDI-over-TCP/IP v2.0 PDF is **not vendored in the repo** (only WING
manuals exist under `Docs/manuals/`), so protocol facts below rest on the
brief's evidence list; that itself is a finding (see §2.1).

---

## 1. Strongest part of the Codex proposal

The **fail-closed posture combined with the fresh-readback fix** is the core
value and is exactly right:

- Disabling protocol-unverified families (pan, comp, gate, delay, polarity)
  instead of "warn and keep sending" is the only option consistent with the
  brief's safety boundary and AGENTS.md ("smallest safe diff", "unsupported
  families fail closed"). The alternative the proposal correctly rejects —
  keep the client and add warnings — leaves wrong bytes reachable over a live
  socket, which is unacceptable on a show network.
- Item 3 (active read-only GET with **fresh timestamp matching**) targets a
  real, confirmed bug: `set_fader_verified()` reads
  `self.state[f"fader/{channel}"]` immediately after send; if any earlier echo
  populated that key, the loop exits instantly and a **stale cached value is
  accepted as confirmation** (`dlive_client.py` L575–592). `state_timestamps`
  already exists but is never consulted. This is the single most dangerous
  defect in the file because it manufactures false "applied" evidence — the
  exact evidence class the brief says dLive currently lacks.
- Keeping 48V forbidden and requiring an explicit input→preamp-socket mapping
  (item 5) correctly identifies that MIDI channel index ≠ physical preamp
  socket; the current `set_gain`/`set_pad`/`set_phantom` implicitly assume
  identity mapping and could drive the wrong head amp.
- The no-socket preflight (item 6) plus byte-vector regression tests from
  official examples (item 7) is the right shape for "prepare without touching
  hardware".

## 2. Possible mistakes / blind spots

1. **No authoritative protocol source in-repo.** The proposal says "correct
   … from the official v2.0 document" but the repo does not contain that
   document. The brief's claims (7-bit fader values, NRPN parameter `0x17`,
   distinct Surface/MixRack ports, note-number ranges, running status) are
   asserted, not locally verifiable. Implementing byte-exact builders "from
   memory" risks swapping one speculative protocol for another.
   *Required addition:* vendor the v2.0 PDF (or an extracted, cited example
   table) into `Docs/` and make every byte-vector test cite a document
   section/example.

2. **TLS/auth fiction not confronted head-on.** The official v2.0 protocol
   defines plain TCP rendezvous ports (Surface vs MixRack); the client's
   `51328/51329` ports and the `user_profile,password` → `"AuthOK"` handshake
   (`dlive_client.py` L25–26, L326–334) appear in no A&H MIDI/TCP document.
   The proposal's risks section still says TLS "may require a trusted
   CA/hostname and credentials" — that perpetuates the fiction instead of
   resolving it. The critique-level requirement: either cite a document that
   defines this handshake, or **remove/quarantine the TLS+auth code path
   entirely** (fail closed), and fix the port constants to the documented
   Surface/MixRack pair. Related confirmed defect the proposal only implies:
   `server.connect_dlive()` accepts TLS fields that the WebSocket handler
   drops (brief, evidence list) — endpoint/TLS validation must happen
   **before** socket creation, not inside `connect()`.

3. **Receiver reassembly is broken for the documented 7-bit protocol, and the
   proposal doesn't name it.** `_handle_cc` only reconstructs a fader value
   when Data Entry **LSB** (`0x26`) arrives (L465–475), keyed on NRPN MSB
   `0x00`. Under the official protocol the fader family uses NRPN parameter
   `0x17` with a **7-bit value sent as Data Entry MSB (`0x06`) only** — so
   even after fixing the builders, readback would never populate `fader/N`
   and `set_fader_verified` would return `unconfirmed` forever (or worse,
   match stale state per §1). The fix must reassemble on Data Entry MSB alone
   for 7-bit parameter families, keyed on the documented parameter ID, and
   must handle **running status** (brief evidence) and TCP fragmentation
   together — the current parser treats a data byte after a truncated status
   as a new status (`else: i += 1`), which silently desynchronizes the stream.
   The proposal's "running-status parsing" bullet covers this only if these
   three defects are explicitly enumerated in the plan.

4. **`set_fader_verified()` bypasses the fader ceiling.** `set_fader()`
   clamps to `AUTOMIXER_FADER_CEILING_DB` (default 0 dBFS); the verified path
   builds its NRPN from the **unclamped** `value_db` (L569–570). So the
   "safer" verified API is actually the one that can send +10 dB. Any rewrite
   must apply the ceiling in both paths and compare readback against the
   value actually sent. The proposal does not mention this.

5. **Fader law is not "linear interpolation in 7 bits".** All current
   converters (`db_to_fader_value`, `freq_to_nrpn`, `q_to_nrpn`, …) are
   ad-hoc linear/log maps onto 14 bits. The official doc defines the fader
   curve (piecewise/table). The proposal says "correct 7-bit value scales"
   but does not call out that the **dB↔value law itself** must come from the
   document table, not from re-scaling the existing formulas. Same for mute
   velocity semantics, scene bank/program numbering (`//128`, `%128` at
   L673–675 vs. the documented scheme), and SysEx message-type bytes
   (`0x03/0x05/0x0A/0x0B/0x13`) which are currently guesses with plausible
   numbers.

6. **Channel-type offset table conflates distinct ranges.**
   `CHANNEL_TYPE_OFFSET` maps `mono_group` and `stereo_group` to the same
   offset (and likewise aux/FX-send, matrix pairs) (L75–86). The brief states
   the official note-number ranges differ for stereo buses, FX returns,
   mains, DCA and mute groups. Item 1 says "channel selectors" but the plan
   must enumerate the full note-range table per type, or stereo/main/DCA
   addressing stays wrong in a subtler way than today.

7. **Scope/risk of touching live-adjacent surfaces.** The file list includes
   `server.py`, `connection_handlers.py`, `mixer_discovery.py`,
   `operator_product_state.py` consumers (cache-address strings like
   `note/{midi_ch}/{ch-1}` at L2291/2412 depend on the current key scheme),
   and config files. Changing state-key naming or NRPN reassembly silently
   breaks those shadow/read-surface consumers. The proposal's test plan does
   not mention regression coverage for `operator_product_state` dLive rows.

## 3. Where the solution is too complex, fragile, or costly

- **Breadth vs. the "smallest safe diff" rule.** The proposal touches ≥10
  files plus docs plus tests in one cycle. Cheaper and safer: split into
  (A) protocol-truth layer in `dlive_client.py` + new tests only, then
  (B) server/config/preflight plumbing. Each independently mergeable and
  reviewable; (A) alone removes the ability to send wrong bytes.
- **A full generic MIDI codec is rejected — correctly.** But the bounded
  rewrite should stay bounded: builders + a streaming parser + a
  fail-closed capability table, nothing more. Do not grow a scene/FX/name
  management layer before physical evidence exists.
- **Test replacement cost is understated.** Existing dLive tests encode the
  wrong protocol (proposal risk #1). Replacing them is not just deletion —
  every removed test should be accounted for in the ADR so the diff doesn't
  look like coverage loss. Budget time for this.
- **Preflight cost/benefit is good** only if it stays pure (no socket). The
  moment it does a "TCP-open probe" it violates this cycle's safety boundary
  (brief: no TCP connection). The proposal's risk #3 ("TCP-open identifies a
  service, not the console") hints at a probe; keep the probe out of this
  cycle entirely — it belongs to the attended runbook.

## 4. What should be kept from the Codex proposal

1. Fail-closed removal of pan/dynamics/gate/delay/polarity sends (return
   `False`, zero bytes) — non-negotiable.
2. Fresh-timestamp readback for fader/mute/name with active GET requests.
3. Explicit input→preamp-socket mapping gate before gain/pad; 48V forbidden
   in the normal integration regardless.
4. Base MIDI channel restricted to the documented `0..11` range; explicit
   Surface/MixRack endpoint validation before socket creation.
5. Pure no-socket preflight report + attended read-only runbook + ADR naming
   the remaining hardware evidence gates.
6. Byte-vector unit tests from official examples; replacement (not
   preservation) of wrong-protocol tests.
7. Keeping the WING milestone untouched and denying dLive live authority —
   matches the brief exactly.
8. The existing `AUTOMIXER_FADER_CEILING_DB` transport-boundary clamp —
   extended to the verified path (see §2.4).

## 5. Revised recommendation

Adopt the Codex proposal **with five mandatory amendments**, phased:

**Phase A — protocol truth layer (this cycle, no other files):**
- A1. Vendor/cite the official v2.0 document in `Docs/`; every constant,
  range table and byte-vector test cites a document section. No value may be
  implemented from memory.
- A2. Rewrite builders to documented bytes: correct Surface/MixRack ports;
  base channel `0..11`; full note-number range table per channel type
  (incl. stereo buses, FX returns, mains, DCA, mute groups); NRPN fader
  parameter `0x17` with 7-bit Data-Entry-MSB-only values and the documented
  fader dB table; documented mute, scene, name/colour SysEx formats.
- A3. Fail closed: pan/comp/gate/delay/polarity/FX-send builders return
  `False` and send nothing; TLS+auth path removed or quarantined behind an
  explicit "undocumented, disabled" gate; gain/pad blocked until explicit
  socket mapping; 48V always blocked.
- A4. Fix the receiver: running status, TCP-fragment-safe framing, 7-bit
  NRPN reassembly completing on Data Entry MSB, per-family parameter keys.
- A5. Fix `set_fader_verified`: apply the ceiling, require readback
  timestamp strictly newer than send time, compare against the sent value.
- A6. Replace wrong-protocol tests; add stale-cache, zero-send,
  running-status/fragmentation, and ceiling-bypass regression tests.

**Phase B — integration & handoff (separate patch):**
- Server/WebSocket endpoint+TLS field consistency validated before socket
  creation; `operator_product_state` key-scheme consumers updated with
  regression tests; pure no-socket preflight; runbook + ADR naming the
  attended read-only capture and the one-parameter apply/readback/rollback
  proof as the remaining physical gates.

**Explicitly out of scope for both phases:** any TCP probe or connection to
dLive, hardware discovery scans, dLive live authority, WING milestone
changes.

**Where I may be wrong:** if the team can produce an A&H document that does
define the TLS/auth handshake and ports 51328/51329, amendment A3's TLS
removal softens to "validate and document"; the burden of proof sits with
the document, not the code.
