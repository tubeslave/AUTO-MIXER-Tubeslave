# System Inspector Capability Matrix

Status: current implementation truth, 2026-07-27

## Capability states

- `SUPPORTED` — working adapter plus tests in this repository.
- `PARTIALLY_SUPPORTED` — bounded subset works and limitations are explicit.
- `IMPORT_ONLY` — only validated offline import works.
- `MANUAL_CONFIRMATION` — operator evidence is required.
- `UNAVAILABLE` — no working adapter exists.
- `EXPERIMENTAL` — not a supported production contract.

## Current matrix

| Source | Data | Method | Current state | Limitations | Reliability |
|---|---|---|---|---|---|
| Simulated macOS | computer, interfaces, routes metadata, CoreAudio metadata, processes | Versioned JSON snapshot replay | `SUPPORTED` | Simulator evidence only; no local OS observation | Deterministic for fixture bytes |
| Live macOS | interfaces, IP, routes, DNS, link/service order, processes, power | Allowlisted system APIs/commands | `UNAVAILABLE` | Collector not implemented; active observation is a later phase | None claimed |
| Simulated WING | device, firmware label, sample rate, input patch, Main assignments | Versioned JSON snapshot replay | `SUPPORTED` | Not physical readback | Deterministic for fixture bytes |
| Live WING Rack | model/firmware, channels, patch, buses, Main, meters | Typed zero-argument OSC queries through future `WingReadClient` | `UNAVAILABLE` for Inspector | Existing `WingClient` mixes reads/writes and is intentionally not imported | No Inspector live claim |
| Imported Dante JSON | devices, endpoints, subscriptions, imported clock metadata | `DanteSnapshotFileProvider` schema represented by replay loader | `IMPORT_ONLY` | It proves imported records, not live discovery, clock sync or audio flow | High for validated bytes; freshness depends on source |
| Dante Controller preset XML | device roles, selected configuration and subscriptions | Offline preset import | `UNAVAILABLE` | Audinate presets can contain subscriptions, but no parser exists here | None claimed |
| Dante Controller live state | devices, subscriptions, clock and health | No assumed public Controller API | `UNAVAILABLE` | GUI automation is not a production method | None claimed |
| Dante Managed API / DDM | managed devices, channels, status and routing | Licensed/authenticated official API when DDM is present | `UNAVAILABLE` | No DDM, credentials, license or adapter is configured; using it is a manual checkpoint | None claimed |
| Dante Application Library | Dante audio in a macOS/Windows application | Licensed Audinate library | `UNAVAILABLE` | Not a Controller-topology shortcut; license/SDK use requires approval | None claimed |
| Dante DVS state | process/device presence and sample rate | macOS process/CoreAudio observation | `UNAVAILABLE` live, fixture `SUPPORTED` | Process presence does not prove subscription or audio signal | None claimed live |
| Ableton Live | tracks, output mapping, sample rate | Live API/OSC/project parser after validation | `UNAVAILABLE` | No Inspector adapter; no project/control access | None claimed |
| Existing audio capture | device list and samples | `backend/audio_capture.py` / sounddevice | `UNAVAILABLE` to Inspector | Live capture is an explicit permission checkpoint; Inspector imports are prohibited in Phase 0 | None claimed |
| CoreAudio metadata | device UID, channel layout, nominal/actual rate, clock domain where exposed | Core Audio property APIs | `UNAVAILABLE` live, fixture `SUPPORTED` | No Inspector live adapter; some properties can be unavailable by device | None claimed live |
| Baseline | expected devices and routes | Safe YAML profile + versioned baseline JSON | `SUPPORTED` for comparison | Create/version/export/import workflows are not implemented yet | Deterministic |
| JSON report | graph, comparisons, findings, readiness/coverage | Canonical sorted JSON | `SUPPORTED` | HTML/PDF not yet implemented | Byte-deterministic |
| HTML/PDF report | operator report | Future renderer | `UNAVAILABLE` | No renderer/test yet | None claimed |
| UI tab | overview, graph, findings, baseline | Future frontend adapter | `UNAVAILABLE` | No backend WebSocket/API surface or React tab yet | None claimed |

## WING protocol evidence

Repository sources:

- `Docs/Behringer_WING_Official/text/WING_Remote_Protocols_V3.1-03.txt`
  documents UDP 2222/2223 initiation, zero-argument parameter retrieval,
  value-bearing modification, data types and subscription behavior.
- `Docs/TOOLS.md` records the project’s confirmed WING fw3.0.5 protocol
  conventions.
- `backend/wing_client.py` contains working query/readback code but also
  value-bearing writes and setters; it is not an Inspector dependency.

Firmware/version support must be bound to the actual console and exact
documented endpoint set at the later live checkpoint.

## Official Dante evidence

- Audinate documents that Controller presets are XML and may include Rx
  subscriptions:
  <https://dev.audinate.com/GA/dante-controller/userguide/webhelp/content/saving_presets-2.htm>
- Controller can display devices, routes, clock, performance and errors, but
  this does not imply an application API:
  <https://dev.audinate.com/GA/dante-controller/userguide/webhelp/content/about_dante_controller.htm>
- Dante Managed API exists with Dante Domain Manager and is authenticated; it
  is not assumed present in the user system:
  <https://dev.audinate.com/GA/ddm/userguide/1.8/webhelp/content/appendix/dante_managed_api_security_notes.htm>
- Audinate lists Dante API and Dante Application Library as licensed software:
  <https://www.audinate.com/legal/software-licensing/>

Therefore the first supported Dante path is validated import. No undocumented
protocol, Controller click automation or ping-based audio claim is used.

## Official macOS evidence

- Apple documents `networksetup` capabilities, including listing network
  services; the live collector must allowlist read-only flags and never expose
  setters:
  <https://support.apple.com/guide/remote-desktop/about-networksetup-apdd0c5a2d5/mac>
- Apple Core Audio exposes read property APIs and nominal sample-rate
  properties:
  <https://developer.apple.com/documentation/coreaudio/audioobjectgetpropertydata(_:_:_:_:_:_:)>
  and
  <https://developer.apple.com/documentation/coreaudio/kaudiodevicepropertynominalsamplerate>

No live command/API is invoked in Phase 0.

