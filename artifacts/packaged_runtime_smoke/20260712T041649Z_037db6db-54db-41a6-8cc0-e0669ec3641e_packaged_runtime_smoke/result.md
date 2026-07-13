# Packaged runtime smoke

- Overall: PASS
- Started packaged app: PASS
- Backend ready within 20000 ms: PASS
- Secure IPC bootstrap: PASS
- Invalid or expired token rejected: PASS
- Fresh token recovery authenticated: PASS
- Immutable offline lock reported: PASS
- Mixer connect/discovery operations blocked: PASS
- Stale session token removed on reload: PASS
- Hardware controls hidden: PASS
- OS-enforced backend non-loopback network deny: PASS
- Launched copy matches signed app before/after: PASS
- External process-group connections: 0
- Continuous network samples: 24
- WING-port socket observations: 0
- Canonical WING log unchanged: PASS
- Console WAL unchanged: PASS
- Complete application tree unchanged: PASS
- Backend onedir bundle unchanged: PASS
- Temporary write artifacts absent: PASS
- Source binding: dirty_tree_non_clean_checkout
- Build profile: local
- Release eligible: false
- Build profile bound to app: PASS

This artifact proves an offline packaged runtime launch only. It grants no live authority and is not macOS notarization evidence.
