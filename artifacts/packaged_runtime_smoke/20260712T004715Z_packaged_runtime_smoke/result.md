# Packaged runtime smoke

- Overall: FAIL
- Started packaged app: PASS
- Secure IPC bootstrap: PASS
- Invalid or expired token rejected: PASS
- Fresh token recovery authenticated: PASS
- Immutable offline lock reported: PASS
- Mixer connect/discovery operations blocked: PASS
- Stale session token removed on reload: FAIL
- Hardware controls hidden: FAIL
- External process-group connections: unknown
- Canonical WING log unchanged: PASS
- Console WAL unchanged: PASS
- Source binding: dirty_tree_non_clean_checkout

This artifact proves an offline packaged runtime launch only. It grants no live authority and is not macOS notarization evidence.
