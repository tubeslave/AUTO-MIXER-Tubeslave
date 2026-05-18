# Automixer ChromaDB Project Index

Local vector index for Automixer project knowledge. It writes to ChromaDB under
`.chromadb/` by default and does not call Paperclip, WING, OSC, GitHub, or live
runtime paths.

## Index

```bash
python3 tools/project_vector_index/automixer_chromadb_index.py index
```

Default sources include project docs, `Docs/reports/tub/TUB-*.md`, runtime AI
knowledge, safe automation READMEs, selected config files, and selected
backend/replay code. Local `.env`, secrets, `config/user_config.json`, logs,
`external/`, `node_modules/`, and generated artifacts are excluded.

Optional local Paperclip reports can be included explicitly:

```bash
python3 tools/project_vector_index/automixer_chromadb_index.py index --include-paperclip-reports
```

## Search

```bash
python3 tools/project_vector_index/automixer_chromadb_index.py search "WING supervised write rollback"
```

## Add Information

```bash
python3 tools/project_vector_index/automixer_chromadb_index.py add \
  --title "Operator note" \
  --category operator_note \
  --text "Short durable fact to store in the Automixer ChromaDB index."
```

For longer notes:

```bash
python3 tools/project_vector_index/automixer_chromadb_index.py add \
  --title "Audit handoff" \
  --category audit_handoff \
  --file /path/to/report.md
```

## Sync Paperclip

Pull read-only Paperclip summaries into the same ChromaDB collection:

```bash
python3 tools/project_vector_index/automixer_chromadb_index.py paperclip-sync
```

This calls only GET endpoints: health, companies, dashboard, agents, issues, and
live-runs. It does not assign issues, wake agents, dispatch work, or update
Paperclip.

## Regular Sync

For local recurring sync runs, use the scheduler wrapper instead of a shell loop:

```bash
cp tools/project_vector_index/.env.example tools/project_vector_index/.env
python3 tools/project_vector_index/automixer_vector_paperclip_sync.py once
python3 tools/project_vector_index/automixer_vector_paperclip_sync.py status
```

The wrapper:

- reuses the existing read-only `paperclip-sync` ingestion path;
- stores state in `.paperclip/vector_paperclip_sync_state.json`;
- uses an advisory lock to avoid overlapping runs;
- logs to `logs/automixer_vector_paperclip_sync.log`;
- supports `AUTOMIXER_VECTOR_SYNC_DRY_RUN=true` for safe preview.

For a local macOS schedule, install:

```text
tools/project_vector_index/com.tubeslave.automixer-vector-paperclip-sync.plist
```

Manual install:

```bash
cp tools/project_vector_index/com.tubeslave.automixer-vector-paperclip-sync.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.tubeslave.automixer-vector-paperclip-sync.plist
launchctl enable gui/$(id -u)/com.tubeslave.automixer-vector-paperclip-sync
```

The plist runs `once` every 900 seconds. Keep Paperclip credentials in
`tools/project_vector_index/.env`, not in the plist itself.

## GPT Bridge

After indexing, the GPT control bridge exposes the collection through read-only
HTTP endpoints:

```bash
curl -s http://127.0.0.1:8788/v1/knowledge/automixer/stats
curl -s 'http://127.0.0.1:8788/v1/knowledge/automixer/search?q=WING%20rollback&limit=3'
```

## Stats

```bash
python3 tools/project_vector_index/automixer_chromadb_index.py stats
```

The manifest is written to `.chromadb/automixer_project_manifest.json`.
