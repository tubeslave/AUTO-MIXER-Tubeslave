from __future__ import annotations

import json
from pathlib import Path

from tools.project_vector_index import automixer_vector_paperclip_sync as scheduler


def test_settings_load_local_env_defaults(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(scheduler, "WATCHER_DIR", tmp_path)
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "PAPERCLIP_API_URL=http://127.0.0.1:3100",
                "PAPERCLIP_COMPANY_ID=company-tub",
                "PAPERCLIP_TOKEN=secret-token",
                "AUTOMIXER_VECTOR_SYNC_INTERVAL_SECONDS=600",
                "AUTOMIXER_VECTOR_SYNC_DRY_RUN=true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("PAPERCLIP_COMPANY_ID", raising=False)
    settings = scheduler.Settings.from_env(
        {"AUTOMIXER_VECTOR_SYNC_PROJECT_ROOT": str(tmp_path)}
    )

    assert settings.project_root == tmp_path.resolve()
    assert settings.paperclip_url == "http://127.0.0.1:3100"
    assert settings.company_id == "company-tub"
    assert settings.token == "secret-token"
    assert settings.interval_seconds == 600
    assert settings.dry_run is True


def test_run_sync_once_persists_summary_without_upsert_on_dry_run(tmp_path: Path, monkeypatch) -> None:
    settings = scheduler.Settings(
        project_root=tmp_path,
        persist_dir=".chromadb",
        collection="automixer-project",
        batch_size=32,
        paperclip_url="http://paperclip.test",
        company_id="company-tub",
        token="secret-token",
        statuses="todo,in_progress",
        limit=10,
        timeout_seconds=1.0,
        interval_seconds=900,
        dry_run=True,
        log_path=tmp_path / "logs" / "vector-sync.log",
        state_path=tmp_path / ".paperclip" / "vector-sync-state.json",
        lock_path=tmp_path / ".paperclip" / "vector-sync.lock",
    )

    calls: list[tuple[str, int]] = []

    def fake_build_paperclip_chunks(**kwargs):
        return (
            [
                indexer_chunk(
                    record_id="paperclip-1",
                    document="safe document",
                    metadata={"source_type": "paperclip_api"},
                )
            ],
            {
                "paperclip_api_ok": True,
                "company_id": kwargs["company_id"],
                "errors": [],
            },
        )

    def fake_upsert_records(*args, **kwargs):
        calls.append(("upsert", kwargs["batch_size"]))

    monkeypatch.setattr(scheduler.indexer, "build_paperclip_chunks", fake_build_paperclip_chunks)
    monkeypatch.setattr(scheduler.indexer, "upsert_records", fake_upsert_records)

    result = scheduler.run_sync_once(settings)

    assert result.result == "ok"
    assert result.summary["dry_run"] is True
    assert result.summary["record_count"] == 1
    assert calls == []
    state = json.loads(settings.state_path.read_text(encoding="utf-8"))
    assert state["last_result"] == "ok"
    assert state["last_summary"]["company_id"] == "company-tub"


def test_run_sync_once_records_errors_without_leaking_token(tmp_path: Path, monkeypatch) -> None:
    settings = scheduler.Settings(
        project_root=tmp_path,
        persist_dir=".chromadb",
        collection="automixer-project",
        batch_size=32,
        paperclip_url="http://paperclip.test",
        company_id="company-tub",
        token="secret-token",
        statuses="todo",
        limit=10,
        timeout_seconds=1.0,
        interval_seconds=900,
        dry_run=False,
        log_path=tmp_path / "logs" / "vector-sync.log",
        state_path=tmp_path / ".paperclip" / "vector-sync-state.json",
        lock_path=tmp_path / ".paperclip" / "vector-sync.lock",
    )

    def fake_build_paperclip_chunks(**kwargs):
        raise RuntimeError(f"upstream echoed {kwargs['token']}")

    monkeypatch.setattr(scheduler.indexer, "build_paperclip_chunks", fake_build_paperclip_chunks)

    result = scheduler.run_sync_once(settings)

    assert result.result == "error"
    assert "secret-token" not in json.dumps(result.summary, ensure_ascii=False)
    assert "<redacted>" in json.dumps(result.summary, ensure_ascii=False)
    log_text = settings.log_path.read_text(encoding="utf-8")
    assert "secret-token" not in log_text
    assert "<redacted>" in log_text


def indexer_chunk(*, record_id: str, document: str, metadata: dict) -> scheduler.indexer.SourceChunk:
    return scheduler.indexer.SourceChunk(
        record_id=record_id,
        document=document,
        metadata=metadata,
        embedding=[0.0] * scheduler.indexer.DEFAULT_EMBEDDING_DIM,
    )
