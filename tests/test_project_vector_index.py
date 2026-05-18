import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.project_vector_index import automixer_chromadb_index as indexer


class FakeCollection:
    def __init__(self, records: list[indexer.SourceChunk]) -> None:
        self.records = records

    def get(self, include: list[str] | None = None) -> dict[str, Any]:
        _ = include
        return {
            "ids": [record.record_id for record in self.records],
            "documents": [record.document for record in self.records],
            "metadatas": [record.metadata for record in self.records],
        }

    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        _ = include
        query_embedding = query_embeddings[0]
        scored: list[tuple[float, indexer.SourceChunk]] = []
        for record in self.records:
            similarity = sum(a * b for a, b in zip(query_embedding, record.embedding))
            scored.append((similarity, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        top = scored[:n_results]
        return {
            "ids": [[record.record_id for _score, record in top]],
            "documents": [[record.document for _score, record in top]],
            "metadatas": [[record.metadata for _score, record in top]],
            "distances": [[max(0.0, 1.0 - score) for score, _record in top]],
        }


def test_iter_source_paths_excludes_secrets_and_external(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("project overview", encoding="utf-8")
    (tmp_path / "Docs").mkdir()
    (tmp_path / "Docs" / "ARCHITECTURE.md").write_text("architecture", encoding="utf-8")
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-secret", encoding="utf-8")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "user_config.json").write_text('{"local": true}', encoding="utf-8")
    (tmp_path / "external").mkdir()
    (tmp_path / "external" / "README.md").write_text("external", encoding="utf-8")

    paths = [path.relative_to(tmp_path).as_posix() for path in indexer.iter_source_paths(tmp_path)]

    assert "README.md" in paths
    assert "Docs/ARCHITECTURE.md" in paths
    assert ".env" not in paths
    assert "config/user_config.json" not in paths
    assert "external/README.md" not in paths


def test_redact_secrets_masks_known_token_shapes() -> None:
    text = "\n".join(
        [
            "OPENAI_API_KEY=sk-proj-abc1234567890abcdefghijklmnop",
            "PAPERCLIP_TOKEN=paperclip-token",
            "Authorization: Bearer abc.def-123",
        ]
    )

    redacted = indexer.redact_secrets(text)

    assert "sk-proj-" not in redacted
    assert "paperclip-token" not in redacted
    assert "abc.def-123" not in redacted
    assert "<redacted>" in redacted


def test_sanitize_payload_redacts_nested_paperclip_secrets() -> None:
    payload = {
        "adapterConfig": {
            "privateKeyPem": "-----BEGIN PRIVATE KEY-----\nsecret\n-----END PRIVATE KEY-----",
            "headers": {"x-openclaw-token": "token-value"},
            "command": "/usr/local/bin/codex",
        },
        "title": "safe",
    }

    sanitized = indexer.sanitize_payload(payload)

    assert sanitized["adapterConfig"]["privateKeyPem"] == "<redacted>"
    assert sanitized["adapterConfig"]["headers"]["x-openclaw-token"] == "<redacted>"
    assert sanitized["adapterConfig"]["command"] == "/usr/local/bin/codex"
    assert sanitized["title"] == "safe"


def test_build_source_chunks_adds_metadata_and_embeddings(tmp_path: Path) -> None:
    (tmp_path / "backend").mkdir()
    source = tmp_path / "backend" / "TUB-345-WING-Supervised-Write-Runbook.md"
    source.write_text("## Rollback\nReadback and rollback proof for WING writes.", encoding="utf-8")

    records = indexer.build_source_chunks(tmp_path, [source], indexed_at="2026-05-16T00:00:00+00:00")

    assert len(records) == 1
    record = records[0]
    assert record.metadata["source_path"] == "backend/TUB-345-WING-Supervised-Write-Runbook.md"
    assert record.metadata["category"] == "tub_report"
    assert record.metadata["chunk_index"] == 0
    assert len(record.embedding) == indexer.DEFAULT_EMBEDDING_DIM
    assert any(value != 0 for value in record.embedding)


def test_docs_tub_reports_keep_tub_report_category(tmp_path: Path) -> None:
    source = tmp_path / "Docs" / "reports" / "tub" / "TUB-343-WING-Write-Gate-Report.md"
    source.parent.mkdir(parents=True)
    source.write_text("Supervised write gate report.", encoding="utf-8")

    records = indexer.build_source_chunks(tmp_path, [source], indexed_at="2026-05-18T00:00:00+00:00")

    assert len(records) == 1
    assert records[0].metadata["source_path"] == "Docs/reports/tub/TUB-343-WING-Write-Gate-Report.md"
    assert records[0].metadata["category"] == "tub_report"


def test_manual_chunks_can_be_serialized() -> None:
    records = indexer.build_manual_chunks(
        "Store this note in ChromaDB.",
        title="Manual Note",
        category="operator_note",
        source="manual:test",
        indexed_at="2026-05-16T00:00:00+00:00",
    )

    payload = {
        "ids": [record.record_id for record in records],
        "documents": [record.document for record in records],
        "metadatas": [record.metadata for record in records],
    }

    assert json.loads(json.dumps(payload))["metadatas"][0]["source_type"] == "manual_note"


def test_select_paperclip_company_prefers_tub_company() -> None:
    company_id = indexer.select_paperclip_company(
        [
            {"id": "company-aut", "name": "Automixer Lab", "issuePrefix": "AUT", "issueCounter": 0},
            {"id": "company-tub", "name": "Tubeslave Automixer", "issuePrefix": "TUB", "issueCounter": 353},
        ]
    )

    assert company_id == "company-tub"


def test_paperclip_chunks_are_read_only_metadata(monkeypatch) -> None:
    routes = {
        "/api/health": {"status": "ok"},
        "/api/companies": [{"id": "company-tub", "name": "Tubeslave Automixer", "issuePrefix": "TUB"}],
        "/api/companies/company-tub/dashboard": {"tasks": {"todo": 1}},
        "/api/companies/company-tub/agents": [{"id": "agent-1", "status": "idle"}],
        "/api/companies/company-tub/issues": [{"id": "issue-1", "identifier": "TUB-1"}],
        "/api/companies/company-tub/live-runs": [],
    }
    calls = []

    def fake_get(base_url, path, *, token=None, params=None, timeout_seconds=5.0):
        calls.append((path, params))
        assert base_url == "http://paperclip.test"
        assert token is None
        return routes[path]

    monkeypatch.setattr(indexer, "paperclip_get_json", fake_get)

    records, summary = indexer.build_paperclip_chunks(
        base_url="http://paperclip.test",
        company_id=None,
        token=None,
        statuses="todo",
        limit=10,
        timeout_seconds=1.0,
        indexed_at="2026-05-16T00:00:00+00:00",
    )

    assert summary["paperclip_api_ok"] is True
    assert summary["company_id"] == "company-tub"
    assert summary["errors"] == []
    assert records
    assert all(record.metadata["source_type"] == "paperclip_api" for record in records)
    assert all(path.startswith("/api/") for path, _params in calls)


def test_paperclip_chunks_redact_secrets_in_indexed_documents(monkeypatch) -> None:
    secret = "paperclip-secret-token"
    routes = {
        "/api/health": {"status": "ok"},
        "/api/companies": [{"id": "company-tub", "name": "Tubeslave Automixer", "issuePrefix": "TUB"}],
        "/api/companies/company-tub/dashboard": {
            "adapterConfig": {
                "privateKeyPem": "-----BEGIN PRIVATE KEY-----\nsecret\n-----END PRIVATE KEY-----",
                "headers": {"x-openclaw-token": secret},
            },
            "notes": [
                "Authorization: Bearer abc.def-123",
                f"PAPERCLIP_TOKEN={secret}",
            ],
        },
        "/api/companies/company-tub/agents": [],
        "/api/companies/company-tub/issues": [],
        "/api/companies/company-tub/live-runs": [],
    }

    def fake_get(base_url, path, *, token=None, params=None, timeout_seconds=5.0):
        _ = (base_url, token, params, timeout_seconds)
        return routes[path]

    monkeypatch.setattr(indexer, "paperclip_get_json", fake_get)

    records, summary = indexer.build_paperclip_chunks(
        base_url="http://paperclip.test",
        company_id=None,
        token=secret,
        statuses="todo",
        limit=10,
        timeout_seconds=1.0,
        indexed_at="2026-05-16T00:00:00+00:00",
    )

    assert summary["errors"] == []
    combined_documents = "\n".join(record.document for record in records)
    assert secret not in combined_documents
    assert "abc.def-123" not in combined_documents
    assert "BEGIN PRIVATE KEY" not in combined_documents
    assert combined_documents.count("<redacted>") >= 3


def test_paperclip_chunk_summary_errors_redact_echoed_token(monkeypatch) -> None:
    secret = "paperclip-secret-token"

    def fake_get(base_url, path, *, token=None, params=None, timeout_seconds=5.0):
        _ = (base_url, path, params, timeout_seconds)
        raise OSError(f"upstream echoed {token}")

    monkeypatch.setattr(indexer, "paperclip_get_json", fake_get)

    records, summary = indexer.build_paperclip_chunks(
        base_url="http://paperclip.test",
        company_id="company-tub",
        token=secret,
        statuses="todo",
        limit=10,
        timeout_seconds=1.0,
        indexed_at="2026-05-16T00:00:00+00:00",
    )

    assert records == []
    assert summary["paperclip_api_ok"] is False
    assert summary["errors"]
    assert secret not in "\n".join(summary["errors"])
    assert "<redacted>" in "\n".join(summary["errors"])


def test_search_collection_prioritizes_wing_runtime_over_operator_readme(tmp_path: Path) -> None:
    readme = tmp_path / "tools" / "automixer_operator" / "README.md"
    readme.parent.mkdir(parents=True)
    readme.write_text(
        (
            "Operator examples only.\n"
            "Use vector-search with the sample phrase WING rollback readback.\n"
        )
        * 40,
        encoding="utf-8",
    )

    write_gate = tmp_path / "backend" / "TUB-343-WING-Write-Gate-Report.md"
    write_gate.parent.mkdir(parents=True)
    write_gate.write_text(
        "\n".join(
            [
                "# WING Write Gate",
                "Supervised WING write gate with approval metadata.",
                "Requires readback verification and rollback on mismatch.",
                "Keeps live console writes dry-run or disarmed until supervised.",
            ]
        ),
        encoding="utf-8",
    )

    wing_client = tmp_path / "backend" / "wing_client.py"
    wing_client.write_text(
        "\n".join(
            [
                "class WingClient:",
                "    # live console runtime safety",
                "    # throttle rate limit readback rollback supervised gate",
            ]
        ),
        encoding="utf-8",
    )

    records = indexer.build_source_chunks(tmp_path, [readme, write_gate, wing_client])
    hits = indexer.search_collection(FakeCollection(records), "WING rollback readback", limit=5)

    paths = [hit["metadata"]["source_path"] for hit in hits]
    assert paths[0] in {
        "backend/TUB-343-WING-Write-Gate-Report.md",
        "backend/wing_client.py",
    }
    assert paths.count("tools/automixer_operator/README.md") == 1
    assert paths.index("tools/automixer_operator/README.md") > 0


def test_search_collection_prioritizes_paperclip_api_sources_for_api_queries() -> None:
    paperclip_records = indexer.build_external_chunks(
        '{"endpoint": "/api/companies", "payload": [{"id": "company-tub"}]}',
        title="Paperclip companies",
        category="paperclip_api",
        source="paperclip:/api/companies",
        source_type="paperclip_api",
        indexed_at="2026-05-16T00:00:00+00:00",
    )
    wing_records = indexer.build_external_chunks(
        "WING rollback readback write gate notes",
        title="WING note",
        category="operator_note",
        source="manual:wing",
        source_type="manual_note",
        indexed_at="2026-05-16T00:00:00+00:00",
    )

    hits = indexer.search_collection(
        FakeCollection(paperclip_records + wing_records),
        "paperclip:/api/companies",
        limit=3,
    )

    assert hits[0]["metadata"]["source_path"] == "paperclip:/api/companies"
    assert hits[0]["metadata"]["source_type"] == "paperclip_api"


def test_search_collection_prioritizes_safety_gate_reports_for_safety_queries(tmp_path: Path) -> None:
    director = tmp_path / "Docs" / "PAPERCLIP_DIRECTOR_WORKFLOW.md"
    director.parent.mkdir(parents=True)
    director.write_text(
        "Director notes mention explicit live-safety rules and issue drafting.",
        encoding="utf-8",
    )

    write_gate = tmp_path / "backend" / "TUB-343-WING-Write-Gate-Report.md"
    write_gate.parent.mkdir(parents=True)
    write_gate.write_text(
        "\n".join(
            [
                "# WING Supervised Write Gate Report",
                "Safety gate for live console control.",
                "Approval, cooldown, readback, rollback, and emergency stop are required.",
            ]
        ),
        encoding="utf-8",
    )

    runtime = tmp_path / "backend" / "wing_client.py"
    runtime.write_text(
        "Wing runtime enforces throttle, write gate, cooldown, and emergency stop.",
        encoding="utf-8",
    )

    readme = tmp_path / "tools" / "automixer_operator" / "README.md"
    readme.parent.mkdir(parents=True)
    readme.write_text("Sample vector-search examples for operators.", encoding="utf-8")

    records = indexer.build_source_chunks(tmp_path, [director, write_gate, runtime, readme])
    hits = indexer.search_collection(FakeCollection(records), "WING safety gate", limit=4)

    top_paths = [hit["metadata"]["source_path"] for hit in hits[:2]]
    assert "backend/TUB-343-WING-Write-Gate-Report.md" in top_paths
    assert "backend/wing_client.py" in top_paths
