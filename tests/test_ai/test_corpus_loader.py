"""Tests for ai.corpus_loader and KnowledgeBase.load_markdown_directory."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))

from ai.knowledge_base import KnowledgeBase  # noqa: E402
from ai.corpus_loader import build_knowledge_base, _project_root_from_backend  # noqa: E402


def test_load_markdown_directory(tmp_path):
    d = tmp_path / "extra"
    d.mkdir()
    (d / "note.md").write_text("## Section A\nHello extra corpus.\n", encoding="utf-8")
    kb = KnowledgeBase(use_vector_db=False)
    before = kb.entry_count()
    kb.load_markdown_directory(str(d))
    assert kb.entry_count() > before


def test_build_knowledge_base_ingests_backup(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    (root / "presets").mkdir(parents=True)
    snap = {
        "timestamp": "2026-01-01",
        "channels": {"1": {"fader_db": -6.0, "name": "Kick"}},
    }
    (root / "presets" / "channel_backup_test.json").write_text(
        json.dumps(snap), encoding="utf-8"
    )
    cfg = {
        "ai": {
            "knowledge_dir": str(tmp_path / "empty_k"),
            "use_chromadb": False,
            "knowledge_extra_paths": [],
            "corpus_ingest": {
                "backend_log_glob": "",
                "channel_backup_glob": "presets/channel_backup_*.json",
                "mix_events_glob": "",
                "max_backup_files": 10,
            },
        }
    }
    (tmp_path / "empty_k").mkdir()
    (tmp_path / "empty_k" / "stub.md").write_text("## X\nstub\n", encoding="utf-8")
    kb = build_knowledge_base(cfg, project_root=str(root))
    cats = kb.get_categories()
    assert "channel_snapshot" in cats
    assert kb.entry_count() >= 1


def test_project_root_points_to_repo():
    pr = _project_root_from_backend()
    assert os.path.isdir(pr)
    assert os.path.isdir(os.path.join(pr, "backend", "ai", "knowledge"))
