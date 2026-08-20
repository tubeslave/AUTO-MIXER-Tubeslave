import json
from pathlib import Path

import pytest

from pipeline_entrypoint_catalog import load_pipeline_entrypoint_catalog


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_catalog_separates_working_and_experimental_entrypoints():
    catalog = load_pipeline_entrypoint_catalog(PROJECT_ROOT)
    working = {row["path"] for row in catalog["working"]}
    experimental = {row["path"] for row in catalog["experimental"]}

    assert working.isdisjoint(experimental)
    assert "backend/server.py" in working
    assert "tools/wing_attended_soundcheck_runner.py" in working
    assert "tools/live_wing_soundcheck_v2.py" in experimental
    assert catalog["summary"]["experimental_live_execution_allowed"] is False


def test_duplicate_families_point_from_working_to_experimental_only():
    catalog = load_pipeline_entrypoint_catalog(PROJECT_ROOT)
    working_ids = {row["id"] for row in catalog["working"]}
    experimental_ids = {row["id"] for row in catalog["experimental"]}

    for family in catalog["duplicate_families"]:
        assert family["canonical_id"] in working_ids
        assert set(family["duplicate_ids"]) <= experimental_ids


def test_catalog_rejects_one_path_in_both_groups(tmp_path):
    (tmp_path / "backend").mkdir()
    (tmp_path / "configs/pipelines").mkdir(parents=True)
    (tmp_path / "backend/server.py").write_text("", encoding="utf-8")
    payload = {
        "schema": "automixer.pipeline_entrypoint_catalog.v1",
        "working": [{"id": "main", "path": "backend/server.py"}],
        "experimental": [{"id": "copy", "path": "backend/server.py"}],
        "duplicate_families": [],
    }
    (tmp_path / "configs/pipelines/catalog.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="path_in_multiple_groups"):
        load_pipeline_entrypoint_catalog(tmp_path)
