"""Validated separation of canonical and experimental pipeline entrypoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


CATALOG_SCHEMA = "automixer.pipeline_entrypoint_catalog.v1"
DEFAULT_CATALOG_PATH = Path("configs/pipelines/catalog.json")


def _records(value: Any, field: str) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError(f"pipeline_catalog_{field}_must_be_list")
    records: list[dict[str, str]] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ValueError(f"pipeline_catalog_{field}_entry_invalid")
        record = {str(key): str(item) for key, item in raw.items()}
        if not record.get("id") or not record.get("path"):
            raise ValueError(f"pipeline_catalog_{field}_entry_invalid")
        records.append(record)
    return records


def load_pipeline_entrypoint_catalog(
    repo_root: str | Path,
    catalog_path: str | Path = DEFAULT_CATALOG_PATH,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    source = root / Path(catalog_path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("schema") != CATALOG_SCHEMA:
        raise ValueError("pipeline_catalog_schema_invalid")

    working = _records(raw.get("working"), "working")
    experimental = _records(raw.get("experimental"), "experimental")
    all_records = working + experimental
    ids = [record["id"] for record in all_records]
    paths = [record["path"] for record in all_records]
    if len(ids) != len(set(ids)):
        raise ValueError("pipeline_catalog_duplicate_id")
    if len(paths) != len(set(paths)):
        raise ValueError("pipeline_catalog_path_in_multiple_groups")
    for relative in paths:
        candidate = (root / relative).resolve()
        if root not in candidate.parents or not candidate.is_file():
            raise ValueError(f"pipeline_catalog_entrypoint_missing:{relative}")

    duplicate_families = raw.get("duplicate_families")
    if not isinstance(duplicate_families, list):
        raise ValueError("pipeline_catalog_duplicate_families_must_be_list")
    working_ids = {record["id"] for record in working}
    experimental_ids = {record["id"] for record in experimental}
    for family in duplicate_families:
        if not isinstance(family, Mapping):
            raise ValueError("pipeline_catalog_duplicate_family_invalid")
        canonical_id = family.get("canonical_id")
        duplicates = family.get("duplicate_ids")
        if canonical_id not in working_ids or not isinstance(duplicates, list):
            raise ValueError("pipeline_catalog_duplicate_family_invalid")
        if not duplicates or any(item not in experimental_ids for item in duplicates):
            raise ValueError("pipeline_catalog_duplicate_family_invalid")

    return {
        "schema": CATALOG_SCHEMA,
        "working": working,
        "experimental": experimental,
        "duplicate_families": duplicate_families,
        "summary": {
            "working_count": len(working),
            "experimental_count": len(experimental),
            "duplicate_family_count": len(duplicate_families),
            "canonical_live_entrypoint": "backend/server.py",
            "experimental_live_execution_allowed": False,
        },
    }
