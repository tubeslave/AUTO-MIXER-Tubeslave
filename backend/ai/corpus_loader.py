"""
Build KnowledgeBase from primary knowledge dir, extra paths, and correction artifacts.
Paths in config are resolved relative to project root (repo root).
"""
from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _project_root_from_backend() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _resolve_path(project_root: str, p: str) -> str:
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(project_root, p))


def _tail_lines(path: str, max_lines: int) -> List[str]:
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return lines[-max_lines:] if len(lines) > max_lines else lines
    except OSError as e:
        logger.warning("Could not read log %s: %s", path, e)
        return []


def _filter_correction_lines(lines: List[str]) -> List[str]:
    keys = (
        "eq",
        "auto_eq",
        "gain",
        "safe_gain",
        "fader",
        "trim",
        "compress",
        "wing",
        "osc",
        "correction",
        "lufs",
    )
    out = []
    for ln in lines:
        low = ln.lower()
        if any(k in low for k in keys):
            out.append(ln.rstrip())
    return out


def _backup_to_text(data: Any, max_channels: int = 40) -> str:
    """Flatten channel_backup JSON into short RAG-friendly text."""
    parts: List[str] = []
    if not isinstance(data, dict):
        return json.dumps(data, ensure_ascii=False)[:4000]
    ts = data.get("timestamp") or data.get("time")
    if ts:
        parts.append(f"Snapshot time: {ts}")
    chans = data.get("channels")
    if isinstance(chans, dict):
        for i, (k, v) in enumerate(chans.items()):
            if i >= max_channels:
                parts.append("… (truncated)")
                break
            if isinstance(v, dict):
                parts.append(f"Channel {k}: {json.dumps(v, ensure_ascii=False)[:500]}")
            else:
                parts.append(f"Channel {k}: {str(v)[:300]}")
    elif isinstance(chans, list):
        for i, v in enumerate(chans[:max_channels]):
            parts.append(f"Entry {i}: {json.dumps(v, ensure_ascii=False)[:500]}")
    else:
        parts.append(json.dumps(data, ensure_ascii=False)[:8000])
    return "\n".join(parts)


def build_knowledge_base(
    config: Dict[str, Any],
    project_root: Optional[str] = None,
) -> Any:
    """
    Construct a KnowledgeBase with markdown dirs + optional ingest of logs/JSON/JSONL.

    Config shape (under key ``ai``)::

        {
          "knowledge_dir": null,  # default backend/ai/knowledge
          "knowledge_extra_paths": ["../notes/mixing"],
          "use_chromadb": true,
          "corpus_ingest": {
            "backend_log_glob": "logs/automixer-backend.log",
            "channel_backup_glob": "presets/channel_backup_*.json",
            "mix_events_glob": "",
            "max_log_lines": 500,
            "max_backup_files": 20,
            "max_jsonl_rows": 200
          }
        }
    """
    from .knowledge_base import KnowledgeBase, KnowledgeEntry

    root = project_root or _project_root_from_backend()
    ai_cfg = config.get("ai") or {}
    kd = ai_cfg.get("knowledge_dir")
    if kd:
        knowledge_dir = _resolve_path(root, kd)
    else:
        knowledge_dir = os.path.join(os.path.dirname(__file__), "knowledge")

    use_chroma = ai_cfg.get("use_chromadb", True)
    kb = KnowledgeBase(knowledge_dir=knowledge_dir, use_vector_db=use_chroma)

    for rel in ai_cfg.get("knowledge_extra_paths") or []:
        abs_dir = _resolve_path(root, rel)
        if os.path.isdir(abs_dir):
            kb.load_markdown_directory(abs_dir)
        else:
            logger.warning("Extra knowledge path missing: %s", abs_dir)

    ingest = ai_cfg.get("corpus_ingest") or {}
    max_log = int(ingest.get("max_log_lines", 500))
    max_back = int(ingest.get("max_backup_files", 20))
    max_jsonl = int(ingest.get("max_jsonl_rows", 200))

    log_glob = ingest.get("backend_log_glob") or ""
    if log_glob:
        for path in glob.glob(_resolve_path(root, log_glob))[:5]:
            lines = _tail_lines(path, max_log)
            picked = _filter_correction_lines(lines)
            if not picked:
                continue
            chunk = "\n".join(picked[-400:])
            entry_id = f"correction_log_{hashlib.md5(chunk[:500].encode()).hexdigest()[:12]}"
            kb.add_entry(
                KnowledgeEntry(
                    id=entry_id,
                    content=f"## Backend log excerpt ({os.path.basename(path)})\n{chunk}",
                    category="correction_log",
                    metadata={"title": os.path.basename(path), "source": "correction_log"},
                )
            )

    backup_glob = ingest.get("channel_backup_glob") or ""
    if backup_glob:
        paths = sorted(glob.glob(_resolve_path(root, backup_glob)))[-max_back:]
        for path in paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("Skip backup %s: %s", path, e)
                continue
            text = _backup_to_text(data)
            if len(text) < 20:
                continue
            base = os.path.basename(path).replace(".json", "")
            entry_id = f"snapshot_{base}_{hashlib.md5(text[:300].encode()).hexdigest()[:8]}"
            kb.add_entry(
                KnowledgeEntry(
                    id=entry_id,
                    content=f"## Channel backup {base}\n{text}",
                    category="channel_snapshot",
                    metadata={"title": base, "source": "channel_snapshot"},
                )
            )

    jsonl_glob = ingest.get("mix_events_glob") or ""
    if jsonl_glob:
        try:
            from ml.training_dataset_io import iter_jsonl_dicts
        except ImportError:
            iter_jsonl_dicts = None  # type: ignore
        if iter_jsonl_dicts:
            for path in glob.glob(_resolve_path(root, jsonl_glob))[:3]:
                count = 0
                for row in iter_jsonl_dicts(path):
                    if count >= max_jsonl:
                        break
                    count += 1
                    line = json.dumps(row, ensure_ascii=False)[:2000]
                    eid = f"mixev_{hashlib.md5(line.encode()).hexdigest()[:12]}"
                    kb.add_entry(
                        KnowledgeEntry(
                            id=eid,
                            content=f"## Mix training event\n{line}",
                            category="mix_events",
                            metadata={"title": path, "source": "mix_events"},
                        )
                    )

    logger.info(
        "Knowledge base ready: %s entries (dir=%s)",
        kb.entry_count(),
        knowledge_dir,
    )
    return kb
