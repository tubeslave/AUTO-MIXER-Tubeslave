"""Service for periodic/triggered online training of ML models."""

from __future__ import annotations

import asyncio
import csv
import gzip
import hashlib
import html
import os
import json
import logging
import tempfile
import shutil
import tarfile
import zipfile
import re
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
from urllib.parse import quote_plus, quote, urlparse
import xml.etree.ElementTree as ET

import requests

from .train_classifier import train_classifier
from .train_gain_predictor import train_gain_predictor
from .train_mix_console import train_mix_console

logger = logging.getLogger(__name__)


class _StudyHTMLTextExtractor(HTMLParser):
    """Simple HTML -> text converter used for lightweight source ingestion."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag.lower() in {"script", "style", "noscript"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        self._chunks.append(data)

    def get_text(self) -> str:
        text = " ".join(part.strip() for part in self._chunks if part.strip())
        return html.unescape(re.sub(r"\s+", " ", text)).strip()


DEFAULT_DISCOVERY_QUERIES = {
    "channel_classifier": [
        "audio instrument classification dataset jsonl",
        "music spectral features instrument",
        "audio events melspectrogram classification",
    ],
    "gain_pan_predictor": [
        "multichannel audio gains pans dataset npz",
        "music mix parameters gain pan npz",
        "source separation gain pan metadata",
    ],
    "mix_console": [
        "multitrack music stems dataset",
        "audio multitrack reference mix dataset",
        "music source separation multitracks",
    ],
}

DEFAULT_STUDY_QUERIES = {
    "books": [
        "audio engineering handbook",
        "mixing and mastering reference",
        "digital signal processing audio",
    ],
    "articles": [
        "audio mixing article",
        "digital signal processing article",
        "music production research",
    ],
    "videos": [
        "audio mixing tutorial",
        "recording engineering class",
    ],
}

DEFAULT_STUDY_SOURCE_TYPES = ("books", "articles", "videos")
DEFAULT_STUDY_DOMAINS = [
    "arxiv.org",
    "openlibrary.org",
    "archive.org",
    "wikipedia.org",
    "ieee.org",
    "acm.org",
    "mdpi.com",
    "nature.com",
]

TARGET_DATASET_EXTENSIONS = {
    "channel_classifier": {".jsonl", ".npz", ".json", ".csv"},
    "gain_pan_predictor": {".jsonl", ".npz", ".json", ".csv"},
    "mix_console": {".jsonl", ".npz", ".json", ".csv"},
}

TARGET_FILE_PATTERNS = {
    "channel_classifier": ("spectrogram", "features", "spectral"),
    "gain_pan_predictor": ("channels", "gains", "pans"),
    "mix_console": ("multitracks", "references", "channels"),
}
TARGET_FILE_HINT_KEYWORDS = {
    "channel_classifier": (
        "spectrogram",
        "melspec",
        "mel",
        "feature",
        "instrument",
        "class",
    ),
    "gain_pan_predictor": (
        "gain",
        "pan",
        "channel",
        "multichannel",
        "stems",
    ),
    "mix_console": (
        "multitrack",
        "multitracks",
        "reference",
        "mix",
        "stems",
    ),
}


class AgentTrainingService:
    """Background trainer for mixer ML models with optional internet datasets."""

    def __init__(self, config: Optional[Dict[str, Any]], repo_root: str):
        self.config = self._normalize_config(config or {})
        self.repo_root = Path(repo_root)
        self.artifacts_dir = self._abs_path(self.config["artifacts_dir"])
        self.dataset_dir = self._abs_path(self.config["dataset_dir"])
        self.state_file = self._abs_path(self.config["state_file"])
        self.study_dir = self._abs_path(self.config["study"]["output_dir"])

        self._state = self._load_state()
        self._running_task: Optional[asyncio.Task[None]] = None
        self._active_task: Optional[asyncio.Task[None]] = None
        self._stop_requested = False
        self._run_in_progress = False
        self._run_count = int(self._state.get("run_count", 0))
        self._last_success = self._state.get("last_success")
        self._last_error = self._state.get("last_error")
        self._next_run_at: Optional[datetime] = None
        self._run_download_paths: set[Path] = set()

    def _abs_path(self, path: str) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return (self.repo_root / p).resolve()

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return default

    @staticmethod
    def _safe_list(value: Any, default: List[Any]) -> List[Any]:
        if isinstance(value, list):
            return value
        if value is None:
            return list(default)
        return [value]

    @staticmethod
    def _safe_dict(value: Any, default: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        return dict(default)

    def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        discovery_config = self._safe_dict(config.get("discovery"), {})
        discovery_enabled = self._safe_bool(discovery_config.get("enabled"), False)
        discovery_queries = self._safe_dict(discovery_config.get("queries"), DEFAULT_DISCOVERY_QUERIES)
        merged_queries = {}
        for target_name, query in DEFAULT_DISCOVERY_QUERIES.items():
            merged_queries[target_name] = []
            for item in self._safe_list(discovery_queries.get(target_name), []):
                if isinstance(item, str) and item.strip():
                    merged_queries[target_name].append(item.strip())
            if not merged_queries[target_name]:
                merged_queries[target_name] = list(query)

        study_config = self._safe_dict(config.get("study"), {})
        study_enabled = self._safe_bool(study_config.get("enabled"), False)
        study_queries = self._safe_dict(study_config.get("queries"), DEFAULT_STUDY_QUERIES)
        merged_study_queries = {}
        for source_name, query_list in DEFAULT_STUDY_QUERIES.items():
            merged_study_queries[source_name] = []
            for item in self._safe_list(study_queries.get(source_name), []):
                if isinstance(item, str) and item.strip():
                    merged_study_queries[source_name].append(item.strip())
            if not merged_study_queries[source_name]:
                merged_study_queries[source_name] = list(query_list)
        normalized_source_types = [
            item.lower()
            for item in self._safe_list(study_config.get("source_types"), list(DEFAULT_STUDY_SOURCE_TYPES))
            if isinstance(item, str) and item.strip()
        ]
        normalized_source_types = [
            item for item in normalized_source_types
            if item in DEFAULT_STUDY_SOURCE_TYPES
        ]
        if not normalized_source_types:
            normalized_source_types = list(DEFAULT_STUDY_SOURCE_TYPES)

        default_targets = {
            "channel_classifier": {
                "enabled": False,
                "output_path": "models/channel_classifier.pt",
                "n_epochs": 20,
                "batch_size": 32,
                "lr": 1e-3,
                "n_samples_per_class": 100,
                "n_mels": 64,
                "n_frames": 64,
                "n_channels": 8,
                "device": "cpu",
                "train_without_dataset": False,
            },
            "gain_pan_predictor": {
                "enabled": False,
                "output_path": "models/gain_pan_predictor.pt",
                "n_epochs": 30,
                "batch_size": 16,
                "lr": 1e-3,
                "n_channels": 8,
                "n_samples": 500,
                "n_audio_len": 8192,
                "device": "cpu",
                "train_without_dataset": False,
            },
            "mix_console": {
                "enabled": False,
                "output_path": "models/mix_console.pt",
                "n_epochs": 30,
                "lr": 1e-2,
                "n_channels": 8,
                "audio_len": 16384,
                "n_samples": 200,
                "sample_rate": 48000,
                "device": "cpu",
                "train_without_dataset": False,
            },
        }

        interval_minutes = self._safe_float(config.get("interval_minutes"), 360.0)
        request_timeout = self._safe_float(config.get("request_timeout_sec"), 30.0)
        max_dataset_bytes = self._safe_int(config.get("max_dataset_bytes"), 30 * 1024 * 1024)
        merged = {
            "enabled": bool(config.get("enabled", False)),
            "safe_autostart": self._safe_bool(config.get("safe_autostart"), False),
            "interval_seconds": max(60, int(interval_minutes * 60)),
            "manifest_url": config.get("manifest_url"),
            "request_timeout_sec": max(5, request_timeout),
            "max_retries": self._safe_int(config.get("max_retries"), 0),
            "max_dataset_bytes": max(1024 * 1024, max_dataset_bytes),
            "cleanup_downloads": self._safe_bool(config.get("cleanup_downloads"), True),
            "study": {
                "enabled": study_enabled,
                "max_resources_per_run": self._safe_int(study_config.get("max_resources_per_run"), 3),
                "max_resources_per_type": self._safe_int(study_config.get("max_resources_per_type"), 2),
                "max_resource_bytes": self._safe_int(study_config.get("max_resource_bytes"), 2 * 1024 * 1024),
                "timeout_sec": self._safe_float(study_config.get("timeout_sec"), 30.0),
                "max_retries": self._safe_int(study_config.get("max_retries"), 0),
                "output_dir": study_config.get("output_dir", os.path.join("backend", "ai", "knowledge")),
                "source_types": normalized_source_types,
                "allowed_domains": self._safe_list(
                    study_config.get("allowed_domains"),
                    DEFAULT_STUDY_DOMAINS,
                ),
                "queries": merged_study_queries,
                "resource_urls": self._safe_list(study_config.get("resource_urls"), []),
                "feeds": self._safe_list(study_config.get("feeds"), []),
            },
            "discovery": {
                "enabled": discovery_enabled,
                "queries": merged_queries,
                "max_candidates_per_target": self._safe_int(
                    discovery_config.get("max_candidates_per_target"),
                    3,
                ),
                "hf_search_limit": self._safe_int(discovery_config.get("hf_search_limit"), 8),
                "candidate_urls": self._safe_list(discovery_config.get("candidate_urls"), []),
            },
            "artifacts_dir": config.get("artifacts_dir", "models"),
            "dataset_dir": config.get("dataset_dir", os.path.join("models", "training_datasets")),
            "state_file": config.get("state_file", os.path.join("models", "training_state.json")),
            "targets": {},
        }
        user_targets = config.get("targets") if isinstance(config.get("targets"), dict) else {}
        for target_name, target_defaults in default_targets.items():
            merged_target = dict(target_defaults)
            merged_target.update(user_targets.get(target_name, {}))
            merged["targets"][target_name] = merged_target
        return merged

    def _load_state(self) -> Dict[str, Any]:
        data = {
            "version": "1.1",
            "run_count": 0,
            "last_run": None,
            "safe_autostart_done": False,
            "last_success": None,
            "last_error": None,
            "targets": {
                "channel_classifier": {},
                "gain_pan_predictor": {},
                "mix_console": {},
            },
        }
        if not self.state_file.exists():
            return data
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                data.update(loaded)
        except Exception as exc:
            logger.warning("Failed to load training state from %s: %s", self.state_file, exc)
        return data

    def _save_state(self) -> None:
        payload = dict(self._state)
        payload["run_count"] = self._run_count
        payload["last_success"] = self._last_success
        payload["last_error"] = self._last_error
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".tmp", encoding="utf-8") as tmp:
            tmp.write(json.dumps(payload, indent=2, ensure_ascii=False))
            tmp.flush()
            tmp_path = Path(tmp.name)
        tmp_path.replace(self.state_file)

    def get_status(self) -> Dict[str, Any]:
        next_run_at = self._next_run_at
        if isinstance(next_run_at, datetime):
            next_run_at = next_run_at.isoformat()
        return {
            "enabled": self.config["enabled"],
            "running": self._run_in_progress,
            "scheduler_running": bool(self._running_task and not self._running_task.done()),
            "safe_autostart": {
                "enabled": self.config["safe_autostart"],
                "dry_run_completed": bool(self._state.get("safe_autostart_done", False)),
            },
            "last_run": self._state.get("last_run"),
            "run_count": self._run_count,
            "last_success": self._last_success,
            "last_error": self._last_error,
            "next_run_at": next_run_at,
            "targets": self._state.get("targets", {}),
        }

    async def start_scheduler(self) -> None:
        if not self.config["enabled"]:
            logger.info("Online training scheduler is disabled by config")
            return
        if self._running_task and not self._running_task.done():
            return
        self._stop_requested = False
        self._next_run_at = datetime.now(timezone.utc) + timedelta(seconds=float(self.config["interval_seconds"]))
        self._running_task = asyncio.create_task(self._scheduler_loop())

    async def stop(self) -> None:
        self._stop_requested = True

        for task_key in ("_active_task", "_running_task"):
            task = getattr(self, task_key)
            if task is None or task.done():
                setattr(self, task_key, None)
                continue
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Error while stopping %s", task_key)
            setattr(self, task_key, None)

        self._run_in_progress = False

    async def start_once(
        self,
        force: bool = False,
        reason: str = "manual",
        manifest_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self._active_task and not self._active_task.done():
            return {"started": False, "reason": "training already running"}

        run_id = uuid4().hex[:12]
        self._active_task = asyncio.create_task(
            self._run_once(
                run_id=run_id,
                force=force,
                reason=reason,
                manifest_url=manifest_url,
            )
        )
        return {
            "started": True,
            "run_id": run_id,
            "reason": reason,
        }

    async def _scheduler_loop(self) -> None:
        interval = float(self.config["interval_seconds"])
        self._next_run_at = datetime.now(timezone.utc) + timedelta(seconds=interval)
        while not self._stop_requested:
            try:
                now = datetime.now(timezone.utc)
                if self._next_run_at and now >= self._next_run_at:
                    await self.start_once(force=False, reason="scheduler")
                    if self._active_task:
                        await self._active_task
                    self._next_run_at = datetime.now(timezone.utc) + timedelta(seconds=interval)
                await asyncio.sleep(1)
            except Exception as exc:
                logger.error("Scheduler tick error: %s", exc)
                self._last_error = f"scheduler_error:{exc}"
                self._state["last_error"] = self._last_error
                self._save_state()
                await asyncio.sleep(interval)
                self._next_run_at = datetime.now(timezone.utc) + timedelta(seconds=interval)

    async def _run_once(
        self,
        run_id: str,
        force: bool,
        reason: str,
        manifest_url: Optional[str] = None,
    ) -> None:
        if self._run_in_progress:
            return

        started = datetime.now(timezone.utc).isoformat()
        self._run_in_progress = True
        self._run_count += 1
        self._run_download_paths = set()
        should_dry_run = (
            self.config["safe_autostart"]
            and not bool(self._state.get("safe_autostart_done", False))
            and not force
        )
        study_results: Dict[str, Any] = {
            "enabled": self.config["study"].get("enabled", False),
            "status": "not_started",
            "selected": [],
            "checked": [],
        }
        self._state["last_run"] = {
            "run_id": run_id,
            "reason": reason,
            "started": started,
            "safe_autostart": self.config["safe_autostart"],
            "dry_run": should_dry_run,
            "limits": {
                "max_dataset_bytes": self.config["max_dataset_bytes"],
                "max_candidates_per_target": self.config["discovery"]["max_candidates_per_target"],
                "hf_search_limit": self.config["discovery"]["hf_search_limit"],
                "study": {
                    "max_resources_per_run": self.config["study"]["max_resources_per_run"],
                    "max_resources_per_type": self.config["study"]["max_resources_per_type"],
                    "max_resource_bytes": self.config["study"]["max_resource_bytes"],
                    "source_types": self.config["study"]["source_types"],
                },
            },
        }

        results: Dict[str, Any] = {}
        all_ok = True
        try:
            manifest = await self._fetch_manifest_with_retries(manifest_url=manifest_url)
            try:
                study_results = self._run_study(manifest=manifest, dry_run=should_dry_run)
            except Exception as exc:
                logger.exception("Study resource ingestion failed")
                study_results = {
                    "enabled": self.config["study"].get("enabled", False),
                    "status": "error",
                    "error": str(exc),
                    "selected": [],
                    "checked": [],
                }
                all_ok = False

            for target_name, target in self.config["targets"].items():
                target_state = self._state.setdefault("targets", {}).setdefault(target_name, {})
                previous_dataset_id = target_state.get("dataset_id")
                try:
                    merged = dict(target)
                    if manifest and isinstance(manifest.get("targets"), dict):
                        merged.update(manifest.get("targets", {}).get(target_name, {}))

                    if manifest and not merged.get("dataset_id"):
                        merged["dataset_id"] = manifest.get("dataset_id") or manifest.get("version")

                    if not merged.get("enabled", False):
                        results[target_name] = {"skipped": True, "reason": "disabled"}
                        continue

                    dataset_path, dataset_id, changed, _, resolve_info = await self._resolve_dataset(
                        target_name=target_name,
                        cfg=merged,
                        previous_dataset_id=previous_dataset_id,
                        dry_run=should_dry_run,
                    )
                    dataset_version = dataset_id or f"synthetic:{target_name}"

                    if should_dry_run:
                        if resolve_info.get("selected"):
                            logger.info(
                                "Dry-run dataset selection for %s: %s",
                                target_name,
                                resolve_info["selected"],
                            )
                        else:
                            logger.info(
                                "Dry-run dataset selection for %s: no candidate selected",
                                target_name,
                            )
                        results[target_name] = {
                            "skipped": True,
                            "status": "dry_run",
                            "reason": "safe_autostart_first_run",
                            "selection": resolve_info,
                        }
                        target_state["last_status"] = "dry_run"
                        if dataset_id:
                            target_state["dataset_id"] = dataset_id
                        continue

                    if (
                        not force
                        and not changed
                        and target_state.get("last_status") == "ok"
                        and previous_dataset_id == dataset_version
                    ):
                        results[target_name] = {"skipped": True, "reason": "dataset unchanged"}
                        continue

                    if dataset_path is None and not merged.get("train_without_dataset", False):
                        results[target_name] = {"skipped": True, "reason": "no dataset and training disabled"}
                        continue

                    await asyncio.to_thread(
                        self._train_target,
                        target_name=target_name,
                        config=merged,
                        dataset_path=dataset_path,
                        dataset_id=dataset_version,
                    )

                    target_state["last_status"] = "ok"
                    target_state["dataset_id"] = dataset_version
                    target_state["trained_at"] = datetime.now(timezone.utc).isoformat()
                    target_state["output_path"] = merged["output_path"]
                    results[target_name] = {"skipped": False, "status": "ok"}
                except Exception as exc:
                    logger.exception("Training target %s failed", target_name)
                    target_state["last_status"] = "error"
                    target_state["last_error"] = str(exc)
                    results[target_name] = {"skipped": False, "status": "error", "error": str(exc)}
                    all_ok = False
        finally:
            self._cleanup_downloads()
            self._run_in_progress = False
            if should_dry_run:
                self._state["safe_autostart_done"] = True
                self._state["last_run"]["dry_run_completed"] = True
            self._state["targets"] = self._state.get("targets", {})
            self._state["last_run"]["study"] = study_results
            self._state["last_run"]["completed"] = datetime.now(timezone.utc).isoformat()
            self._state["last_run"]["results"] = results
            self._state["last_run"]["all_ok"] = all_ok
            if should_dry_run:
                self._last_error = "safe_autostart_dry_run"
            elif all_ok:
                self._last_success = self._state["last_run"]["completed"]
                self._last_error = None
            else:
                self._last_error = "see target status"
            self._state["last_error"] = self._last_error
            self._save_state()

    def _run_study(self, manifest: Optional[Dict[str, Any]], dry_run: bool = False) -> Dict[str, Any]:
        study_cfg = self.config["study"]
        if not study_cfg.get("enabled", False):
            return {"enabled": False, "status": "disabled", "selected": [], "checked": []}

        max_resources = self._safe_int(study_cfg.get("max_resources_per_run"), 0)
        max_resources_per_type = self._safe_int(study_cfg.get("max_resources_per_type"), 0)
        max_bytes = self._safe_int(study_cfg.get("max_resource_bytes"), 2 * 1024 * 1024)
        timeout = self._safe_float(study_cfg.get("timeout_sec"), self.config["request_timeout_sec"])
        candidates = self._collect_study_candidates(manifest=manifest, max_per_type=max_resources_per_type)

        if not candidates:
            return {
                "enabled": True,
                "status": "no_candidates",
                "checked": [],
                "selected": [],
                "max_resources": max_resources,
            }

        checked: List[Dict[str, Any]] = []
        selected: List[Dict[str, Any]] = []
        per_type_counter: Dict[str, int] = {}

        for candidate in candidates:
            if max_resources > 0 and len(selected) >= max_resources:
                checked.append({"status": "skipped", "reason": "max_resources_per_run_reached", "candidate": candidate})
                continue

            source_type = str(candidate.get("source_type") or "").lower() or "articles"
            if source_type not in set(study_cfg.get("source_types") or []):
                checked.append({"status": "skipped", "reason": f"source_type_disabled:{source_type}", "candidate": candidate})
                continue

            if source_type not in per_type_counter:
                per_type_counter[source_type] = 0
            if max_resources_per_type and per_type_counter[source_type] >= max_resources_per_type:
                checked.append({"status": "skipped", "reason": f"max_per_type_reached:{source_type}", "candidate": candidate})
                continue

            url = candidate.get("url")
            if not isinstance(url, str) or not url.strip():
                checked.append({"status": "skipped", "reason": "invalid_url", "candidate": candidate})
                continue
            url = url.strip()

            if not self._is_authoritative_domain(url):
                checked.append({"status": "rejected", "reason": "domain_not_allowed", "url": url, "source_type": source_type})
                continue

            checked_record: Dict[str, Any] = {
                "source_type": source_type,
                "source": candidate.get("source"),
                "title": candidate.get("title"),
                "url": url,
                "status": "considered",
            }
            checked.append(checked_record)

            if dry_run:
                checked_record["status"] = "dry_run_selected"
                checked_record["reason"] = "safe_autostart_dry_run"
                if max_resources_per_type and per_type_counter[source_type] >= max_resources_per_type:
                    checked_record["status"] = "skipped"
                    checked_record["reason"] = f"max_per_type_reached:{source_type}"
                    continue
                if max_resources > 0 and len(selected) >= max_resources:
                    checked_record["status"] = "skipped"
                    checked_record["reason"] = "max_resources_per_run_reached"
                    continue
                per_type_counter[source_type] += 1
                selected.append(checked_record)
                continue

            try:
                path = self._write_study_markdown(candidate, max_bytes=max_bytes, timeout=timeout)
                checked_record["path"] = str(path)
                per_type_counter[source_type] += 1
                checked_record["status"] = "selected"
                selected.append(checked_record)
                logger.info("Selected study resource for ingestion: %s", path)
            except Exception as exc:
                logger.warning("Failed to ingest study resource %s: %s", url, exc)
                checked_record = checked[-1]
                checked_record["status"] = "failed"
                checked_record["reason"] = str(exc)

        return {
            "enabled": True,
            "status": "completed",
            "max_candidates": len(candidates),
            "selected_count": len(selected),
            "selected": selected,
            "checked": checked,
            "max_resources": max_resources,
        }

    def _collect_study_candidates(
        self,
        manifest: Optional[Dict[str, Any]],
        max_per_type: int,
    ) -> List[Dict[str, Any]]:
        study_cfg = self.config["study"]
        source_types = self._safe_list(study_cfg.get("source_types"), list(DEFAULT_STUDY_SOURCE_TYPES))
        candidates: List[Dict[str, Any]] = []
        seen: set[str] = set()
        per_type_cap = max_per_type if max_per_type > 0 else 200

        for source_type in source_types:
            norm_source = str(source_type).lower().strip()
            if norm_source not in DEFAULT_STUDY_SOURCE_TYPES:
                continue
            queries = self._safe_list(study_cfg.get("queries", {}).get(norm_source, DEFAULT_STUDY_QUERIES.get(norm_source, [])), [])
            source_limit = per_type_cap
            for query in queries:
                for candidate in self._discover_study_resources(norm_source, str(query), source_limit):
                    candidate_url = (candidate.get("url") or "").strip() if isinstance(candidate.get("url"), str) else ""
                    if not candidate_url or candidate_url in seen:
                        continue
                    candidate["source_type"] = norm_source
                    candidates.append(candidate)
                    seen.add(candidate_url)

        for entry in self._iter_study_manifest_resources(manifest=manifest):
            url = (entry.get("url") or "").strip() if isinstance(entry.get("url"), str) else ""
            source_type = str(entry.get("source_type") or entry.get("type") or "articles").lower()
            if source_type == "video":
                source_type = "videos"
            if source_type == "article":
                source_type = "articles"
            if source_type == "book":
                source_type = "books"
            if source_type not in set(source_types):
                continue
            if url and url not in seen:
                entry["source_type"] = source_type
                candidates.append(entry)
                seen.add(url)

        for item in self._safe_list(self.config["study"].get("resource_urls"), []):
            if not isinstance(item, str) or not item.strip():
                continue
            guessed_type = str(source_types[0]) if source_types else "articles"
            if item not in seen:
                candidates.append({"url": item, "source_type": guessed_type, "source": "configured"})
                seen.add(item)

        for feed_item in self._safe_list(study_cfg.get("feeds"), []):
            if not isinstance(feed_item, str) or not feed_item.strip():
                continue
            for candidate in self._discover_feed_items(feed_item.strip(), per_type_cap):
                candidate_url = candidate.get("url")
                if not isinstance(candidate_url, str) or not candidate_url or candidate_url in seen:
                    continue
                guessed_type = str(source_types[0]) if source_types else "articles"
                source_type = str(candidate.get("source_type") or guessed_type)
                if source_type not in set(source_types):
                    continue
                candidate["source_type"] = source_type
                candidates.append(candidate)
                seen.add(candidate_url)

        return candidates

    def _iter_study_manifest_resources(self, manifest: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not manifest or not isinstance(manifest, dict):
            return []
        items = manifest.get("learning_resources")
        if items is None:
            items = manifest.get("study_resources")
        if not isinstance(items, list):
            return []
        parsed: List[Dict[str, Any]] = []
        for item in items:
            if isinstance(item, str):
                parsed.append({"url": item, "source": "manifest"})
                continue
            if not isinstance(item, dict):
                continue
            parsed_entry: Dict[str, Any] = {
                "source": "manifest",
            }
            for key in ("url", "title", "summary", "source_type", "type", "source"):
                if key in item:
                    parsed_entry[key] = item[key]
            if not parsed_entry.get("url"):
                continue
            parsed.append(parsed_entry)
        return parsed

    def _discover_study_resources(self, source_type: str, query: str, limit: int) -> List[Dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            return []
        if source_type == "books":
            return self._discover_books(query.strip(), limit)
        if source_type == "articles":
            return self._discover_articles(query.strip(), limit)
        if source_type == "videos":
            return self._discover_videos(query.strip(), limit)
        return []

    def _discover_books(self, query: str, limit: int) -> List[Dict[str, Any]]:
        timeout = self.config["request_timeout_sec"]
        url = f"https://openlibrary.org/search.json?q={quote_plus(query)}&limit={max(1, min(10, limit))}"
        candidates: List[Dict[str, Any]] = []
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            for item in payload.get("docs", [])[:limit]:
                key = item.get("key")
                if not key:
                    continue
                title = item.get("title") or "openlibrary book"
                authors = ", ".join(item.get("author_name", [])[:2]) if item.get("author_name") else ""
                year = item.get("first_publish_year")
                summary = item.get("subtitle") or item.get("title_suggest") or ""
                if authors and not summary:
                    summary = f"Authors: {authors}"
                if year:
                    summary = f"{summary} ({year})".strip()
                candidates.append({
                    "url": f"https://openlibrary.org{key}",
                    "title": title,
                    "summary": summary,
                    "source": "openlibrary",
                })
        except Exception as exc:
            logger.debug("Book discovery failed for query=%s: %s", query, exc, exc_info=True)
        return candidates

    def _discover_articles(self, query: str, limit: int) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        candidates.extend(self._discover_arxiv_articles(query, limit))
        if len(candidates) < limit:
            candidates.extend(self._discover_crossref_articles(query, limit - len(candidates)))
        return candidates[:limit]

    def _discover_arxiv_articles(self, query: str, limit: int) -> List[Dict[str, Any]]:
        timeout = self.config["request_timeout_sec"]
        max_results = max(1, limit)
        query_payload = quote_plus(query)
        url = (
            "https://export.arxiv.org/api/query?search_query=all:"
            + query_payload
            + "&start=0&max_results="
            + str(max_results)
        )
        candidates: List[Dict[str, Any]] = []
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            root = ET.fromstring(response.text)
            ns = {"a": "http://www.w3.org/2005/Atom"}
            for entry in root.findall(".//a:entry", ns):
                entry_id = entry.findtext("a:id", default="", namespaces=ns).strip()
                title = entry.findtext("a:title", default="arXiv article", namespaces=ns).strip()
                summary = self._clean_html_like_text(
                    entry.findtext("a:summary", default="", namespaces=ns).strip()
                )
                if not entry_id:
                    continue
                candidates.append({
                    "url": entry_id,
                    "title": title,
                    "summary": summary,
                    "source": "arxiv",
                })
                if len(candidates) >= limit:
                    break
        except Exception as exc:
            logger.debug("arXiv discovery failed for query=%s: %s", query, exc, exc_info=True)
        return candidates

    def _discover_crossref_articles(self, query: str, limit: int) -> List[Dict[str, Any]]:
        timeout = self.config["request_timeout_sec"]
        url = (
            "https://api.crossref.org/works?query="
            + quote_plus(query)
            + "&rows="
            + str(max(1, limit))
        )
        candidates: List[Dict[str, Any]] = []
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            items = payload.get("message", {}).get("items", [])
            for item in items:
                item_url = item.get("URL")
                if not item_url:
                    continue
                title_items = item.get("title", [])
                title = title_items[0] if isinstance(title_items, list) and title_items else str(item_url)
                summary = self._clean_html_like_text(item.get("abstract", ""))
                candidates.append({
                    "url": item_url,
                    "title": title,
                    "summary": summary,
                    "source": "crossref",
                })
                if len(candidates) >= limit:
                    break
        except Exception as exc:
            logger.debug("CrossRef discovery failed for query=%s: %s", query, exc, exc_info=True)
        return candidates

    def _discover_videos(self, query: str, limit: int) -> List[Dict[str, Any]]:
        timeout = self.config["request_timeout_sec"]
        limit = max(1, limit)
        search_query = quote_plus(f"({query}) AND mediatype:movies")
        url = (
            "https://archive.org/advancedsearch.php?q="
            + search_query
            + "&fl[]=identifier&fl[]=title&fl[]=description&rows="
            + str(limit)
            + "&output=json"
        )
        candidates: List[Dict[str, Any]] = []
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            for doc in payload.get("response", {}).get("docs", [])[:limit]:
                identifier = doc.get("identifier")
                if not identifier:
                    continue
                title = doc.get("title", "")
                summary = doc.get("description", "") or ""
                candidates.append({
                    "url": f"https://archive.org/details/{identifier}",
                    "title": title or identifier,
                    "summary": self._clean_html_like_text(summary),
                    "source": "archive.org",
                })
        except Exception as exc:
            logger.debug("Archive.org video discovery failed for query=%s: %s", query, exc, exc_info=True)
        return candidates

    def _discover_feed_items(self, feed_url: str, limit: int) -> List[Dict[str, Any]]:
        timeout = self.config["request_timeout_sec"]
        candidates: List[Dict[str, Any]] = []
        try:
            response = requests.get(feed_url, timeout=timeout)
            response.raise_for_status()
            root = ET.fromstring(response.text)
            items = root.findall(".//item")
            if not items:
                items = root.findall(".//{http://www.w3.org/2005/Atom}entry")
            for item in items[:max(1, limit)]:
                link_el = item.find("link")
                if link_el is None:
                    link_el = item.find("{http://www.w3.org/2005/Atom}link")
                if link_el is None or not link_el.text:
                    if link_el is not None and not link_el.text and link_el.attrib.get("href"):
                        link = link_el.attrib.get("href")
                    else:
                        continue
                else:
                    link = link_el.text
                title_el = item.find("title")
                if title_el is None:
                    title_el = item.find("{http://www.w3.org/2005/Atom}title")
                title = (title_el.text if title_el is not None else "feed item") or "feed item"
                description = ""
                summary_el = item.find("description")
                if summary_el is None:
                    summary_el = item.find("summary")
                if summary_el is not None:
                    description = summary_el.text or ""
                candidates.append({
                    "source": f"feed:{urlparse(feed_url).netloc}",
                    "url": link,
                    "title": title,
                    "summary": self._clean_html_like_text(description),
                })
        except Exception as exc:
            logger.debug("Feed discovery failed for %s: %s", feed_url, exc, exc_info=True)
        return candidates

    def _is_authoritative_domain(self, url: str) -> bool:
        study_cfg = self.config["study"]
        allowed_domains = set(
            d.lower().lstrip(".")
            for d in self._safe_list(study_cfg.get("allowed_domains"), [])
            if isinstance(d, str) and d.strip()
        )
        if not allowed_domains:
            return True
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        if not host:
            return False
        for domain in allowed_domains:
            if host == domain or host.endswith(f".{domain}"):
                return True
        return False

    def _clean_html_like_text(self, text: str) -> str:
        if not text:
            return ""
        cleaned = re.sub(r"<[^>]+>", " ", text)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def _write_study_markdown(self, candidate: Dict[str, Any], max_bytes: int, timeout: float) -> Path:
        source_type = str(candidate.get("source_type") or "articles")
        title = str(candidate.get("title") or candidate.get("url") or "resource")
        source = str(candidate.get("source") or "unknown")
        url = str(candidate.get("url") or "").strip()
        if not url:
            raise ValueError("missing url")
        if source_type not in DEFAULT_STUDY_SOURCE_TYPES:
            source_type = "articles"

        existing_id = candidate.get("resource_id")
        existing_name = None
        if existing_id:
            for path in self.study_dir.glob("*.md"):
                if existing_id in path.name:
                    existing_name = path
                    break
        if existing_name is not None:
            return existing_name

        text = ""
        summary = str(candidate.get("summary") or "").strip()
        try:
            text = self._fetch_study_text(url=url, timeout=float(timeout), max_bytes=max_bytes)
        except Exception as exc:
            if not summary:
                raise
            logger.warning("Failed to fetch full study text for %s: %s. Falling back to summary.", url, exc)
            text = summary

        if not text:
            if summary:
                text = summary
            else:
                raise ValueError("empty study content")

        self.study_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
        safe_title = re.sub(r"[^a-zA-Z0-9._-]+", "_", title.lower())[:40].strip("_") or "resource"
        filename = f"{source_type}_{safe_title}_{digest}.md"
        path = self.study_dir / filename
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                current = f.read()
            if candidate.get("url") and candidate["url"] in current:
                return path

        with open(path, "w", encoding="utf-8") as f:
            f.write(f"<!-- source_type: {source_type} -->\n")
            f.write(f"# {title}\n\n")
            f.write(f"- source_type: {source_type}\n")
            f.write(f"- source: {source}\n")
            f.write(f"- url: {url}\n")
            f.write(f"- discovered_at: {datetime.now(timezone.utc).isoformat()}\n\n")
            f.write("## Notes\n\n")
            f.write(f"{text.strip()}\n")
        return path

    def _fetch_study_text(self, url: str, timeout: float, max_bytes: int) -> str:
        response = requests.get(
            url,
            timeout=timeout,
            stream=True,
            allow_redirects=True,
        )
        response.raise_for_status()

        content_length = response.headers.get("Content-Length") or response.headers.get("content-length")
        enforce_byte_limit = max_bytes > 0
        if content_length:
            try:
                if enforce_byte_limit and int(content_length) > max_bytes:
                    raise RuntimeError(f"resource exceeds max_resource_bytes: {url}")
            except ValueError:
                pass

        chunks: List[bytes] = []
        downloaded = 0
        for chunk in response.iter_content(chunk_size=16 * 1024):
            if not chunk:
                continue
            downloaded += len(chunk)
            if enforce_byte_limit and downloaded > max_bytes:
                raise RuntimeError(f"resource exceeds max_resource_bytes: {url}")
            chunks.append(chunk)
        raw = b"".join(chunks)
        encoding = response.encoding or "utf-8"
        content_type = response.headers.get("Content-Type", "").lower()
        text = raw.decode(encoding, errors="ignore")
        if "text/html" in content_type or "<html" in text.lower():
            parser = _StudyHTMLTextExtractor()
            parser.feed(text)
            return parser.get_text()
        return text

    def _train_target(
        self,
        target_name: str,
        config: Dict[str, Any],
        dataset_path: Optional[str],
        dataset_id: Optional[str],
    ) -> None:
        output_path = self._abs_path(config["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = Path(f"{output_path}.tmp")

        try:
            if target_name == "channel_classifier":
                train_classifier(
                    output_path=str(tmp_path),
                    n_epochs=int(config.get("n_epochs", 20)),
                    batch_size=int(config.get("batch_size", 32)),
                    lr=float(config.get("lr", 1e-3)),
                    n_samples_per_class=int(config.get("n_samples_per_class", 100)),
                    n_mels=int(config.get("n_mels", 64)),
                    n_frames=int(config.get("n_frames", 64)),
                    dataset_path=dataset_path,
                    device=str(config.get("device", "cpu")),
                )
            elif target_name == "gain_pan_predictor":
                train_gain_predictor(
                    output_path=str(tmp_path),
                    n_epochs=int(config.get("n_epochs", 30)),
                    batch_size=int(config.get("batch_size", 16)),
                    lr=float(config.get("lr", 1e-3)),
                    n_channels=int(config.get("n_channels", 8)),
                    n_samples=int(config.get("n_samples", 500)),
                    dataset_path=dataset_path,
                    device=str(config.get("device", "cpu")),
                )
            elif target_name == "mix_console":
                train_mix_console(
                    output_path=str(tmp_path),
                    n_epochs=int(config.get("n_epochs", 50)),
                    lr=float(config.get("lr", 1e-2)),
                    n_channels=int(config.get("n_channels", 8)),
                    audio_len=int(config.get("audio_len", 16384)),
                    n_samples=int(config.get("n_samples", 200)),
                    sample_rate=int(config.get("sample_rate", 48000)),
                    dataset_path=dataset_path,
                    device=str(config.get("device", "cpu")),
                )
            else:
                raise ValueError(f"Unknown target: {target_name}")

            tmp_path.replace(output_path)
            logger.info("Trained model saved: %s (dataset_id=%s)", output_path, dataset_id)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

    async def _fetch_manifest_with_retries(self, manifest_url: Optional[str] = None) -> Optional[Dict[str, Any]]:
        url = manifest_url or self.config.get("manifest_url")
        if not url:
            return None

        timeout = self.config["request_timeout_sec"]
        retries = max(0, int(self.config.get("max_retries", 0)))
        last_error: Optional[str] = None

        for attempt in range(retries + 1):
            try:
                response = requests.get(url, timeout=timeout)
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                last_error = str(exc)
                logger.warning("Manifest fetch failed (attempt %s/%s): %s", attempt + 1, retries + 1, exc)
                if attempt >= retries:
                    logger.error("Manifest fetch failed permanently: %s", exc)
                    self._last_error = f"manifest_fetch:{exc}"
                    return None
                await asyncio.sleep(min(5, attempt + 1))
        if last_error:
            self._last_error = f"manifest_fetch:{last_error}"
        return None

    @staticmethod
    def _dataset_signature(path: Path) -> str:
        stat = path.stat()
        return f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}"

    def _dataset_cache_path(self, target_name: str, url: str) -> Path:
        target_dir = self.dataset_dir / target_name
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = os.path.basename(url.split("?")[0])
        if not filename:
            filename = f"{target_name}.dataset"
        return target_dir / filename

    def _register_run_download(self, path: Optional[Path]) -> None:
        if not self.config.get("cleanup_downloads"):
            return
        if path and path.exists():
            self._run_download_paths.add(path)

    def _cleanup_downloads(self) -> None:
        if not self.config.get("cleanup_downloads"):
            self._run_download_paths.clear()
            return

        for path in sorted(self._run_download_paths, key=lambda p: len(str(p)), reverse=True):
            if not path.exists():
                continue
            try:
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
            except Exception:
                logger.debug("Failed to remove temp dataset file: %s", path, exc_info=True)
        self._run_download_paths = set()

    def _coerce_to_target_row(self, target: str, row: Any) -> Optional[Dict[str, Any]]:
        patterns = TARGET_FILE_PATTERNS.get(target, ())
        if not row:
            return None
        if isinstance(row, dict):
            if target == "channel_classifier":
                feature = None
                for key in patterns:
                    if key in row:
                        feature = row.get(key)
                        break
                if feature is None:
                    return None
                label = row.get("label")
                if label is None:
                    label = row.get("class")
                if label is None:
                    label = row.get("target")
                if label is None:
                    label = row.get("y", 0)
                return {"spectrogram": feature, "label": label}
            if target == "gain_pan_predictor":
                channels = row.get("channels") or row.get("channel_audios")
                gains = row.get("gains") or row.get("gain")
                pans = row.get("pans") or row.get("pan")
                if channels is None or gains is None or pans is None:
                    return None
                return {"channels": channels, "gains": gains, "pans": pans}
            if target == "mix_console":
                multitracks = row.get("multitracks") or row.get("channels")
                references = row.get("references") or row.get("reference")
                if multitracks is None or references is None:
                    return None
                return {"multitracks": multitracks, "references": references}
            # fallback by filename-like keys in case of compact schemas
            if target == "channel_classifier" and any(k in row for k in patterns):
                for k in patterns:
                    if k in row:
                        label = row.get("label", row.get("class", row.get("target", 0)))
                        return {"spectrogram": row.get(k), "label": label}
            if target == "gain_pan_predictor":
                if any(k in row for k in patterns):
                    return {
                        "channels": row.get("channels") or row.get("channel_audios"),
                        "gains": row.get("gains"),
                        "pans": row.get("pans"),
                    }
            if target == "mix_console":
                if any(k in row for k in patterns):
                    return {
                        "multitracks": row.get("multitracks") or row.get("channels"),
                        "references": row.get("references") or row.get("reference"),
                    }
            return None

        if isinstance(row, (list, tuple)) and target == "channel_classifier":
            if len(row) >= 2:
                return {"spectrogram": row[0], "label": row[1]}
            if len(row) == 1:
                return {"spectrogram": row[0], "label": 0}
        return None

    def _convert_to_jsonl(self, target_name: str, raw_path: Path) -> Path:
        output_path = raw_path.with_name(f"{raw_path.stem}.{target_name}.jsonl")
        converted = 0
        with open(raw_path, "r", encoding="utf-8") as source, open(
            output_path, "w", encoding="utf-8"
        ) as sink:
            raw = json.load(source)
            iterable: Any = raw
            if isinstance(raw, dict):
                for container_key in ("data", "rows", "examples", "dataset"):
                    if container_key in raw and isinstance(raw[container_key], list):
                        iterable = raw[container_key]
                        break
            if not isinstance(iterable, list):
                raise ValueError(f"Unsupported JSON structure: {raw_path}")
            for item in iterable:
                row = self._coerce_to_target_row(target_name, item)
                if row is None:
                    continue
                sink.write(json.dumps(row, ensure_ascii=False))
                sink.write("\n")
                converted += 1
        if converted == 0:
            output_path.unlink(missing_ok=True)
            raise ValueError(f"No usable rows after JSON conversion: {raw_path}")
        self._register_run_download(output_path)
        return output_path

    def _convert_csv_to_jsonl(self, target_name: str, raw_path: Path) -> Path:
        output_path = raw_path.with_name(f"{raw_path.stem}.{target_name}.jsonl")
        converted = 0
        with open(raw_path, "r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            with open(output_path, "w", encoding="utf-8") as sink:
                for row in reader:
                    normalized = self._coerce_to_target_row(target_name, row)
                    if normalized is None:
                        continue
                    sink.write(json.dumps(normalized, ensure_ascii=False))
                    sink.write("\n")
                    converted += 1
        if converted == 0:
            output_path.unlink(missing_ok=True)
            raise ValueError(f"No usable rows after CSV conversion: {raw_path}")
        self._register_run_download(output_path)
        return output_path

    def _extract_first_supported_in_archive(self, target_name: str, archive: Path) -> Optional[Path]:
        target_dir = archive.parent / f".tmp_{archive.stem}"
        target_dir.mkdir(parents=True, exist_ok=True)
        self._register_run_download(target_dir)

        archive_name_lower = archive.name.lower()
        is_zip = archive_name_lower.endswith(".zip")
        is_targz = archive_name_lower.endswith(".tar.gz")
        is_tgz = archive.suffix.lower() == ".tgz"
        is_tar = archive.suffix.lower() == ".tar" or is_targz or is_tgz

        if is_zip:
            with zipfile.ZipFile(archive) as zf:
                members = zf.namelist()
                chosen = None
                for name in members:
                    if any(name.lower().endswith(ext) for ext in TARGET_DATASET_EXTENSIONS[target_name]):
                        chosen = name
                        break
                if chosen is None:
                    raise ValueError(f"No supported files in archive: {archive}")
                if chosen.endswith("/"):
                    raise ValueError(f"Archive entry is a directory: {chosen}")
                raw = target_dir / os.path.basename(chosen)
                with zf.open(chosen) as source, open(raw, "wb") as sink:
                    sink.write(source.read())
        elif is_tar:
            mode = "r:gz" if (is_targz or is_tgz) else "r"
            with tarfile.open(archive, mode=mode) as tf:
                members = tf.getmembers()
                chosen = None
                for member in members:
                    if not member.isfile():
                        continue
                    if any(member.name.lower().endswith(ext) for ext in TARGET_DATASET_EXTENSIONS[target_name]):
                        chosen = member
                        break
                if chosen is None:
                    raise ValueError(f"No supported files in archive: {archive}")
                raw = target_dir / os.path.basename(chosen.name)
                extracted = tf.extractfile(chosen)
                if extracted is None:
                    raise ValueError(f"Unable to extract archive entry: {chosen.name}")
                with extracted, open(raw, "wb") as sink:
                    shutil.copyfileobj(extracted, sink)
        else:
            # Single-file gzip payload (for example .jsonl.gz / .npz.gz).
            raw = target_dir / archive.with_suffix("").name
            with gzip.open(archive, "rb") as source, open(raw, "wb") as sink:
                shutil.copyfileobj(source, sink)

        return self._prepare_dataset_path(target_name, raw)

    def _prepare_dataset_path(self, target_name: str, raw_path: Path) -> Path:
        extension = raw_path.suffix.lower()
        if extension in {".jsonl", ".npz"}:
            self._register_run_download(raw_path)
            return raw_path
        if extension == ".json":
            return self._convert_to_jsonl(target_name, raw_path)
        if extension == ".csv":
            return self._convert_csv_to_jsonl(target_name, raw_path)
        if extension in {".zip", ".tgz", ".tar", ".gz"}:
            normalized = self._extract_first_supported_in_archive(target_name, raw_path)
            self._register_run_download(raw_path)
            return normalized
        raise ValueError(f"Unsupported dataset file format: {raw_path}")

    def _select_candidate_dataset_urls(self, target_name: str, cfg: Dict[str, Any]) -> List[str]:
        urls: List[str] = []
        discovery = self.config.get("discovery", {})
        for item in self._safe_list(cfg.get("dataset_urls"), []):
            if isinstance(item, str) and item.strip():
                urls.append(item.strip())
        url = cfg.get("dataset_url") or cfg.get("url")
        if isinstance(url, str) and url.strip():
            urls.append(url.strip())

        if not urls:
            if self._safe_bool(discovery.get("enabled"), False):
                for query in self._safe_list(
                    discovery.get("queries", {}).get(target_name),
                    DEFAULT_DISCOVERY_QUERIES.get(target_name, []),
                ):
                    if not isinstance(query, str) or not query.strip():
                        continue
                    urls.extend(self._discover_hf_urls(target_name, query.strip()))

        for item in self._safe_list(discovery.get("candidate_urls"), []):
            if not isinstance(item, str):
                continue
            normalized = item.strip()
            if not normalized:
                continue
            if normalized.lower().startswith(("http://", "https://")):
                urls.append(normalized)
                continue
            # Treat non-URL tokens as Hugging Face dataset IDs and discover files there.
            urls.extend(self._discover_hf_dataset_urls(
                target_name=target_name,
                dataset_id=normalized,
            ))

        max_candidates = max(1, self._safe_int(discovery.get("max_candidates_per_target"), 3))
        dedupe = []
        seen = set()
        for item in urls:
            if item in seen:
                continue
            seen.add(item)
            dedupe.append(item)
            if len(dedupe) >= max_candidates:
                break
        urls = dedupe
        return urls

    def _resolve_hf_file_candidates(
        self,
        target_name: str,
        dataset_id: str,
        max_bytes: int,
    ) -> List[str]:
        timeout = self.config["request_timeout_sec"]
        dataset_meta_url = (
            f"https://huggingface.co/api/datasets/{quote(dataset_id, safe='/')}/revision/main"
        )
        candidates: List[Tuple[int, str]] = []
        try:
            dataset_meta = requests.get(dataset_meta_url, timeout=timeout)
            dataset_meta.raise_for_status()
            files = dataset_meta.json().get("siblings", [])
            for file in files:
                filename = file.get("rfilename")
                if not filename:
                    continue
                lower = filename.lower()
                if lower.startswith(".") or filename.endswith("/"):
                    continue
                allowed = TARGET_DATASET_EXTENSIONS.get(target_name, {".jsonl", ".npz"})
                if not any(lower.endswith(ext) for ext in allowed):
                    continue
                if not self._is_within_size_limit(
                    f"https://huggingface.co/{dataset_id}/resolve/main/{quote(filename, safe='/')}?download=true",
                    max_bytes,
                ):
                    continue
                score = self._score_file_name_for_target(target_name, filename)
                candidates.append((score, filename))
            candidates.sort(key=lambda item: item[0], reverse=True)
            urls: List[str] = []
            for _, filename in candidates:
                raw = f"https://huggingface.co/{dataset_id}/resolve/main/{quote(filename, safe='/')}?download=true"
                urls.append(raw)
            return urls
        except Exception as exc:
            logger.debug("HF dataset scan failed (%s): %s", dataset_id, exc, exc_info=True)
            return []

    def _discover_hf_dataset_urls(self, target_name: str, dataset_id: str) -> List[str]:
        if not dataset_id or not isinstance(dataset_id, str):
            return []
        dataset_id = dataset_id.strip()
        if not dataset_id:
            return []
        if "/" not in dataset_id:
            return []
        max_bytes = self._safe_int(
            self.config.get("discovery", {}).get("max_dataset_bytes"),
            self._safe_int(self.config.get("max_dataset_bytes"), 30 * 1024 * 1024),
        )
        return self._resolve_hf_file_candidates(target_name, dataset_id, max_bytes)

    def _score_file_name_for_target(self, target_name: str, filename: str) -> int:
        name = filename.lower()
        score = 0
        for token in TARGET_FILE_HINT_KEYWORDS.get(target_name, ()):
            if token in name:
                score += 1
        if filename.endswith(".jsonl"):
            score += 1
        if filename.endswith(".npz"):
            score += 1
        if filename.endswith((".json", ".csv")):
            score += 1
        return score


    def _discover_hf_urls(self, target_name: str, query: str) -> List[str]:
        discovery = self.config.get("discovery", {})
        limit = self._safe_int(discovery.get("hf_search_limit"), 8)
        timeout = self.config["request_timeout_sec"]
        urls: List[str] = []
        max_bytes = self._safe_int(
            discovery.get("max_dataset_bytes"),
            self._safe_int(self.config.get("max_dataset_bytes"), 30 * 1024 * 1024),
        )
        try:
            search_url = (
                "https://huggingface.co/api/datasets?search="
                + quote_plus(query)
                + "&limit="
                + str(limit)
                + "&full=true"
            )
            response = requests.get(search_url, timeout=timeout)
            response.raise_for_status()
            datasets = response.json()
            if not isinstance(datasets, list):
                return []

            scored: List[Tuple[int, str]] = []
            for item in datasets:
                dataset_id = item.get("id")
                if not dataset_id:
                    continue
                files = self._resolve_hf_file_candidates(target_name, dataset_id, max_bytes)
                if not files:
                    continue
                dataset_score = 0
                dataset_name = str(dataset_id).lower()
                for keyword in TARGET_FILE_HINT_KEYWORDS.get(target_name, ()):
                    if keyword in dataset_name:
                        dataset_score += 1
                for file_url in files:
                    scored.append((dataset_score, file_url))
            if not scored:
                return []
            scored.sort(key=lambda item: item[0], reverse=True)
            urls.extend(item[1] for item in scored)
            return urls
        except Exception as exc:
            logger.debug("HF discovery failed for %s / %s: %s", target_name, query, exc, exc_info=True)
            return []

    def _is_within_size_limit(self, url: str, max_bytes: int) -> bool:
        if not max_bytes:
            return True
        try:
            head = requests.head(url, timeout=self.config["request_timeout_sec"], allow_redirects=True)
            content_length = head.headers.get("Content-Length") or head.headers.get("content-length")
            if content_length:
                return int(content_length) <= max_bytes
            return True
        except Exception:
            return True

    async def _resolve_dataset(
        self,
        target_name: str,
        cfg: Dict[str, Any],
        previous_dataset_id: Optional[str] = None,
        dry_run: bool = False,
    ) -> Tuple[Optional[str], Optional[str], bool, bool, Dict[str, Any]]:
        manifest_id = cfg.get("dataset_id") or cfg.get("version")
        dataset_file = cfg.get("dataset_file")
        local_path = cfg.get("data_path")
        candidate_urls = self._select_candidate_dataset_urls(target_name, cfg)
        resolve_info = {
            "status": "not_started",
            "manifest_id": manifest_id,
            "candidate_urls": candidate_urls,
            "checked": [],
            "selected": None,
            "reason": None,
            "max_dataset_bytes": None,
        }
        max_bytes = self._safe_int(
            cfg.get("max_dataset_bytes"),
            self._safe_int(
                self.config.get("discovery", {}).get("max_dataset_bytes"),
                self._safe_int(self.config.get("max_dataset_bytes"), 30 * 1024 * 1024),
            ),
        )
        resolve_info["max_dataset_bytes"] = max_bytes

        if dataset_file:
            resolved = self._abs_path(str(dataset_file))
            if resolved.exists():
                dataset_id = manifest_id or self._dataset_signature(resolved)
                resolve_info["status"] = "resolved_local_dataset_file"
                resolve_info["selected"] = {
                    "mode": "dataset_file",
                    "path": str(resolved),
                }
                return str(resolved), dataset_id, dataset_id != previous_dataset_id, False, resolve_info
            if candidate_urls:
                logger.warning("dataset_file not found: %s; fallback to download", resolved)
                resolve_info["reason"] = "dataset_file_missing"
            else:
                error = f"dataset_file not found: {resolved}"
                resolve_info["status"] = "error"
                resolve_info["reason"] = error
                raise FileNotFoundError(error)

        if local_path:
            resolved = self._abs_path(str(local_path))
            if not resolved.exists():
                error = f"data_path not found: {resolved}"
                resolve_info["status"] = "error"
                resolve_info["reason"] = error
                raise FileNotFoundError(error)
            dataset_id = manifest_id or self._dataset_signature(resolved)
            resolve_info["status"] = "resolved_local_path"
            resolve_info["selected"] = {
                "mode": "local_path",
                "path": str(resolved),
            }
            return str(resolved), dataset_id, dataset_id != previous_dataset_id, False, resolve_info

        if not candidate_urls:
            resolve_info["status"] = "no_candidates"
            resolve_info["reason"] = "no dataset_file / data_path and candidate_urls are empty"
            return (
                None,
                manifest_id,
                manifest_id is not None and manifest_id != previous_dataset_id,
                False,
                resolve_info,
            )

        last_error = None
        for url in candidate_urls:
            if not isinstance(url, str):
                continue
            url = url.strip()
            if not url:
                continue
            checked = {"url": url}
            checked["within_limit"] = self._is_within_size_limit(url, max_bytes)
            if not checked["within_limit"]:
                logger.warning("Skipping dataset too large: %s", url)
                last_error = f"too_large:{url}"
                checked["reason"] = "size_limit"
                resolve_info["checked"].append(checked)
                continue

            expected_path = self._dataset_cache_path(target_name, url)
            if manifest_id and previous_dataset_id == manifest_id and expected_path.exists():
                checked["cached_hit"] = True
                try:
                    prepared_path = self._prepare_dataset_path(target_name, expected_path)
                    resolve_info["status"] = "resolved_cached"
                    resolve_info["selected"] = {
                        "mode": "cached",
                        "url": url,
                        "path": str(prepared_path),
                    }
                    resolve_info["checked"].append(checked)
                    return str(prepared_path), manifest_id, False, False, resolve_info
                except Exception as exc:
                    logger.warning(
                        "Cached dataset invalid for %s (%s), re-downloading: %s",
                        target_name,
                        expected_path,
                        exc,
                    )
                    checked["cached_hit"] = False
                    checked["cached_invalid_reason"] = str(exc)
                    resolve_info["checked"].append(checked)
                    try:
                        if expected_path.exists():
                            expected_path.unlink()
                    except Exception:
                        logger.debug(
                            "Failed to remove cached dataset before re-download: %s",
                            expected_path,
                            exc_info=True,
                        )

            if dry_run:
                resolve_info["status"] = "dry_run_selected"
                resolve_info["selected"] = {
                    "mode": "candidate_url",
                    "url": url,
                }
                resolve_info["reason"] = "safe_autostart_dry_run"
                resolve_info["checked"].append(checked)
                dataset_id = manifest_id or url
                return None, dataset_id, dataset_id != previous_dataset_id, False, resolve_info

            try:
                downloaded_path, download_id = await self._download_dataset(target_name, url)
                prepared_path = self._prepare_dataset_path(target_name, downloaded_path)
                if (
                    prepared_path.suffix.lower() == ".jsonl"
                    or prepared_path.suffix.lower() == ".npz"
                ):
                    self._register_run_download(prepared_path)
                dataset_id = manifest_id or download_id
                resolve_info["status"] = "downloaded"
                resolve_info["selected"] = {
                    "mode": "downloaded",
                    "url": url,
                    "path": str(prepared_path),
                }
                resolve_info["checked"].append(checked)
                return str(prepared_path), dataset_id, dataset_id != previous_dataset_id, True, resolve_info
            except Exception as exc:
                logger.warning("Dataset candidate failed for %s (%s): %s", target_name, url, exc)
                last_error = str(exc)
                checked["reason"] = str(exc)
                resolve_info["checked"].append(checked)
                continue

        if last_error and "too_large" not in last_error:
            self._last_error = f"dataset_resolve:{last_error}"
            resolve_info["reason"] = last_error
        elif not resolve_info["selected"]:
            resolve_info["reason"] = "no candidates accepted by limits/candidates"
            resolve_info["status"] = "no_accepted_candidates"
        return (
            None,
            manifest_id,
            manifest_id is not None and manifest_id != previous_dataset_id,
            False,
            resolve_info,
        )

    async def _download_dataset(self, target_name: str, url: str) -> Tuple[Path, str]:
        timeout = self.config["request_timeout_sec"]
        attempts = max(0, int(self.config.get("max_retries", 0)))
        final_path = self._dataset_cache_path(target_name, url)
        temp_path = Path(f"{final_path}.tmp")
        max_bytes = self._safe_int(
            self.config.get("max_dataset_bytes"),
            30 * 1024 * 1024,
        )
        if final_path.exists():
            if final_path.stat().st_size <= max_bytes:
                self._register_run_download(final_path)
                h = hashlib.sha256()
                h.update(final_path.read_bytes())
                return final_path, h.hexdigest()
            final_path.unlink(missing_ok=True)

        for attempt in range(attempts + 1):
            try:
                if not self._is_within_size_limit(url, max_bytes):
                    raise RuntimeError(f"dataset too large: {url}")
                resp = requests.get(url, timeout=timeout, stream=True)
                resp.raise_for_status()
                h = hashlib.sha256()
                downloaded = 0
                with open(temp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
                            h.update(chunk)
                            downloaded += len(chunk)
                            if downloaded > max_bytes:
                                raise RuntimeError(f"dataset exceeds max_dataset_bytes: {url}")
                data_id = (
                    resp.headers.get("ETag")
                    or resp.headers.get("Last-Modified")
                    or h.hexdigest()
                )
                temp_path.replace(final_path)
                self._register_run_download(final_path)
                return final_path, str(data_id)
            except Exception as exc:
                logger.warning(
                    "Dataset download failed (%s -> %s, attempt %s/%s): %s",
                    target_name, url, attempt + 1, attempts + 1, exc,
                )
                if attempt >= attempts:
                    if temp_path.exists():
                        try:
                            temp_path.unlink()
                        except Exception:
                            pass
                    raise
                await asyncio.sleep(min(10, attempt + 2))

        raise RuntimeError(f"Failed to download dataset for {target_name}: {url}")
