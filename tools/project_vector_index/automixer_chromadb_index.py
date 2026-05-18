#!/usr/bin/env python3
"""Local ChromaDB indexer for Automixer project knowledge."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_NAME = "AUTO-MIXER-Tubeslave"
DEFAULT_COLLECTION = "automixer_project_knowledge"
DEFAULT_PERSIST_DIR = ".chromadb/automixer_project"
DEFAULT_MANIFEST = ".chromadb/automixer_project_manifest.json"
DEFAULT_EMBEDDING_DIM = 384
DEFAULT_MAX_FILE_BYTES = 768 * 1024
DEFAULT_CHUNK_CHARS = 2200
DEFAULT_CHUNK_OVERLAP = 240
DEFAULT_BATCH_SIZE = 96
DEFAULT_PAPERCLIP_API_URL = "http://127.0.0.1:3100"
DEFAULT_PAPERCLIP_STATUSES = "backlog,todo,in_progress,in_review,blocked"
DEFAULT_PAPERCLIP_LIMIT = 200
DEFAULT_HTTP_TIMEOUT_SECONDS = 5.0

DEFAULT_INCLUDE_GLOBS = (
    "README.md",
    "CLAUDE.md",
    "BUILD.md",
    "Docs/**/*.md",
    "Docs/reports/tub/TUB-*.md",
    "backend/TUB-*.md",
    "backend/ai/knowledge/**/*.md",
    "tools/**/*.md",
    "config/**/*.yaml",
    "config/**/*.yml",
    "config/**/*.json",
    "ai_mixing_pipeline/**/*.py",
    "backend/ai/**/*.py",
    "backend/handlers/**/*.py",
    "backend/wing_client.py",
    "backend/server.py",
    "backend/auto_soundcheck_engine.py",
    "backend/tub345_supervised_write_runner.py",
)
PAPERCLIP_REPORT_GLOBS = (".paperclip/reports/**/*.md",)
EXCLUDED_PARTS = {
    ".git",
    ".chromadb",
    ".mypy_cache",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "artifacts",
    "dist",
    "env",
    "external",
    "frontend/node_modules",
    "logs",
    "node_modules",
    "venv",
}
EXCLUDED_REL_PATHS = {
    "config/user_config.json",
}
SECRET_FILENAMES = {
    ".env",
    ".secret",
    "id_rsa",
    "id_ed25519",
}
SECRET_SUFFIXES = (".key", ".pem", ".p12", ".pfx")
TOKEN_RE = re.compile(r"[\w./:-]+", re.UNICODE)
TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")
SENSITIVE_JSON_KEY_RE = re.compile(
    r"(authorization|token|secret|api[_-]?key|private.*key|privatekeypem|keypem|password)",
    re.IGNORECASE,
)
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL),
    re.compile(r"(OPENAI_API_KEY\s*=\s*)[^\s]+", re.IGNORECASE),
    re.compile(r"(PAPERCLIP_TOKEN\s*=\s*)[^\s]+", re.IGNORECASE),
    re.compile(r"(TELEGRAM_BOT_TOKEN\s*=\s*)[^\s]+", re.IGNORECASE),
    re.compile(r"(Authorization:\s*Bearer\s+)[A-Za-z0-9._\-]+", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"\bsk-proj-[A-Za-z0-9_\-]{16,}\b"),
)
WING_QUERY_TERMS = {"wing", "xremote", "osc", "readback", "rollback", "supervised", "console"}
SAFETY_QUERY_TERMS = {
    "approval",
    "armed",
    "cooldown",
    "dry",
    "dryrun",
    "emergency",
    "gate",
    "panic",
    "rate",
    "readback",
    "rollback",
    "safety",
    "stop",
    "supervised",
    "write",
}
PAPERCLIP_QUERY_TERMS = {"paperclip", "api", "companies", "agents", "issues", "dashboard"}
LIVE_CONSOLE_TERMS = {"wing", "wing rack", "osc", "xremote", "/ch/", "/main/", "/bus/"}
SAFETY_HIT_TERMS = {
    "approval",
    "armed",
    "cooldown",
    "disarmed",
    "dry-run",
    "dry run",
    "emergency",
    "panic",
    "rate limit",
    "readback",
    "rollback",
    "safety",
    "supervised",
    "throttle",
    "write gate",
}
KEYWORD_EXACT_PHRASE_BONUS = 24.0
KEYWORD_TOKEN_COVERAGE_BONUS = 30.0
KEYWORD_ALL_TOKENS_BONUS = 8.0
VECTOR_SCORE_SCALE = 12.0
MIN_SEARCH_CANDIDATES = 12
SEARCH_CANDIDATE_MULTIPLIER = 6


@dataclass(frozen=True)
class SourceChunk:
    record_id: str
    document: str
    embedding: list[float]
    metadata: dict[str, str | int | float | bool]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def redact_secrets(text: str) -> str:
    redacted = text
    for pattern in SECRET_PATTERNS:
        if pattern.groups:
            redacted = pattern.sub(r"\1<redacted>", redacted)
        else:
            redacted = pattern.sub("<redacted>", redacted)
    return redacted


def sanitize_payload(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, nested in value.items():
            key_text = str(key)
            if SENSITIVE_JSON_KEY_RE.search(key_text):
                sanitized[key_text] = "<redacted>"
            else:
                sanitized[key_text] = sanitize_payload(nested)
        return sanitized
    if isinstance(value, list):
        return [sanitize_payload(item) for item in value]
    if isinstance(value, str):
        return redact_secrets(value)
    return value


def sanitize_secret_text(text: str, secrets: Iterable[str | None] = ()) -> str:
    sanitized = redact_secrets(text)
    for secret in secrets:
        if secret:
            sanitized = sanitized.replace(secret, "<redacted>")
    return sanitized


def path_is_secret(path: Path) -> bool:
    name = path.name
    lower_name = name.lower()
    return name in SECRET_FILENAMES or lower_name.endswith(SECRET_SUFFIXES)


def path_has_excluded_part(path: Path) -> bool:
    normalized_parts = {part for part in path.parts}
    joined_parts = "/".join(path.parts)
    return any(part in normalized_parts or part in joined_parts for part in EXCLUDED_PARTS)


def iter_source_paths(
    project_root: Path,
    *,
    include_paperclip_reports: bool = False,
    extra_globs: Iterable[str] = (),
) -> list[Path]:
    patterns = list(DEFAULT_INCLUDE_GLOBS)
    if include_paperclip_reports:
        patterns.extend(PAPERCLIP_REPORT_GLOBS)
    patterns.extend(extra_globs)

    seen: set[Path] = set()
    result: list[Path] = []
    for pattern in patterns:
        for path in project_root.glob(pattern):
            if not path.is_file():
                continue
            try:
                rel_path = path.relative_to(project_root)
            except ValueError:
                continue
            if rel_path.as_posix() in EXCLUDED_REL_PATHS:
                continue
            if path_has_excluded_part(rel_path) or path_is_secret(rel_path):
                continue
            if path not in seen:
                seen.add(path)
                result.append(path)
    return sorted(result)


def read_text(path: Path, max_bytes: int = DEFAULT_MAX_FILE_BYTES) -> str | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    if stat.st_size <= 0 or stat.st_size > max_bytes:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
    except OSError:
        return None


def chunk_text(
    text: str,
    *,
    max_chars: int = DEFAULT_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        hard_end = min(text_len, start + max_chars)
        end = hard_end
        if hard_end < text_len:
            newline = text.rfind("\n", start + max_chars // 2, hard_end)
            if newline > start:
                end = newline
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start = max(end - overlap_chars, start + 1)
    return chunks


def extract_search_tokens(text: str) -> list[str]:
    tokens: set[str] = set()
    for token in TOKEN_RE.findall(text.lower()):
        if len(token) > 1:
            tokens.add(token)
        for part in TOKEN_SPLIT_RE.split(token):
            if len(part) > 1:
                tokens.add(part)
    return sorted(tokens)


def classify_query_intent(query: str) -> dict[str, bool]:
    tokens = set(extract_search_tokens(query))
    return {
        "wing": bool(tokens & WING_QUERY_TERMS),
        "safety": bool(tokens & SAFETY_QUERY_TERMS),
        "paperclip": bool(tokens & PAPERCLIP_QUERY_TERMS),
        "live": "live" in tokens or "runtime" in tokens or "console" in tokens,
        "supervised": "supervised" in tokens,
    }


def hit_priority_flags(metadata: dict[str, Any], document: str) -> dict[str, bool]:
    source_path = str(metadata.get("source_path") or "").lower()
    source_name = str(metadata.get("source_name") or "").lower()
    category = str(metadata.get("category") or "").lower()
    source_type = str(metadata.get("source_type") or "").lower()
    haystack = f"{document}\n{source_path}\n{source_name}\n{category}".lower()
    live_console = any(term in haystack for term in LIVE_CONSOLE_TERMS)
    safety = any(term in haystack for term in SAFETY_HIT_TERMS)
    backend_runtime = category in {"backend_code", "runtime_ai_knowledge", "replay_pipeline_code"}
    report = category in {"tub_report", "paperclip_report"}
    paperclip = source_type == "paperclip_api" or source_path.startswith("paperclip:")
    return {
        "backend_runtime": backend_runtime,
        "live_console": live_console,
        "paperclip": paperclip,
        "report": report,
        "safety": safety,
    }


def intent_boost(query_intent: dict[str, bool], flags: dict[str, bool]) -> float:
    boost = 0.0
    if query_intent["wing"]:
        if flags["live_console"]:
            boost += 24.0
        if flags["backend_runtime"]:
            boost += 10.0
        if flags["report"]:
            boost += 8.0
    if query_intent["safety"] or query_intent["supervised"]:
        if flags["safety"]:
            boost += 22.0
        if flags["report"]:
            boost += 10.0
        if flags["backend_runtime"]:
            boost += 8.0
    elif query_intent["live"] and flags["backend_runtime"]:
        boost += 6.0
    if query_intent["paperclip"] and flags["paperclip"]:
        boost += 26.0
    return boost


def keyword_score_hit(
    query: str,
    metadata: dict[str, Any],
    document: str,
    *,
    query_tokens: list[str] | None = None,
    query_intent: dict[str, bool] | None = None,
) -> float:
    query_lower = query.lower().strip()
    query_tokens = query_tokens or extract_search_tokens(query_lower)
    if not query_tokens:
        return 0.0
    query_intent = query_intent or classify_query_intent(query_lower)
    source_path = str(metadata.get("source_path") or "").lower()
    source_name = str(metadata.get("source_name") or "").lower()
    category = str(metadata.get("category") or "").lower()
    haystack = f"{document}\n{source_path}\n{source_name}\n{category}".lower()
    score = 0.0
    if query_lower and query_lower in haystack:
        score += KEYWORD_EXACT_PHRASE_BONUS

    matched_tokens = 0
    for token in query_tokens:
        count = haystack.count(token)
        if count:
            matched_tokens += 1
            token_weight = min(4.0, max(1.0, len(token) / 3.5))
            score += min(2, count) * token_weight
            if token in source_path:
                score += 8.0
            if token in source_name:
                score += 6.0
            if token in category:
                score += 5.0
    if matched_tokens:
        coverage = matched_tokens / len(query_tokens)
        score += KEYWORD_TOKEN_COVERAGE_BONUS * coverage
        if matched_tokens == len(query_tokens):
            score += KEYWORD_ALL_TOKENS_BONUS

    score += intent_boost(query_intent, hit_priority_flags(metadata, document))
    return score


def dedupe_key(hit: dict[str, Any]) -> str:
    metadata = hit.get("metadata") or {}
    source_path = str(metadata.get("source_path") or "").strip()
    if source_path:
        return source_path
    return str(hit.get("id") or "")


def merged_rank_score(hit: dict[str, Any]) -> float:
    keyword_score = float(hit.get("keyword_score") or 0.0)
    distance = hit.get("distance")
    vector_score = 0.0
    if isinstance(distance, (int, float)):
        vector_score = max(0.0, 1.0 - float(distance))
    return keyword_score + vector_score * VECTOR_SCORE_SCALE


def hash_embedding(text: str, *, dim: int = DEFAULT_EMBEDDING_DIM) -> list[float]:
    vector = [0.0] * dim
    tokens = TOKEN_RE.findall(text.lower())
    if not tokens:
        return vector
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def source_category(rel_path: Path) -> str:
    parts = rel_path.parts
    if parts[:2] == (".paperclip", "reports"):
        return "paperclip_report"
    if parts[:3] == ("Docs", "reports", "tub") and rel_path.name.startswith("TUB-"):
        return "tub_report"
    if parts and parts[0] == "Docs":
        return "docs"
    if parts[:3] == ("backend", "ai", "knowledge"):
        return "runtime_ai_knowledge"
    if parts and parts[0] == "backend" and rel_path.name.startswith("TUB-"):
        return "tub_report"
    if parts and parts[0] == "tools":
        return "automation_tooling"
    if parts and parts[0] == "config":
        return "config"
    if parts and parts[0] == "ai_mixing_pipeline":
        return "replay_pipeline_code"
    if parts and parts[0] == "backend":
        return "backend_code"
    return "project_doc"


def build_source_chunks(
    project_root: Path,
    paths: Iterable[Path],
    *,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    indexed_at: str | None = None,
) -> list[SourceChunk]:
    timestamp = indexed_at or utc_now()
    records: list[SourceChunk] = []
    for path in paths:
        raw = read_text(path)
        if raw is None:
            continue
        safe_text = redact_secrets(raw)
        try:
            rel_path = path.relative_to(project_root)
        except ValueError:
            rel_path = path
        rel_text = rel_path.as_posix()
        file_hash = hashlib.sha256(safe_text.encode("utf-8")).hexdigest()
        chunks = chunk_text(safe_text)
        for index, chunk in enumerate(chunks):
            chunk_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
            record_id = "automixer-" + hashlib.sha256(
                f"{rel_text}:{index}:{chunk_hash}".encode("utf-8")
            ).hexdigest()[:32]
            records.append(
                SourceChunk(
                    record_id=record_id,
                    document=chunk,
                    embedding=hash_embedding(chunk, dim=embedding_dim),
                    metadata={
                        "project": PROJECT_NAME,
                        "source_path": rel_text,
                        "source_name": rel_path.name,
                        "source_type": "file",
                        "category": source_category(rel_path),
                        "chunk_index": index,
                        "chunk_count": len(chunks),
                        "content_sha256": file_hash,
                        "chunk_sha256": chunk_hash,
                        "indexed_at": timestamp,
                        "embedding": f"hash_bow_v1:{embedding_dim}",
                    },
                )
            )
    return records


def build_manual_chunks(
    text: str,
    *,
    title: str,
    category: str,
    source: str,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    indexed_at: str | None = None,
) -> list[SourceChunk]:
    timestamp = indexed_at or utc_now()
    safe_text = redact_secrets(text)
    chunks = chunk_text(safe_text)
    records: list[SourceChunk] = []
    for index, chunk in enumerate(chunks):
        chunk_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
        record_id = "automixer-note-" + hashlib.sha256(
            f"{source}:{title}:{timestamp}:{index}:{chunk_hash}".encode("utf-8")
        ).hexdigest()[:32]
        records.append(
            SourceChunk(
                record_id=record_id,
                document=chunk,
                embedding=hash_embedding(chunk, dim=embedding_dim),
                metadata={
                    "project": PROJECT_NAME,
                    "source_path": source,
                    "source_name": title,
                    "source_type": "manual_note",
                    "category": category,
                    "chunk_index": index,
                    "chunk_count": len(chunks),
                    "content_sha256": hashlib.sha256(safe_text.encode("utf-8")).hexdigest(),
                    "chunk_sha256": chunk_hash,
                    "indexed_at": timestamp,
                    "embedding": f"hash_bow_v1:{embedding_dim}",
                },
            )
        )
    return records


def build_external_chunks(
    text: str,
    *,
    title: str,
    category: str,
    source: str,
    source_type: str,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    indexed_at: str | None = None,
) -> list[SourceChunk]:
    timestamp = indexed_at or utc_now()
    safe_text = redact_secrets(text)
    chunks = chunk_text(safe_text)
    records: list[SourceChunk] = []
    for index, chunk in enumerate(chunks):
        chunk_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
        record_id = "automixer-ext-" + hashlib.sha256(
            f"{source}:{title}:{source_type}:{index}:{chunk_hash}".encode("utf-8")
        ).hexdigest()[:32]
        records.append(
            SourceChunk(
                record_id=record_id,
                document=chunk,
                embedding=hash_embedding(chunk, dim=embedding_dim),
                metadata={
                    "project": PROJECT_NAME,
                    "source_path": source,
                    "source_name": title,
                    "source_type": source_type,
                    "category": category,
                    "chunk_index": index,
                    "chunk_count": len(chunks),
                    "content_sha256": hashlib.sha256(safe_text.encode("utf-8")).hexdigest(),
                    "chunk_sha256": chunk_hash,
                    "indexed_at": timestamp,
                    "embedding": f"hash_bow_v1:{embedding_dim}",
                },
            )
        )
    return records


def paperclip_get_json(
    base_url: str,
    path: str,
    *,
    token: str | None = None,
    params: dict[str, Any] | None = None,
    timeout_seconds: float = DEFAULT_HTTP_TIMEOUT_SECONDS,
) -> Any:
    query = ""
    if params:
        query = "?" + urllib.parse.urlencode(
            {key: value for key, value in params.items() if value is not None}
        )
    request = urllib.request.Request(f"{base_url.rstrip('/')}{path}{query}", method="GET")
    request.add_header("Accept", "application/json")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        raw = response.read().decode("utf-8", errors="replace")
    if not raw:
        return None
    return json.loads(raw)


def extract_items(payload: Any, candidate_keys: list[str]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in candidate_keys:
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            nested = extract_items(value, candidate_keys)
            if nested:
                return nested
    for value in payload.values():
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            return value
    return []


def select_paperclip_company(companies_payload: Any, preferred_company_id: str | None = None) -> str | None:
    companies = extract_items(companies_payload, ["companies", "data", "items", "results"])
    if preferred_company_id:
        for company in companies:
            if str(company.get("id") or company.get("companyId") or "") == preferred_company_id:
                return preferred_company_id
        return preferred_company_id
    if not companies:
        return None

    def preference(company: dict[str, Any]) -> tuple[int, int, str]:
        name = str(company.get("name") or "").strip().lower()
        prefix = str(company.get("issuePrefix") or company.get("issue_prefix") or "").strip().lower()
        try:
            counter = int(company.get("issueCounter") or company.get("issue_counter") or 0)
        except (TypeError, ValueError):
            counter = 0
        priority = 0 if prefix == "tub" or "tubeslave automixer" in name else 1
        return (priority, -counter, name)

    selected = sorted(companies, key=preference)[0]
    company_id = selected.get("id") or selected.get("companyId")
    return str(company_id) if company_id else None


def compact_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str)


def build_paperclip_chunks(
    *,
    base_url: str,
    company_id: str | None,
    token: str | None,
    statuses: str,
    limit: int,
    timeout_seconds: float,
    indexed_at: str | None = None,
) -> tuple[list[SourceChunk], dict[str, Any]]:
    timestamp = indexed_at or utc_now()
    records: list[SourceChunk] = []
    errors: list[str] = []
    fetched: list[str] = []
    secrets = [token]

    def add_payload(name: str, path: str, payload: Any, params: dict[str, Any] | None = None) -> None:
        source = f"paperclip:{path}"
        if params:
            source += "?" + urllib.parse.urlencode(params)
        title = f"Paperclip {name}"
        text = compact_json({"endpoint": path, "params": params or {}, "payload": sanitize_payload(payload)})
        records.extend(
            build_external_chunks(
                text,
                title=title,
                category="paperclip_api",
                source=source,
                source_type="paperclip_api",
                indexed_at=timestamp,
            )
        )
        fetched.append(source)

    def fetch(name: str, path: str, params: dict[str, Any] | None = None) -> Any:
        try:
            payload = paperclip_get_json(
                base_url,
                path,
                token=token,
                params=params,
                timeout_seconds=timeout_seconds,
            )
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError) as exc:
            error_text = sanitize_secret_text(str(exc), secrets)
            errors.append(f"{path}: {type(exc).__name__}: {error_text}")
            return None
        add_payload(name, path, payload, params=params)
        return payload

    health = fetch("health", "/api/health")
    companies_payload = fetch("companies", "/api/companies")
    resolved_company_id = select_paperclip_company(companies_payload, company_id)
    if resolved_company_id:
        prefix = f"/api/companies/{urllib.parse.quote(resolved_company_id, safe='')}"
        fetch("dashboard", f"{prefix}/dashboard")
        fetch("agents", f"{prefix}/agents")
        fetch("issues", f"{prefix}/issues", {"status": statuses, "limit": limit})
        fetch("live-runs", f"{prefix}/live-runs", {"limit": min(limit, 100), "minCount": 0})
    else:
        errors.append("company_id_unavailable")

    summary = {
        "paperclip_api_ok": bool(health is not None),
        "base_url": base_url.rstrip("/"),
        "company_id": resolved_company_id,
        "fetched": fetched,
        "errors": errors,
        "record_count": len(records),
    }
    return records, summary


def batched(records: list[SourceChunk], batch_size: int) -> Iterable[list[SourceChunk]]:
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def chroma_collection(
    persist_dir: Path,
    collection_name: str,
    *,
    reset: bool = False,
):
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("chromadb is not installed; install requirements.txt first") from exc

    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=collection_name,
        metadata={
            "project": PROJECT_NAME,
            "embedding": f"hash_bow_v1:{DEFAULT_EMBEDDING_DIM}",
            "hnsw:space": "cosine",
        },
    )


def upsert_records(collection: Any, records: list[SourceChunk], *, batch_size: int) -> None:
    for batch in batched(records, batch_size):
        collection.upsert(
            ids=[record.record_id for record in batch],
            documents=[record.document for record in batch],
            embeddings=[record.embedding for record in batch],
            metadatas=[record.metadata for record in batch],
        )


def write_manifest(
    manifest_path: Path,
    *,
    project_root: Path,
    persist_dir: Path,
    collection_name: str,
    records: list[SourceChunk],
    sources: list[Path],
    include_paperclip_reports: bool,
) -> None:
    source_paths = [path.relative_to(project_root).as_posix() for path in sources]
    payload = {
        "project": PROJECT_NAME,
        "indexed_at": utc_now(),
        "project_root": str(project_root),
        "persist_dir": str(persist_dir),
        "collection": collection_name,
        "embedding": f"hash_bow_v1:{DEFAULT_EMBEDDING_DIM}",
        "source_count": len(source_paths),
        "record_count": len(records),
        "include_paperclip_reports": include_paperclip_reports,
        "sources": source_paths,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_path(value: str, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def command_index(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).expanduser().resolve()
    persist_dir = parse_path(args.persist_dir, project_root)
    manifest_path = parse_path(args.manifest, project_root)
    sources = iter_source_paths(
        project_root,
        include_paperclip_reports=args.include_paperclip_reports,
        extra_globs=args.include,
    )
    records = build_source_chunks(project_root, sources)
    summary = {
        "project_root": str(project_root),
        "persist_dir": str(persist_dir),
        "collection": args.collection,
        "source_count": len(sources),
        "record_count": len(records),
        "include_paperclip_reports": args.include_paperclip_reports,
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    collection = chroma_collection(persist_dir, args.collection, reset=not args.append)
    upsert_records(collection, records, batch_size=args.batch_size)
    write_manifest(
        manifest_path,
        project_root=project_root,
        persist_dir=persist_dir,
        collection_name=args.collection,
        records=records,
        sources=sources,
        include_paperclip_reports=args.include_paperclip_reports,
    )
    print(json.dumps({**summary, "manifest": str(manifest_path)}, indent=2, ensure_ascii=False))
    return 0


def read_add_text(args: argparse.Namespace) -> str:
    sources = [bool(args.text), bool(args.file), bool(args.stdin)]
    if sum(sources) != 1:
        raise ValueError("provide exactly one of --text, --file, or --stdin")
    if args.text:
        return args.text
    if args.file:
        path = Path(args.file).expanduser()
        return path.read_text(encoding="utf-8")
    return sys.stdin.read()


def command_add(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).expanduser().resolve()
    persist_dir = parse_path(args.persist_dir, project_root)
    text = read_add_text(args)
    records = build_manual_chunks(
        text,
        title=args.title,
        category=args.category,
        source=args.source,
    )
    collection = chroma_collection(persist_dir, args.collection, reset=False)
    upsert_records(collection, records, batch_size=args.batch_size)
    print(json.dumps({"added_records": len(records), "collection": args.collection}, indent=2))
    return 0


def command_search(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).expanduser().resolve()
    persist_dir = parse_path(args.persist_dir, project_root)
    query = " ".join(args.query).strip()
    if not query:
        raise ValueError("search query is required")
    collection = chroma_collection(persist_dir, args.collection, reset=False)
    hits = search_collection(collection, query, limit=args.limit)
    if args.json:
        print(json.dumps(hits, indent=2, ensure_ascii=False))
        return 0
    for idx, hit in enumerate(hits, start=1):
        metadata = hit["metadata"]
        distance = hit.get("distance")
        score_text = f"distance={distance:.4f}" if isinstance(distance, (int, float)) else f"keyword_score={hit.get('keyword_score')}"
        print(f"{idx}. {metadata.get('source_path')} [{metadata.get('category')}] {score_text}")
        excerpt = str(hit.get("document") or "").replace("\n", " ")
        print(f"   {excerpt[:360]}")
    return 0


def search_collection(collection: Any, query: str, *, limit: int) -> list[dict[str, Any]]:
    candidate_limit = max(MIN_SEARCH_CANDIDATES, limit * SEARCH_CANDIDATE_MULTIPLIER)
    keyword_hits = keyword_search_collection(collection, query, limit=candidate_limit)
    try:
        result = collection.query(
            query_embeddings=[hash_embedding(query)],
            n_results=candidate_limit,
            include=["documents", "metadatas", "distances"],
        )
        vector_hits = normalize_search_results(result)
    except Exception:
        vector_hits = []

    if not keyword_hits and not vector_hits:
        return []

    merged: dict[str, dict[str, Any]] = {}
    for hit in vector_hits:
        merged[str(hit.get("id") or len(merged))] = dict(hit)
    for hit in keyword_hits:
        hit_id = str(hit.get("id") or len(merged))
        existing = merged.get(hit_id)
        if existing is None:
            merged[hit_id] = dict(hit)
            continue
        existing["keyword_score"] = max(
            float(existing.get("keyword_score") or 0.0),
            float(hit.get("keyword_score") or 0.0),
        )
        if existing.get("document") in (None, "") and hit.get("document") not in (None, ""):
            existing["document"] = hit.get("document")
        if not existing.get("metadata") and hit.get("metadata"):
            existing["metadata"] = hit.get("metadata")

    ranked = sorted(
        merged.values(),
        key=lambda hit: (
            -merged_rank_score(hit),
            str((hit.get("metadata") or {}).get("source_path") or ""),
            int((hit.get("metadata") or {}).get("chunk_index") or 0),
            str(hit.get("id") or ""),
        ),
    )

    unique_hits: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for hit in ranked:
        key = dedupe_key(hit)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        enriched = dict(hit)
        enriched["rank_score"] = round(merged_rank_score(hit), 4)
        unique_hits.append(enriched)
        if len(unique_hits) >= limit:
            break
    return unique_hits


def keyword_search_collection(collection: Any, query: str, *, limit: int) -> list[dict[str, Any]]:
    query_lower = query.lower().strip()
    tokens = extract_search_tokens(query_lower)
    if not tokens:
        return []
    query_intent = classify_query_intent(query_lower)
    try:
        payload = collection.get(include=["documents", "metadatas"])
    except Exception:
        return []
    ids = payload.get("ids") or []
    documents = payload.get("documents") or []
    metadatas = payload.get("metadatas") or []
    scored: list[tuple[float, dict[str, Any]]] = []
    for idx, document in enumerate(documents):
        metadata = metadatas[idx] if idx < len(metadatas) and isinstance(metadatas[idx], dict) else {}
        score = keyword_score_hit(
            query_lower,
            metadata,
            str(document or ""),
            query_tokens=tokens,
            query_intent=query_intent,
        )
        if score > 0:
            scored.append(
                (
                    score,
                    {
                        "id": ids[idx] if idx < len(ids) else None,
                        "document": document,
                        "metadata": metadata,
                        "distance": None,
                        "keyword_score": round(score, 4),
                    },
                )
            )
    scored.sort(
        key=lambda item: (
            -item[0],
            str((item[1].get("metadata") or {}).get("source_path") or ""),
            int((item[1].get("metadata") or {}).get("chunk_index") or 0),
            str(item[1].get("id") or ""),
        )
    )
    return [hit for _score, hit in scored[:limit]]


def normalize_search_results(result: dict[str, Any]) -> list[dict[str, Any]]:
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    ids = (result.get("ids") or [[]])[0]
    hits = []
    for idx, document in enumerate(documents):
        hits.append(
            {
                "id": ids[idx] if idx < len(ids) else None,
                "document": document,
                "metadata": metadatas[idx] if idx < len(metadatas) else {},
                "distance": distances[idx] if idx < len(distances) else None,
            }
        )
    return hits


def command_stats(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).expanduser().resolve()
    persist_dir = parse_path(args.persist_dir, project_root)
    manifest_path = parse_path(args.manifest, project_root)
    collection = chroma_collection(persist_dir, args.collection, reset=False)
    payload: dict[str, Any] = {
        "collection": args.collection,
        "persist_dir": str(persist_dir),
        "record_count": collection.count(),
        "manifest": str(manifest_path),
        "manifest_exists": manifest_path.exists(),
    }
    if manifest_path.exists():
        try:
            payload["manifest_data"] = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload["manifest_error"] = "invalid json"
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def command_paperclip_sync(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).expanduser().resolve()
    persist_dir = parse_path(args.persist_dir, project_root)
    records, summary = build_paperclip_chunks(
        base_url=args.paperclip_url,
        company_id=args.company_id,
        token=args.token,
        statuses=args.statuses,
        limit=args.limit,
        timeout_seconds=args.timeout_seconds,
    )
    if args.dry_run:
        print(json.dumps({**summary, "dry_run": True}, indent=2, ensure_ascii=False))
        return 0 if records else 1
    if records:
        collection = chroma_collection(persist_dir, args.collection, reset=False)
        upsert_records(collection, records, batch_size=args.batch_size)
    print(json.dumps({**summary, "collection": args.collection, "dry_run": False}, indent=2, ensure_ascii=False))
    return 0 if records else 1


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index Automixer project knowledge into ChromaDB")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="scan project files and rebuild the ChromaDB index")
    add_common_args(index_parser)
    index_parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    index_parser.add_argument("--include", action="append", default=[], help="extra project-root relative glob")
    index_parser.add_argument("--include-paperclip-reports", action="store_true")
    index_parser.add_argument("--append", action="store_true", help="upsert without deleting the existing collection")
    index_parser.add_argument("--dry-run", action="store_true")
    index_parser.set_defaults(func=command_index)

    add_parser = subparsers.add_parser("add", help="add a manual note or external text into ChromaDB")
    add_common_args(add_parser)
    add_parser.add_argument("--title", required=True)
    add_parser.add_argument("--category", default="operator_note")
    add_parser.add_argument("--source", default="manual:operator")
    add_parser.add_argument("--text")
    add_parser.add_argument("--file")
    add_parser.add_argument("--stdin", action="store_true")
    add_parser.set_defaults(func=command_add)

    search_parser = subparsers.add_parser("search", help="search the local ChromaDB index")
    add_common_args(search_parser)
    search_parser.add_argument("--limit", type=int, default=5)
    search_parser.add_argument("--json", action="store_true")
    search_parser.add_argument("query", nargs=argparse.REMAINDER)
    search_parser.set_defaults(func=command_search)

    stats_parser = subparsers.add_parser("stats", help="show ChromaDB index stats")
    add_common_args(stats_parser)
    stats_parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    stats_parser.set_defaults(func=command_stats)

    paperclip_parser = subparsers.add_parser(
        "paperclip-sync",
        help="read Paperclip GET endpoints and upsert summaries into ChromaDB",
    )
    add_common_args(paperclip_parser)
    paperclip_parser.add_argument(
        "--paperclip-url",
        default=os.environ.get("PAPERCLIP_API_URL", DEFAULT_PAPERCLIP_API_URL),
    )
    paperclip_parser.add_argument("--company-id", default=os.environ.get("PAPERCLIP_COMPANY_ID") or None)
    paperclip_parser.add_argument("--token", default=os.environ.get("PAPERCLIP_TOKEN") or None)
    paperclip_parser.add_argument("--statuses", default=DEFAULT_PAPERCLIP_STATUSES)
    paperclip_parser.add_argument("--limit", type=int, default=DEFAULT_PAPERCLIP_LIMIT)
    paperclip_parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_HTTP_TIMEOUT_SECONDS)
    paperclip_parser.add_argument("--dry-run", action="store_true")
    paperclip_parser.set_defaults(func=command_paperclip_sync)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
