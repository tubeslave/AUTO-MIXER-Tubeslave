from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


HF_API_BASE = "https://huggingface.co/api/datasets"
HF_WEB_BASE = "https://huggingface.co/datasets"
SUPPORTED_DATA_EXTENSIONS = (".parquet", ".json", ".jsonl", ".csv")
README_FILENAMES = {"README.md", "readme.md"}
DEFAULT_MAX_DATASET_BYTES = 50 * 1024 * 1024
DEFAULT_REPORT_PATH = "/Users/dmitrijvolkov/Downloads/deep-research-report.md"


@dataclass(frozen=True)
class DatasetSeed:
    dataset_id: str
    role: str
    priority: str
    notes: str


DEFAULT_DATASET_SEEDS = (
    DatasetSeed(
        dataset_id="mclemcrew/MixAssist",
        role="dialogue_instruction",
        priority="A",
        notes="Audio-grounded expert/amateur mixing dialogue for assistant behavior.",
    ),
    DatasetSeed(
        dataset_id="mclemcrew/MixParams",
        role="parameter_prediction",
        priority="A",
        notes="Track-level gain/pan/EQ/compression parameter rows.",
    ),
    DatasetSeed(
        dataset_id="mclemcrew/mix-evaluation-dataset",
        role="evaluated_mix_state",
        priority="A-",
        notes="Compact Hugging Face package related to mix-evaluation metadata.",
    ),
    DatasetSeed(
        dataset_id="mclemcrew/MixologyDB",
        role="parameter_seed",
        priority="B",
        notes="Small manually annotated mix-parameter seed dataset.",
    ),
)

DEFAULT_SEARCH_QUERIES = (
    "music mixing parameters",
    "audio language music mixing",
    "multitrack mixing dataset",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_dataset_dir(dataset_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "__", dataset_id.strip())
    return safe.strip("._") or "dataset"


def build_hf_api_url(dataset_id: str) -> str:
    return f"{HF_API_BASE}/{quote(dataset_id, safe='/')}/revision/main"


def build_hf_file_url(dataset_id: str, filename: str) -> str:
    encoded_file = quote(filename, safe="/")
    return f"{HF_WEB_BASE}/{dataset_id}/resolve/main/{encoded_file}?download=true"


def http_json(url: str, timeout: float) -> dict[str, Any] | list[Any]:
    request = Request(url, headers={"User-Agent": "AUTO-MIXER-dataset-bootstrap/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def http_content_length(url: str, timeout: float) -> int | None:
    request = Request(url, method="HEAD", headers={"User-Agent": "AUTO-MIXER-dataset-bootstrap/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            value = response.headers.get("Content-Length")
            return int(value) if value else None
    except (HTTPError, URLError, TimeoutError, ValueError):
        return None


def sha256_file(path: Path) -> str:
    try:
        result = subprocess.run(
            ["shasum", "-a", "256", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.split()[0]
    except Exception:
        # Fallback for non-macOS environments.
        import hashlib

        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()


def is_supported_dataset_file(filename: str) -> bool:
    lower = filename.lower()
    return lower.endswith(SUPPORTED_DATA_EXTENSIONS) or filename in README_FILENAMES


def select_supported_files(siblings: list[dict[str, Any]]) -> list[str]:
    selected: list[str] = []
    for item in siblings:
        filename = item.get("rfilename")
        if not isinstance(filename, str) or not filename.strip():
            continue
        if filename.startswith(".") or filename.endswith("/"):
            continue
        if is_supported_dataset_file(filename):
            selected.append(filename)
    return sorted(selected)


def search_hugging_face(query: str, limit: int, timeout: float) -> list[dict[str, Any]]:
    url = f"{HF_API_BASE}?{urlencode({'search': query, 'limit': max(1, limit), 'full': 'true'})}"
    try:
        payload = http_json(url, timeout=timeout)
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    results: list[dict[str, Any]] = []
    for item in payload[:limit]:
        if not isinstance(item, dict):
            continue
        dataset_id = item.get("id")
        if not isinstance(dataset_id, str):
            continue
        results.append(
            {
                "dataset_id": dataset_id,
                "downloads": item.get("downloads"),
                "likes": item.get("likes"),
                "tags": item.get("tags", []),
                "url": f"{HF_WEB_BASE}/{dataset_id}",
            }
        )
    return results


def download_file(url: str, output_path: Path, max_bytes: int, timeout: float) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    request = Request(url, headers={"User-Agent": "AUTO-MIXER-dataset-bootstrap/1.0"})
    downloaded = 0
    with urlopen(request, timeout=timeout) as response, temp_path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 256)
            if not chunk:
                break
            downloaded += len(chunk)
            if downloaded > max_bytes:
                temp_path.unlink(missing_ok=True)
                raise RuntimeError(f"download exceeds limit {max_bytes}: {url}")
            handle.write(chunk)
    temp_path.replace(output_path)
    return {
        "path": str(output_path),
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
    }


def convert_parquet_to_jsonl(path: Path) -> Path | None:
    if path.suffix.lower() != ".parquet":
        return None
    try:
        import pyarrow.parquet as pq
    except Exception:
        return None

    table = pq.read_table(path)
    output_path = path.with_suffix(".jsonl")
    with output_path.open("w", encoding="utf-8") as handle:
        for row in table.to_pylist():
            handle.write(json.dumps(row, ensure_ascii=False, default=str))
            handle.write("\n")
    return output_path


def discover_dataset(
    seed: DatasetSeed,
    output_dir: Path,
    max_bytes: int,
    timeout: float,
    download: bool,
    convert_jsonl: bool,
) -> dict[str, Any]:
    dataset_id = seed.dataset_id
    api_url = build_hf_api_url(dataset_id)
    record: dict[str, Any] = {
        "dataset_id": dataset_id,
        "role": seed.role,
        "priority": seed.priority,
        "notes": seed.notes,
        "url": f"{HF_WEB_BASE}/{dataset_id}",
        "api_url": api_url,
        "status": "pending",
        "files": [],
    }

    try:
        payload = http_json(api_url, timeout=timeout)
    except Exception as exc:
        record["status"] = "metadata_error"
        record["error"] = str(exc)
        return record

    if not isinstance(payload, dict):
        record["status"] = "metadata_error"
        record["error"] = "unexpected Hugging Face API response"
        return record

    card_data = payload.get("cardData")
    if isinstance(card_data, dict):
        record["license"] = card_data.get("license")
    record["downloads"] = payload.get("downloads")
    record["likes"] = payload.get("likes")

    filenames = select_supported_files(payload.get("siblings", []))
    dataset_dir = output_dir / safe_dataset_dir(dataset_id)
    for filename in filenames:
        file_url = build_hf_file_url(dataset_id, filename)
        size = http_content_length(file_url, timeout=timeout)
        file_record: dict[str, Any] = {
            "filename": filename,
            "url": file_url,
            "content_length": size,
            "status": "selected",
        }
        if size is not None and size > max_bytes:
            file_record["status"] = "skipped_too_large"
            record["files"].append(file_record)
            continue
        if download:
            target_path = dataset_dir / filename
            try:
                downloaded = download_file(
                    file_url,
                    target_path,
                    max_bytes=max_bytes,
                    timeout=timeout,
                )
                file_record.update(downloaded)
                file_record["status"] = "downloaded"
                if convert_jsonl:
                    converted = convert_parquet_to_jsonl(target_path)
                    if converted is not None:
                        file_record["jsonl_path"] = str(converted)
                        file_record["jsonl_bytes"] = converted.stat().st_size
                        file_record["jsonl_sha256"] = sha256_file(converted)
            except Exception as exc:
                file_record["status"] = "download_error"
                file_record["error"] = str(exc)
        record["files"].append(file_record)

    downloaded_count = sum(1 for item in record["files"] if item.get("status") == "downloaded")
    record["status"] = "downloaded" if downloaded_count else "discovered"
    return record


def write_markdown_index(manifest_path: Path, manifest: dict[str, Any]) -> Path:
    index_path = manifest_path.with_suffix(".md")
    with index_path.open("w", encoding="utf-8") as handle:
        handle.write("# Audio Mixing Training Dataset Bootstrap\n\n")
        handle.write(f"- generated_at: {manifest['generated_at']}\n")
        handle.write(f"- report_path: {manifest.get('report_path') or 'n/a'}\n")
        handle.write(f"- max_dataset_bytes: {manifest['max_dataset_bytes']}\n\n")
        handle.write("## Downloaded datasets\n\n")
        for dataset in manifest["datasets"]:
            handle.write(f"### {dataset['dataset_id']}\n\n")
            handle.write(f"- role: {dataset['role']}\n")
            handle.write(f"- priority: {dataset['priority']}\n")
            handle.write(f"- url: {dataset['url']}\n")
            handle.write(f"- status: {dataset['status']}\n")
            if dataset.get("license"):
                handle.write(f"- license: {dataset['license']}\n")
            handle.write("\n")
            for file_record in dataset.get("files", []):
                local = file_record.get("path") or "not downloaded"
                handle.write(
                    f"- `{file_record['filename']}`: {file_record['status']} ({local})\n"
                )
            handle.write("\n")
    return index_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Discover and download compact audio-mixing training datasets.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/training_datasets/audio_mixing",
        help="Directory for downloaded dataset files and manifest.",
    )
    parser.add_argument(
        "--report",
        default=DEFAULT_REPORT_PATH if Path(DEFAULT_REPORT_PATH).exists() else "",
        help="Research report used as context for this bootstrap.",
    )
    parser.add_argument(
        "--dataset-id",
        action="append",
        dest="dataset_ids",
        default=[],
        help="Additional Hugging Face dataset id to include.",
    )
    parser.add_argument(
        "--search-query",
        action="append",
        dest="search_queries",
        default=[],
        help="Additional Hugging Face search query to record in the manifest.",
    )
    parser.add_argument("--search-limit", type=int, default=8)
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    parser.add_argument("--max-dataset-bytes", type=int, default=DEFAULT_MAX_DATASET_BYTES)
    parser.add_argument("--manifest-name", default="dataset_manifest.json")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-convert-jsonl", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(DEFAULT_DATASET_SEEDS)
    known_ids = {seed.dataset_id for seed in seeds}
    for dataset_id in args.dataset_ids:
        if dataset_id not in known_ids:
            seeds.append(
                DatasetSeed(
                    dataset_id=dataset_id,
                    role="user_supplied",
                    priority="manual",
                    notes="User-supplied dataset id.",
                )
            )
            known_ids.add(dataset_id)

    search_queries = list(DEFAULT_SEARCH_QUERIES)
    search_queries.extend(args.search_queries)
    search_results = []
    for query in search_queries:
        search_results.append(
            {
                "query": query,
                "results": search_hugging_face(
                    query=query,
                    limit=args.search_limit,
                    timeout=args.timeout_sec,
                ),
            }
        )

    datasets = [
        discover_dataset(
            seed=seed,
            output_dir=output_dir,
            max_bytes=args.max_dataset_bytes,
            timeout=args.timeout_sec,
            download=not args.no_download,
            convert_jsonl=not args.no_convert_jsonl,
        )
        for seed in seeds
    ]

    report_path = str(Path(args.report).expanduser()) if args.report else ""
    manifest = {
        "generated_at": utc_now(),
        "report_path": report_path,
        "download_enabled": not args.no_download,
        "jsonl_conversion_enabled": not args.no_convert_jsonl,
        "max_dataset_bytes": args.max_dataset_bytes,
        "datasets": datasets,
        "search_results": search_results,
        "next_steps": [
            "Review licenses and terms before bulk audio downloads.",
            "Adapt MixParams/MixologyDB schemas before using them as model targets.",
            "Keep Cambridge/SOS REAPER archives as manual review candidates before download.",
        ],
    }

    manifest_path = output_dir / args.manifest_name
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    index_path = write_markdown_index(manifest_path, manifest)

    print(
        json.dumps(
            {
                "status": "ok",
                "manifest_path": str(manifest_path),
                "index_path": str(index_path),
                "datasets": [
                    {
                        "dataset_id": item["dataset_id"],
                        "status": item["status"],
                        "downloaded_files": sum(
                            1 for file_item in item.get("files", [])
                            if file_item.get("status") == "downloaded"
                        ),
                    }
                    for item in datasets
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
