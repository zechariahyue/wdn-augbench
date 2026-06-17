"""Helpers for benchmark and split manifest resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_file(path: Path | str) -> dict[str, Any]:
    """Load a YAML file and return a top-level mapping."""
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at top level: {file_path}")
    return data


def resolve_path(value: str | Path, *, root: Path) -> Path:
    """Resolve a path relative to ``root`` unless it is already absolute."""
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (root / candidate).resolve()


def collect_benchmark_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten staged benchmark entries into a single list."""
    staged = manifest.get("staged_benchmarks", {})
    if not isinstance(staged, dict):
        return []

    entries: list[dict[str, Any]] = []
    for stage_name, stage_items in staged.items():
        if not isinstance(stage_items, list):
            continue
        for item in stage_items:
            if not isinstance(item, dict):
                continue
            entry = dict(item)
            entry["stage"] = str(stage_name)
            entries.append(entry)
    return entries


def index_benchmark_entries(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index benchmark entries by benchmark id."""
    indexed: dict[str, dict[str, Any]] = {}
    for entry in collect_benchmark_entries(manifest):
        benchmark_id = str(entry.get("id", "")).strip()
        if benchmark_id:
            indexed[benchmark_id] = entry
    return indexed


def resolve_split_manifest(split_manifest_path: Path | str, *, root: Path) -> dict[str, Any]:
    """Resolve benchmark ids inside a split manifest into concrete benchmark records."""
    split_manifest = load_yaml_file(split_manifest_path)
    benchmark_manifest_path = resolve_path(split_manifest["source_manifest"], root=root)
    benchmark_manifest = load_yaml_file(benchmark_manifest_path)
    benchmark_index = index_benchmark_entries(benchmark_manifest)

    partitions = split_manifest.get("partitions", {})
    if not isinstance(partitions, dict):
        raise ValueError("Split manifest partitions must be a mapping")

    resolved_partitions: dict[str, list[dict[str, Any]]] = {}
    for split_name, split_info in partitions.items():
        if not isinstance(split_info, dict):
            continue
        benchmark_ids = split_info.get("benchmark_ids", [])
        if not isinstance(benchmark_ids, list):
            raise ValueError(f"benchmark_ids for split '{split_name}' must be a list")

        resolved_entries: list[dict[str, Any]] = []
        for benchmark_id in benchmark_ids:
            benchmark_key = str(benchmark_id)
            if benchmark_key not in benchmark_index:
                raise KeyError(f"Unknown benchmark id in split manifest: {benchmark_key}")

            base_entry = dict(benchmark_index[benchmark_key])
            resolved_entries.append(
                {
                    "id": benchmark_key,
                    "stage": base_entry["stage"],
                    "path": str(resolve_path(base_entry["path"], root=root)),
                    "role": base_entry.get("role", ""),
                    "notes": base_entry.get("notes", ""),
                    "split": str(split_name),
                    "split_role": split_info.get("role", ""),
                }
            )
        resolved_partitions[str(split_name)] = resolved_entries

    return {
        "split_manifest_name": split_manifest.get("split_manifest_name", "unnamed_split_manifest"),
        "split_manifest_version": split_manifest.get("split_manifest_version", 1),
        "strategy": split_manifest.get("strategy", "unspecified"),
        "source_manifest": str(benchmark_manifest_path),
        "description": split_manifest.get("description", ""),
        "settings": split_manifest.get("settings", {}),
        "partitions": resolved_partitions,
    }
