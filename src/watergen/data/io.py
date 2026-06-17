"""Dataset discovery and manifest helpers for the scaffold."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, List, Sequence


def discover_files(root: Path, patterns: Iterable[str]) -> List[Path]:
    """Return sorted files under ``root`` matching one or more glob patterns."""
    matches: List[Path] = []
    for pattern in patterns:
        matches.extend(root.rglob(pattern))
    return sorted(set(path for path in matches if path.is_file()))


def load_inp_paths(
    manifest_or_paths: Path | str | Sequence[Path | str],
    *,
    root: Path | None = None,
) -> List[Path]:
    """Resolve EPANET ``.inp`` paths from a manifest file or explicit list.

    Args:
        manifest_or_paths: A manifest file path (JSON/YAML/text), a single
            ``.inp`` path, or a sequence of ``.inp`` paths.
        root: Optional root used to resolve relative paths. Defaults to CWD.

    Returns:
        Sorted, de-duplicated existing ``.inp`` file paths.
    """
    base = (root or Path.cwd()).resolve()
    raw_paths: List[str] = []

    if isinstance(manifest_or_paths, (str, Path)):
        source = Path(manifest_or_paths)
        if source.suffix.lower() == ".inp":
            raw_paths = [str(source)]
        else:
            raw_paths = _load_paths_from_manifest(source)
    else:
        raw_paths = [str(item) for item in manifest_or_paths]

    resolved: List[Path] = []
    for item in raw_paths:
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = (base / candidate).resolve()
        if candidate.suffix.lower() == ".inp" and candidate.is_file():
            resolved.append(candidate)

    return sorted(set(resolved))


def _load_paths_from_manifest(path: Path) -> List[str]:
    """Load candidate paths from a manifest file."""
    if not path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        content = json.loads(path.read_text(encoding="utf-8"))
        return _extract_inp_strings(content)

    # Keep YAML/text parsing dependency-free by extracting path-like substrings.
    text = path.read_text(encoding="utf-8")
    return _extract_inp_strings(text)


def _extract_inp_strings(content: Any) -> List[str]:
    """Extract ``.inp`` strings from nested content or free-form text."""
    matches: List[str] = []

    if isinstance(content, dict):
        for value in content.values():
            matches.extend(_extract_inp_strings(value))
        return matches

    if isinstance(content, list):
        for item in content:
            matches.extend(_extract_inp_strings(item))
        return matches

    if isinstance(content, str):
        # Capture both plain paths and quoted values ending in .inp.
        pattern = re.compile(r"([A-Za-z0-9_./\\:\-\s]+\.inp)\b", re.IGNORECASE)
        for hit in pattern.findall(content):
            cleaned = hit.strip().strip('"').strip("'")
            if cleaned.lower().endswith(".inp"):
                matches.append(cleaned)
        return matches

    return matches
