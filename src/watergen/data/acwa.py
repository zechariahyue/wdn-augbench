"""ACWA workbook helpers for a lightweight OOD/anomaly case study."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


SHEET_ALIASES = {
    "Water_Levels": "Water_Level",
    "Water_Level": "Water_Level",
    "Water_Pressure": "Water_Pressure",
    "Water_Flow": "Water_Flow",
    "Turbidity": "Turbidity",
    "Nitrate": "Nitrate",
}


def _load_excel(path: Path | str) -> pd.ExcelFile:
    return pd.ExcelFile(path)


def _normalize_sheet_name(name: str) -> str:
    return SHEET_ALIASES.get(name, name)


def _sheet_mapping(excel_file: pd.ExcelFile) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for original in excel_file.sheet_names:
        normalized = _normalize_sheet_name(original)
        if normalized not in mapping:
            mapping[normalized] = original
    return mapping


def _numeric_common_columns(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    common = [name for name in left.columns if name in right.columns]
    numeric: list[str] = []
    for name in common:
        left_num = pd.to_numeric(left[name], errors="coerce")
        right_num = pd.to_numeric(right[name], errors="coerce")
        if left_num.notna().sum() > 0 and right_num.notna().sum() > 0:
            numeric.append(name)
    return numeric


def build_acwa_case_study_frame(
    normal_workbook: Path | str,
    attack_workbook: Path | str,
    *,
    preferred_sheets: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Build a row-level ACWA case-study frame from normal and attack workbooks."""
    normal_xl = _load_excel(normal_workbook)
    attack_xl = _load_excel(attack_workbook)
    normal_map = _sheet_mapping(normal_xl)
    attack_map = _sheet_mapping(attack_xl)

    selected_sheets = list(preferred_sheets or ["Water_Level", "Water_Pressure", "Water_Flow"])
    frames: list[pd.DataFrame] = []

    for normalized_sheet in selected_sheets:
        if normalized_sheet not in normal_map or normalized_sheet not in attack_map:
            continue

        normal_df = pd.read_excel(normal_xl, sheet_name=normal_map[normalized_sheet])
        attack_df = pd.read_excel(attack_xl, sheet_name=attack_map[normalized_sheet])
        common_numeric = _numeric_common_columns(normal_df, attack_df)
        if not common_numeric:
            continue

        keep_columns = common_numeric.copy()
        if "Time" in normal_df.columns and "Time" not in keep_columns and "Time" in attack_df.columns:
            keep_columns.append("Time")
        if "UTCmsec" in normal_df.columns and "UTCmsec" not in keep_columns and "UTCmsec" in attack_df.columns:
            keep_columns.append("UTCmsec")

        normal_subset = normal_df.loc[:, keep_columns].copy()
        attack_subset = attack_df.loc[:, keep_columns].copy()

        normal_subset["sheet_name"] = normalized_sheet
        attack_subset["sheet_name"] = normalized_sheet
        normal_subset["source_label"] = 0
        attack_subset["source_label"] = 1
        normal_subset["row_origin"] = "normal"
        attack_subset["row_origin"] = "attack"

        frames.extend([normal_subset, attack_subset])

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=True)
