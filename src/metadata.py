from __future__ import annotations

import re
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import pandas as pd

try:
    from .config import DATA_DIR
except ImportError:  # pragma: no cover - supports direct imports from notebooks
    from config import DATA_DIR


METADATA_PATH = DATA_DIR / "metadata.xlsx"
SEX_LABELS = {"F": 0, "M": 1}
SEX_NAMES = {0: "female", 1: "male"}


def load_subject_metadata(path: Path = METADATA_PATH) -> pd.DataFrame:
    """
    Load subject/session metadata from ``metadata.xlsx``.

    The function first tries pandas' Excel reader, and falls back to a small
    XLSX XML parser so the project can run even when ``openpyxl`` is missing.
    """
    path = Path(path)
    try:
        return _load_with_pandas(path)
    except ImportError:
        return _load_xlsx_without_openpyxl(path)
    except ModuleNotFoundError:
        return _load_xlsx_without_openpyxl(path)


def participant_sex_map(metadata: pd.DataFrame) -> dict[str, str]:
    """Return one sex value per participant, validating session consistency."""
    values = metadata.groupby("participant_id")["sex"].nunique(dropna=True)
    inconsistent = values[values > 1]
    if not inconsistent.empty:
        raise ValueError(
            f"Inconsistent sex metadata for participants: {inconsistent.index.tolist()}"
        )
    return metadata.groupby("participant_id")["sex"].first().to_dict()


def sex_class_names() -> tuple[str, ...]:
    return tuple(SEX_NAMES[index] for index in sorted(SEX_NAMES))


def _load_with_pandas(path: Path) -> pd.DataFrame:
    sheets = pd.read_excel(path, sheet_name=None, skiprows=[1])
    frames = []

    for sheet_name, sheet in sheets.items():
        if not sheet_name.startswith("Session_"):
            continue
        sheet = sheet.copy()
        sheet["session"] = _normalise_session_name(sheet_name)
        frames.append(sheet)

    if not frames:
        raise ValueError(f"No Session_* sheets found in {path}.")

    return _normalise_metadata_frame(pd.concat(frames, ignore_index=True))


def _load_xlsx_without_openpyxl(path: Path) -> pd.DataFrame:
    ns = {
        "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
        "office_rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }

    with ZipFile(path) as archive:
        shared_strings = _read_shared_strings(archive, ns)
        sheet_paths = _workbook_sheet_paths(archive, ns)
        frames = []

        for sheet_name, sheet_path in sheet_paths.items():
            if not sheet_name.startswith("Session_"):
                continue
            rows = _read_sheet_rows(archive, sheet_path, shared_strings, ns)
            if len(rows) < 3:
                continue

            header = rows[0]
            data_rows = rows[2:]
            frame = pd.DataFrame(data_rows, columns=header)
            frame["session"] = _normalise_session_name(sheet_name)
            frames.append(frame)

    if not frames:
        raise ValueError(f"No Session_* sheets found in {path}.")

    return _normalise_metadata_frame(pd.concat(frames, ignore_index=True))


def _normalise_metadata_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.rename(
        columns={
            "ID": "participant_id",
            "Sex": "sex",
            "Age": "age",
            "Bodymass": "bodymass",
            "Height": "height",
        }
    )
    frame = frame.loc[frame["participant_id"].notna()].copy()
    frame["participant_id"] = frame["participant_id"].astype(str)
    frame["sex"] = frame["sex"].astype(str).str.upper().str.strip()
    frame["sex_label"] = frame["sex"].map(SEX_LABELS)
    frame["sex_name"] = frame["sex_label"].map(SEX_NAMES)

    for column in ["age", "bodymass", "height"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    missing_sex = frame.loc[frame["sex_label"].isna(), "participant_id"].tolist()
    if missing_sex:
        raise ValueError(f"Missing or unknown sex labels for participants: {missing_sex}")

    return frame.reset_index(drop=True)


def _read_shared_strings(archive: ZipFile, ns: dict[str, str]) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []

    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    strings = []
    for item in root.findall("main:si", ns):
        strings.append("".join(text.text or "" for text in item.findall(".//main:t", ns)))
    return strings


def _workbook_sheet_paths(archive: ZipFile, ns: dict[str, str]) -> dict[str, str]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))

    relationship_targets = {
        item.attrib["Id"]: item.attrib["Target"]
        for item in rels.findall("rel:Relationship", ns)
    }

    sheet_paths = {}
    for sheet in workbook.findall(".//main:sheets/main:sheet", ns):
        sheet_name = sheet.attrib["name"]
        rel_id = sheet.attrib[f"{{{ns['office_rel']}}}id"]
        target = relationship_targets[rel_id].lstrip("/")
        if not target.startswith("xl/"):
            target = f"xl/{target}"
        sheet_paths[sheet_name] = target

    return sheet_paths


def _read_sheet_rows(
    archive: ZipFile,
    sheet_path: str,
    shared_strings: list[str],
    ns: dict[str, str],
) -> list[list[str]]:
    sheet = ET.fromstring(archive.read(sheet_path))
    rows = []

    for row in sheet.findall(".//main:sheetData/main:row", ns):
        values = []
        for cell in row.findall("main:c", ns):
            column_index = _cell_column_index(cell.attrib["r"])
            while len(values) < column_index:
                values.append("")
            values.append(_cell_value(cell, shared_strings, ns))
        rows.append(values)

    width = max(len(row) for row in rows)
    return [row + [""] * (width - len(row)) for row in rows]


def _cell_value(cell: ET.Element, shared_strings: list[str], ns: dict[str, str]) -> str:
    cell_type = cell.attrib.get("t")
    value_node = cell.find("main:v", ns)

    if cell_type == "inlineStr":
        return "".join(text.text or "" for text in cell.findall(".//main:t", ns))
    if value_node is None:
        return ""

    value = value_node.text or ""
    if cell_type == "s":
        return shared_strings[int(value)]
    return value


def _cell_column_index(cell_ref: str) -> int:
    letters = re.sub(r"[^A-Z]", "", cell_ref.upper())
    index = 0
    for letter in letters:
        index = index * 26 + ord(letter) - ord("A") + 1
    return index - 1


def _normalise_session_name(sheet_name: str) -> str:
    return sheet_name.replace("_", "")
