"""Make public-safe copies of the original notebooks.

Originals in AKI_Dialysis_Prediction-main/ are never modified. Copies go to
notebooks/sanitized/ with this conservative publication policy:

  * ALL saved outputs are removed (they contain extract-specific counts; the original
    results are documented elsewhere). Colab identity metadata is stripped.
  * Code and markdown are kept, except for logged placeholder rewrites:
      - personal absolute paths (/Users/<name>/...) -> /path/to/
      - replacements listed in local/publication.json (gitignored): names that should not be
        published and extract-specific literals, so this public script does not contain them
      - count-like figures in markdown narrative -> "[figure withheld]"
  * The run fails if a local path or a forbidden term from the local rules remains.

reasons_to_drop() (the earlier per-output allow-list) is kept: scripts/check_commit_safety.py
uses it, and it documents what an output would have to avoid to be publishable.
"""
from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "AKI_Dialysis_Prediction-main"
DST = ROOT / "notebooks" / "sanitized"

DATE_RE = re.compile(r"\b(19|20)\d\d[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b"
                     r"|\b(19|20)\d\d(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\b")
CODE_RES = {
    "icd10": re.compile(r"\b[A-Z]\d{2}\.\d{1,4}\b"),
    "icd9": re.compile(r"(?<![\d.])\d{3}\.\d{1,2}(?![\d.])"),
    "loinc": re.compile(r"\b\d{3,5}-\d\b"),
    "event_token": re.compile(r"\b(diagnosis|procedure|medication|lab)_[A-Za-z0-9.#|-]+"),
}
ID_FIELD_RE = re.compile(r"patient_id|encounter_id|unique_id")
# Absolute local filesystem paths (reveal usernames / machine layout), e.g. library warnings.
LOCAL_PATH_RE = re.compile(r"/Users/|/home/|[A-Za-z]:\\\\Users\\\\")
# In code/markdown cells, the personal directory part of an absolute path is replaced by a
# portable placeholder; the final path component (e.g. the dataset folder name) is kept.
PERSONAL_DIR_RE = re.compile(r"(?:/Users|/home)/[^\"'\n]*/")
PATH_PLACEHOLDER = "/path/to/"
TOKEN_RE = re.compile(r"[A-Za-z0-9#]+")
KEEP_NB_META = ("kernelspec", "language_info")
KEEP_CELL_META = ("id",)


def load_patient_ids() -> set[str] | None:
    p = ROOT / "data" / "patient.csv"
    if not p.exists():
        return None
    import duckdb

    return {r[0] for r in duckdb.sql(f"SELECT DISTINCT patient_id FROM read_csv('{p}', all_varchar=true)").fetchall()}


def output_text(o: dict) -> str | None:
    if o["output_type"] == "stream":
        return "".join(o.get("text", []))
    if o["output_type"] in ("execute_result", "display_data"):
        data = o.get("data", {})
        if set(data) - {"text/plain"}:
            return None  # html / colab table / images -> not allow-listed
        return "".join(data.get("text/plain", []))
    return None  # error


def reasons_to_drop(o: dict, ids: set[str] | None) -> list[str]:
    if ids is None:
        return ["no patient list available"]
    if o["output_type"] == "error":
        return ["error traceback"]
    text = output_text(o)
    if text is None:
        return ["non-text render (" + ",".join(sorted(o.get("data", {}))) + ")"]
    why = []
    if ID_FIELD_RE.search(text):
        why.append("id column name")
    if LOCAL_PATH_RE.search(text):
        why.append("local file path")
    if DATE_RE.search(text):
        why.append("date")
    for name, rx in CODE_RES.items():
        if rx.search(text):
            why.append(name)
    hits = {t for t in TOKEN_RE.findall(text) if t in ids}
    if hits:
        why.append(f"patient_id tokens x{len(hits)}")
    return why


LOCAL_RULES = ROOT / "local" / "publication.json"
# Markdown narrative: count-like figures from the extract (e.g. "~1.23M events", "12,345", 5+ digit numbers).
FIGURE_RE = re.compile(r"~?\d{1,3}(?:,\d{3})+(?:\.\d+)?|~?\d+(?:\.\d+)?\s?M\b|\b\d{5,}\b")
FIGURE_PLACEHOLDER = "[figure withheld]"


def load_local_rules() -> dict:
    """Name and literal replacements live in a gitignored file so the public script does not contain them."""
    if not LOCAL_RULES.exists():
        raise SystemExit(f"missing {LOCAL_RULES}: local publication rules are required to build public copies")
    rules = json.loads(LOCAL_RULES.read_text())
    rules["compiled"] = [(re.compile(p), r) for p, r in rules.get("text_replacements", [])]
    return rules


def _apply(line: str, rules: dict, markdown: bool, counts: dict) -> str:
    new = PERSONAL_DIR_RE.sub(PATH_PLACEHOLDER, line)
    counts["path"] += new != line
    for rx, repl in rules["compiled"]:
        new2 = rx.sub(repl, new)
        counts["local_rule"] += new2 != new
        new = new2
    if markdown:
        new2 = FIGURE_RE.sub(FIGURE_PLACEHOLDER, new)
        counts["figure"] += new2 != new
        new = new2
    return new


def sanitize(path: Path, ids: set[str] | None, rules: dict) -> tuple[dict, dict]:
    """Public copy: all saved outputs removed; sources kept except for logged placeholder rewrites."""
    nb = json.loads(path.read_text())
    out = copy.deepcopy(nb)
    out["metadata"] = {k: v for k, v in nb["metadata"].items() if k in KEEP_NB_META}
    log = {"outputs_removed": 0, "code_cells": 0, "markdown_cells": 0, "rewrites": [], "source_flags": []}
    for i, cell in enumerate(out["cells"]):
        cell["metadata"] = {k: v for k, v in cell.get("metadata", {}).items() if k in KEEP_CELL_META}
        lines = cell["source"] if isinstance(cell["source"], list) else cell["source"].splitlines(keepends=True)
        counts = {"path": 0, "local_rule": 0, "figure": 0}
        new_lines = [_apply(ln, rules, cell["cell_type"] == "markdown", counts) for ln in lines]
        if any(counts.values()):
            log["rewrites"].append({"cell": i, "type": cell["cell_type"], **{k: v for k, v in counts.items() if v}})
            cell["source"] = new_lines
        src = "".join(new_lines)
        if LOCAL_PATH_RE.search(src):
            raise RuntimeError(f"{path.name} cell {i}: local path left after rewrite")
        for term in rules.get("forbidden_terms", []):
            if term.lower() in src.lower():
                raise RuntimeError(f"{path.name} cell {i}: forbidden term left after rewrite")
        src_hits = {t for t in TOKEN_RE.findall(src) if ids and t in ids and len(t) >= 3}
        if DATE_RE.search(src) or src_hits:
            log["source_flags"].append({"cell": i, "date": bool(DATE_RE.search(src)), "id_like_tokens": len(src_hits)})
        if cell["cell_type"] == "code":
            log["code_cells"] += 1
            log["outputs_removed"] += len(cell.get("outputs", []))
            cell["outputs"] = []
            cell["execution_count"] = None
        else:
            log["markdown_cells"] += cell["cell_type"] == "markdown"
    return out, log


def main() -> int:
    ids = load_patient_ids()
    rules = load_local_rules()
    DST.mkdir(parents=True, exist_ok=True)
    manifest = {"source_dir": SRC.name, "policy": "all saved outputs removed; sources kept except logged "
                "placeholder rewrites (personal paths, local publication rules, markdown figures)",
                "patient_id_check": ids is not None, "notebooks": {}}
    for path in sorted(SRC.glob("*.ipynb")):
        before = path.read_bytes()
        clean, log = sanitize(path, ids, rules)
        assert path.read_bytes() == before  # original untouched
        (DST / path.name).write_text(json.dumps(clean, indent=1, ensure_ascii=False) + "\n")
        manifest["notebooks"][path.name] = log
        print(f"{path.name}: outputs removed {log['outputs_removed']}, code cells {log['code_cells']}, "
              f"cells rewritten {len(log['rewrites'])}")
    (DST / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
