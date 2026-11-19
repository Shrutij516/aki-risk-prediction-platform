"""Pre-commit safety check: would anything unsafe be committed?

Lists the files git would add (using a throwaway GIT_DIR outside the project,
so the workspace itself needs no repository), then checks each one:

  FAIL  path rules     data files, databases, pgdata, .env, checkpoints, mlruns,
                       original notebooks (patient-level outputs)
  FAIL  size           any file > 5 MB
  FAIL  secrets        private keys, cloud keys, tokens, DB URLs with passwords
  FAIL  notebooks      any saved output at all (public copies are code + markdown only)
  FAIL  names/figures  names listed in local/publication.json; extract-specific figures (>= 10,000)
                       taken from local/reference_counts.json and the local reports
  FAIL  patient ids    >= 10 distinct real patient_ids (len >= 3) in one file
  WARN  patient ids    3-9 distinct id-like tokens (usually English/code words: review)

Needs data/patient.csv locally for the patient-id checks (FAILs if missing).
Exit code 0 = safe, 1 = problems found. Prints file paths and counts only.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sanitize_notebooks as san  # noqa: E402

MAX_BYTES = 5 * 1024 * 1024
# Binary files reviewed by hand (2026-11-17): text + all 11 embedded images contain
# aggregate results only. Any change to the file invalidates the review.
REVIEWED_BINARIES = {
    "docs/Progress report.docx": "983247b71d90ea689dd994930a10721bb62d5311614ac07f937bf97c22d300fb",
}
BAD_PATH = re.compile(
    r"(^|/)(data|\.pgdata|pgdata|mlruns|mlartifacts|checkpoints|outputs|reports|\.venv)/"
    r"|\.(csv|csv\.gz|parquet|feather|duckdb|sqlite|db|pt|pth|ckpt|h5|keras|onnx|safetensors|pem|key)$"
    r"|(^|/)\.env($|\.(?!example$))"
    r"|^AKI_Dialysis_Prediction-main/.*\.ipynb$"
)
SECRET_RES = {
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "aws key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    "api key": re.compile(r"\b(sk|rk)-[A-Za-z0-9_-]{20,}\b"),
    "db url with password": re.compile(r"postgres(ql)?://[^:/\s@]+:(?!PASSWORD@)[^@\s/]+@"),
    # value: 8+ chars, no whitespace, not a <placeholder> template
    "assigned secret": re.compile(r"(?i)\b(password|passwd|secret|api_key|token)\s*[=:]\s*['\"](?!<)[^'\"\s<>]{8,}['\"]"),
}


def candidate_files() -> list[str]:
    with tempfile.TemporaryDirectory() as gd:
        git = ["git", f"--git-dir={gd}/g", f"--work-tree={ROOT}"]
        subprocess.run(["git", "init", "-q", "--bare", f"{gd}/g"], check=True)
        out = subprocess.run(git + ["ls-files", "--others", "--exclude-standard", "-z"],
                             check=True, capture_output=True, text=True, cwd=ROOT).stdout
    return sorted(p for p in out.split("\0") if p)


def _numbers(obj, out: set) -> set:
    if isinstance(obj, bool):
        return out
    if isinstance(obj, int) and obj >= 10000:
        out.add(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _numbers(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _numbers(v, out)
    return out


def restricted_terms() -> tuple[list[str], list[re.Pattern], list[str]]:
    """Forbidden names and extract-specific figures, gathered from LOCAL files only.

    Figures: integers >= 10,000 in local/reference_counts.json and the local validation/profile
    reports (cohort sizes, event and row counts, vocabulary-derived parameter totals), matched
    with or without thousands separators.
    """
    problems = []
    rules = ROOT / "local" / "publication.json"
    names = [t.lower() for t in json.loads(rules.read_text()).get("forbidden_terms", [])] if rules.exists() else []
    if not rules.exists():
        problems.append("local/publication.json missing: cannot check forbidden names")
    nums: set[int] = set()
    sources = [ROOT / "local" / "reference_counts.json", ROOT / "reports" / "cohort_validation.json",
               ROOT / "reports" / "sequence_validation.json", ROOT / "reports" / "data_profile.json",
               ROOT / "reports" / "extra_checks.json"]
    for src in sources:
        if src.exists():
            _numbers(json.loads(src.read_text()), nums)
        else:
            problems.append(f"{src.relative_to(ROOT)} missing: extract-figure check is incomplete")
    pats = [re.compile(rf"(?<![\d,.]){n}(?![\d,])|(?<![\d,.]){n:,}(?![\d])") for n in sorted(nums)]
    return names, pats, problems


def main() -> int:
    ids = san.load_patient_ids()
    names, figure_pats, problems = restricted_terms()
    files = candidate_files()
    fails, warns = [], []
    fails += [("local", p) for p in problems]
    if ids is None:
        fails.append(("data/patient.csv", "missing: cannot run patient-id checks"))
    for rel in files:
        path = ROOT / rel
        if BAD_PATH.search(rel):
            fails.append((rel, "forbidden path/type"))
            continue
        if path.stat().st_size > MAX_BYTES:
            fails.append((rel, f"size {path.stat().st_size / 1e6:.1f} MB"))
        if rel in REVIEWED_BINARIES:
            import hashlib

            if hashlib.sha256(path.read_bytes()).hexdigest() != REVIEWED_BINARIES[rel]:
                fails.append((rel, "reviewed binary changed since review"))
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            fails.append((rel, "binary file not reviewed"))
            continue
        for name, rx in SECRET_RES.items():
            if rx.search(text):
                fails.append((rel, f"possible secret: {name}"))
        if rel.endswith(".ipynb"):
            nb = json.loads(text)
            n_out = sum(len(c.get("outputs", [])) for c in nb["cells"])
            if n_out:
                fails.append((rel, f"{n_out} saved outputs (public notebooks must have none)"))
            extra_meta = set(nb.get("metadata", {})) - set(san.KEEP_NB_META)
            if extra_meta:
                fails.append((rel, f"notebook metadata not stripped: {sorted(extra_meta)}"))
        low = text.lower()
        for name in names:
            if name in low:
                fails.append((rel, "contains a name listed in local/publication.json"))
        fig_hits = sum(1 for p in figure_pats if p.search(text))
        if fig_hits:
            fails.append((rel, f"{fig_hits} extract-specific figure(s) from local reports"))
        if ids:
            hits = {t for t in san.TOKEN_RE.findall(text) if len(t) >= 3 and t in ids}
            if len(hits) >= 10:
                fails.append((rel, f"{len(hits)} distinct real patient_id tokens"))
            elif len(hits) >= 3:
                warns.append((rel, f"{len(hits)} id-like tokens: {sorted(hits)}"))
    print(f"{len(files)} files would be committed")
    for rel, why in warns:
        print(f"WARN {rel}: {why}")
    for rel, why in fails:
        print(f"FAIL {rel}: {why}")
    print("SAFE" if not fails else f"UNSAFE: {len(fails)} problem(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
