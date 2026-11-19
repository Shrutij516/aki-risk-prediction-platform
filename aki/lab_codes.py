"""Kidney-lab definitions copied verbatim from the original research.

Source: AKI_Dialysis_Prediction-main/RETAIN_WithLabs.ipynb, cell 80
("Phase 2: curated LOINC codes + clinical binning"). Do not edit these values
without recording the change in PROVENANCE.md; they define the original
lab tokens.
"""
from __future__ import annotations

CREATININE_CODES = frozenset({"2160-0", "38483-4", "14682-9", "21232-4", "35203-9", "35591-7"})
EGFR_CODES = frozenset({"33914-3", "48642-3", "48643-1", "62238-1", "88293-6", "88294-4", "50044-7", "50210-4"})
BUN_CODES = frozenset({"3094-0", "6299-2", "11065-0"})
ALL_KIDNEY_CODES = CREATININE_CODES | EGFR_CODES | BUN_CODES

# Units accepted per test in the notebook (rows with other units were dropped).
VALID_UNITS = {
    "creatinine": frozenset({"mg/dL", "umol/L"}),
    "egfr": frozenset({"mL/min/{1.73_m2}"}),
    "bun": frozenset({"mg/dL"}),
}
UMOL_PER_MGDL_CREATININE = 88.4
PLAUSIBLE = {"creatinine": (0.1, 30.0), "egfr": (0.0, 200.0), "bun": (1.0, 200.0)}


def test_type(code: str) -> str | None:
    if code in CREATININE_CODES:
        return "creatinine"
    if code in EGFR_CODES:
        return "egfr"
    if code in BUN_CODES:
        return "bun"
    return None


def severity_bin(test: str, v: float) -> str:
    """Notebook binning: creatinine thresholds, KDIGO eGFR stages, BUN ranges."""
    if test == "creatinine":
        return "creatinine_" + ("normal" if v < 1.2 else "mild" if v < 2.0 else "moderate" if v < 4.0 else "severe")
    if test == "egfr":
        stage = "g1" if v >= 90 else "g2" if v >= 60 else "g3a" if v >= 45 else "g3b" if v >= 30 else "g4" if v >= 15 else "g5"
        return "egfr_" + stage
    if test == "bun":
        return "bun_" + ("normal" if v < 20 else "elevated" if v < 50 else "high")
    raise ValueError(f"unknown test {test!r}")
