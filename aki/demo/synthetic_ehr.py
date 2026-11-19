"""Fully synthetic EHR extract in the TriNetX CSV format, for demos and tests.

Nothing here is derived from the real data: every patient, date, value and
outcome is drawn from a seeded random generator. Codes come from short
hand-written lists of common public codes (ICD-10-CM, CPT, HCPCS, RxNorm,
LOINC) so the files exercise the real pipeline: AKI index (N17.x),
pre-existing ESRD exclusion (N18.6), dialysis outcomes (CPT 90935 etc.),
and kidney labs in the units the binning rules expect.

The outcome is generated from a latent risk score that also drives CKD
diagnoses and kidney lab values, so models have a signal, but the numbers
have no clinical meaning.

    python -m aki.demo.synthetic_ehr --out outputs/demo_data --n 600 --seed 2026
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from aki.ingest.trinetx import DIAGNOSIS, LAB, MEDICATION, PATIENT, PROCEDURE

COMMON_DX = ["I10", "E11.9", "E78.5", "I50.9", "J44.9", "Z79.4", "K21.9", "M54.5", "F32.9", "E66.9",
             "I25.10", "R07.9", "N39.0", "J18.9", "D64.9", "E87.1", "Z00.00", "R53.83", "I48.91", "G47.33"]
CKD_DX = ["N18.30", "N18.4", "N18.5", "E11.22", "I12.9", "N28.9"]
COMMON_CPT = ["99213", "99214", "80053", "85025", "36415", "93000", "71046", "80048", "83036", "81001"]
DIALYSIS = [("CPT", "90935"), ("CPT", "90937"), ("HCPCS", "G0257")]
MEDS = [("314076", "Lisinopril"), ("197361", "Amlodipine"), ("860975", "Metformin"), ("310798", "Hydrochlorothiazide"),
        ("617312", "Atorvastatin"), ("198211", "Simvastatin"), ("310429", "Furosemide"), ("311372", "Omeprazole")]
RACES = ["White", "Black or African American", "Asian", "Other Race", "Unknown",
         "American Indian or Alaska Native", "Native Hawaiian or Other Pacific Islander"]
RACE_P = [0.62, 0.17, 0.05, 0.04, 0.10, 0.01, 0.01]


def _d(x: date) -> str:
    return x.strftime("%Y%m%d")


def generate(out_dir: Path, n: int = 600, seed: int = 2026) -> dict:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    patients, dx, px, meds, labs = [], [], [], [], []
    enc = 0
    counts = {"patients": n, "aki": 0, "preexisting_esrd": 0, "positive_outcome": 0}

    for i in range(n):
        pid = f"SYN{i:06d}"
        risk = rng.normal()
        sex = rng.choice(["F", "M", ""], p=[0.495, 0.495, 0.01])
        yob = int(rng.integers(1930, 1996))
        protected = rng.random() < 0.05
        patients.append([pid, sex, rng.choice(RACES, p=RACE_P),
                         rng.choice(["Not Hispanic or Latino", "Hispanic or Latino", "Unknown"], p=[0.72, 0.08, 0.20]),
                         rng.choice(["Married", "Single", "Unknown"]), "" if protected else str(yob),
                         "Protected" if protected else "Present", "", "",
                         rng.choice(["South", "Northeast", "Midwest", "West"]), "EHR"])
        start = date(2008, 1, 1) + timedelta(days=int(rng.integers(0, 365 * 6)))
        has_aki = rng.random() < 0.8
        index = start + timedelta(days=int(rng.integers(365 * 2, 365 * 8)))
        end_hist = index + timedelta(days=int(rng.integers(30, 900)))

        # visits before (and some after) the index date
        n_visits = max(2, int(rng.lognormal(np.log(25), 0.7)))
        for _ in range(n_visits):
            enc += 1
            e = f"SE{enc}"
            vd = start + timedelta(days=int(rng.integers(0, max(1, (end_hist - start).days))))
            for code in rng.choice(COMMON_DX, size=int(rng.integers(1, 4)), replace=False):
                dx.append([pid, e, "ICD-10-CM", code, rng.choice(["P", "S", "Unknown"]), "U", "U", _d(vd), "F", "EHR"])
            if rng.random() < 1 / (1 + np.exp(-(risk - 0.5) * 2)):
                dx.append([pid, e, "ICD-10-CM", rng.choice(CKD_DX), "Unknown", "U", "U", _d(vd), "F", "EHR"])
            for code in rng.choice(COMMON_CPT, size=int(rng.integers(0, 3)), replace=False):
                px.append([pid, e, "CPT", code, "Unknown", _d(vd), "F", "EHR"])
            if rng.random() < 0.6:
                code, brand = MEDS[int(rng.integers(len(MEDS)))]
                meds.append([pid, e, f"U{enc}", "RxNorm", code, _d(vd), "Oral Product", brand, "Unknown", "30.000000", "30", "F", "EHR"])
            if rng.random() < 0.5:
                creat = float(np.clip(rng.lognormal(np.log(1.0) + 0.35 * max(risk, -1), 0.25), 0.3, 12))
                egfr = float(np.clip(100 / creat ** 1.1 + rng.normal(0, 5), 3, 140))
                bun = float(np.clip(15 * creat ** 0.8 + rng.normal(0, 3), 4, 150))
                if rng.random() < 0.05:   # some creatinine in umol/L
                    labs.append([pid, e, "LOINC", "14682-9", _d(vd), f"{creat * 88.4:.1f}", "", "umol/L", "F", "EHR"])
                else:
                    labs.append([pid, e, "LOINC", "2160-0", _d(vd), f"{creat:.2f}", "", "mg/dL", "F", "EHR"])
                labs.append([pid, e, "LOINC", rng.choice(["33914-3", "62238-1", "88293-6"]), _d(vd), f"{egfr:.1f}", "",
                             "mL/min/{1.73_m2}", "F", "EHR"])
                labs.append([pid, e, "LOINC", "3094-0", _d(vd), f"{bun:.0f}", "", "mg/dL", "F", "EHR"])
            if rng.random() < 0.1:  # non-kidney lab
                labs.append([pid, e, "LOINC", "4548-4", _d(vd), f"{rng.normal(6.5, 1):.1f}", "", "%", "F", "EHR"])

        if has_aki:
            counts["aki"] += 1
            enc += 1
            dx.append([pid, f"SE{enc}", "ICD-10-CM", rng.choice(["N17.9", "N17.0"]), "P", "U", "U", _d(index), "F", "EHR"])
            if rng.random() < 0.05:   # pre-existing ESRD -> excluded from cohort
                counts["preexisting_esrd"] += 1
                d0 = index - timedelta(days=int(rng.integers(1, 700)))
                dx.append([pid, f"SE{enc}", "ICD-10-CM", "N18.6", "Unknown", "U", "U", _d(d0), "F", "EHR"])
            elif rng.random() < 1 / (1 + np.exp(-(-2.2 + 1.3 * risk))):
                counts["positive_outcome"] += 1
                d1 = index + timedelta(days=int(rng.integers(20, 700)))
                if rng.random() < 0.6:
                    dx.append([pid, f"SE{enc}", "ICD-10-CM", "N18.6", "Unknown", "U", "U", _d(d1), "F", "EHR"])
                system, code = DIALYSIS[int(rng.integers(len(DIALYSIS)))]
                px.append([pid, f"SE{enc}", system, code, "Unknown", _d(d1), "F", "EHR"])

    for spec, rows, fname in [(PATIENT, patients, "patient.csv"), (DIAGNOSIS, dx, "diagnosis.csv"),
                              (PROCEDURE, px, "procedure.csv"), (MEDICATION, meds, "medication_drug.csv"),
                              (LAB, labs, "lab_result_clean.csv")]:
        with (out_dir / fname).open("w", newline="") as f:
            w = csv.writer(f, quoting=csv.QUOTE_ALL)
            w.writerow([c for c, _ in spec.columns])
            w.writerows(rows)
    (out_dir / "SYNTHETIC_DATA.txt").write_text(
        f"Synthetic data generated by aki.demo.synthetic_ehr (n={n}, seed={seed}). Not real patients.\n")
    counts.update(diagnosis_rows=len(dx), procedure_rows=len(px), medication_rows=len(meds), lab_rows=len(labs))
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    print(generate(a.out, a.n, a.seed))


if __name__ == "__main__":
    main()
