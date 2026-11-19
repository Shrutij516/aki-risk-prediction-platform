# Data inventory and comparison with the original notebooks

*Reconstruction document (2026-11-17, public version).*

**Public version, counts removed.** Row counts, cohort sizes, value frequencies and other figures from the TriNetX extract have been taken out until the right to publish them is confirmed. The detailed version and the profiling output exist only on the author's machine (`reports/`, gitignored). The original project README already reports a few cohort-level figures; this document adds none.

## Files in the recovered extract (not in this repository)

| File | Contents | Notes |
|---|---|---|
| `patient.csv` | One row per patient: sex, race, ethnicity, marital status, year of birth, death month, region | Year of birth is "Protected" for some patients; death date is `YYYYMM` |
| `diagnosis.csv` | ICD-10-CM / ICD-9-CM diagnoses with dates | A small number of TriNetX-derived rows have an empty `code_system` |
| `procedure.csv` | CPT, HCPCS, SNOMED, ICD-10-PCS and ICD-9-CM procedures with dates | |
| `medication_drug.csv` | RxNorm and NDC medication records with start dates | Quantity and days-supply are mostly empty |
| `lab_result.csv` | LOINC lab results with numeric/text values and units | |
| `lab_result_clean.csv` | Same as `lab_result.csv` | Content-identical: same order-independent hash over all fields; it only drops the quotes |

**Relationships:**
- `patient_id` is the only key shared by every table, and every event row's patient exists in `patient.csv` (checked exactly).
- There's no encounter table.
- All dates parse as `YYYYMMDD` and all numeric fields parse.
- `standardized_terminology.csv`, used by the original naive-LOINC notebooks, isn't in the recovered extract.

## How the extract compares with the notebooks

| Topic | Finding | What follows |
|---|---|---|
| Cohort | `cohort.build_aki_cohort()` reproduces **every** cohort and label count printed in the saved notebook outputs | The recovered CSVs are the same extract the notebooks used, and the SQL cohort is faithful |
| Lab table size | The README's lab row count is about an order of magnitude below the actual row count; the notebook's own log agrees with the actual size | The README figure is a typo-level error, not a different dataset |
| Exact duplicate rows | A substantial share of diagnosis and procedure rows are exact duplicates; the notebooks didn't remove them | Original sequences contain repeated tokens. Kept as-is in `ehr.*` |
| Medication coding | Most NDC prescriptions have an RxNorm twin (same patient, encounter, date, brand, route) | Medication tokens are inflated. Kept |
| Token namespace | Tokens are `event_type_code` without the code system | Possible cross-system collisions. `code_system` is stored in the raw layer |
| Creatinine `14682-9` | Rows labelled µmol/L hold mg/dL-sized values | The notebook's conversion and plausibility filter drop them. Kept as in the original |
| `35591-7` | Listed as a creatinine code but reported in mL/min | Already removed by the notebook's unit filter |
| Malformed lines | A strict parse of every file found none | `on_bad_lines='skip'` in the notebooks dropped nothing, as far as the current files show |
| Sequence truncation | `truncating='post'` keeps the oldest 400 events; median sequence length is about 400 | Many patients lost their most recent events. Recorded, not changed |
