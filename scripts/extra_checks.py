"""Targeted aggregate checks that inform schema/ingestion design (aggregates only)."""
import json, sys
from pathlib import Path
import duckdb
ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "data"
con = duckdb.connect(); con.execute("SET memory_limit='4GB'"); con.execute("SET threads=8")
def src(f): return f"read_csv('{D/f}', all_varchar=true, header=true)"
out = {}
cols = "patient_id,encounter_id,code_system,code,date,lab_result_num_val,lab_result_text_val,units_of_measure,derived_by_TriNetX,source_id"
# order-independent content fingerprint of each lab file
for f in ["lab_result.csv", "lab_result_clean.csv"]:
    out[f"fingerprint_{f}"] = con.execute(
        f"SELECT sum(hash(concat_ws('|', {', '.join(f'coalesce({c},chr(0))' for c in cols.split(','))}))::HUGEINT)::VARCHAR, count(*) FROM {src(f)}").fetchone()
# raw byte-level: is clean just the unquoted version?
out["raw_first_line_has_quotes"] = open(D/"lab_result.csv").readline().startswith('"')
out["clean_first_line_has_quotes"] = open(D/"lab_result_clean.csv").readline().startswith('"')
# medication: same order recorded under both NDC and RxNorm
out["med_rows_by_system"] = con.execute(f"SELECT code_system, count(*) FROM {src('medication_drug.csv')} GROUP BY 1").fetchall()
out["med_ndc_rows_with_rxnorm_twin"] = con.execute(f"""
  WITH m AS (SELECT patient_id, encounter_id, start_date, brand, route, code_system FROM {src('medication_drug.csv')})
  SELECT count(*) FROM m n WHERE code_system='NDC' AND EXISTS (
    SELECT 1 FROM m r WHERE r.code_system='RxNorm' AND r.patient_id=n.patient_id AND r.encounter_id=n.encounter_id
      AND r.start_date=n.start_date AND r.brand=n.brand AND r.route=n.route)""").fetchone()[0]
# exact duplicate rows
out["diagnosis_exact_dup_rows"] = con.execute(f"SELECT count(*) - count(DISTINCT (patient_id,encounter_id,code_system,code,principal_diagnosis_indicator,admitting_diagnosis,reason_for_visit,date,derived_by_TriNetX,source_id)) FROM {src('diagnosis.csv')}").fetchone()[0]
out["procedure_exact_dup_rows"] = con.execute(f"SELECT count(*) - count(DISTINCT (patient_id,encounter_id,code_system,code,principal_procedure_indicator,date,derived_by_TriNetX,source_id)) FROM {src('procedure.csv')}").fetchone()[0]
out["diagnosis_null_code_system_codes_top"] = con.execute(f"SELECT code, count(*) FROM {src('diagnosis.csv')} WHERE code_system IS NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 5").fetchall()
out["month_year_death_patterns"] = con.execute(f"SELECT regexp_replace(month_year_death,'[0-9]','9','g') p, count(*) FROM {src('patient.csv')} GROUP BY 1 ORDER BY 2 DESC").fetchall()
out["kidney_creat_14682_9_value_quantiles"] = con.execute(f"SELECT quantile_cont(try_cast(lab_result_num_val AS DOUBLE),[0.05,0.5,0.95]) FROM {src('lab_result_clean.csv')} WHERE code='14682-9'").fetchone()[0]
(ROOT/"reports"/"extra_checks.json").write_text(json.dumps(out, indent=1, default=str)); print(json.dumps(out, indent=1, default=str))
