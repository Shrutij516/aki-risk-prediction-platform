"""Lab binning must reproduce RETAIN_WithLabs.ipynb cell 80 exactly, including boundaries."""
import pytest

from aki.lab_codes import ALL_KIDNEY_CODES, severity_bin, test_type as lab_test_type


@pytest.mark.parametrize("v,expected", [
    (1.19, "creatinine_normal"), (1.2, "creatinine_mild"), (1.99, "creatinine_mild"),
    (2.0, "creatinine_moderate"), (3.99, "creatinine_moderate"), (4.0, "creatinine_severe"),
])
def test_creatinine_bins(v, expected):
    assert severity_bin("creatinine", v) == expected


@pytest.mark.parametrize("v,expected", [
    (90, "egfr_g1"), (89.9, "egfr_g2"), (60, "egfr_g2"), (59.9, "egfr_g3a"), (45, "egfr_g3a"),
    (44.9, "egfr_g3b"), (30, "egfr_g3b"), (29.9, "egfr_g4"), (15, "egfr_g4"), (14.9, "egfr_g5"),
])
def test_egfr_kdigo_bins(v, expected):
    assert severity_bin("egfr", v) == expected


@pytest.mark.parametrize("v,expected", [(19.9, "bun_normal"), (20, "bun_elevated"), (49.9, "bun_elevated"), (50, "bun_high")])
def test_bun_bins(v, expected):
    assert severity_bin("bun", v) == expected


def test_code_sets_match_notebook():
    assert len(ALL_KIDNEY_CODES) == 17  # 6 creatinine + 8 eGFR + 3 BUN in cell 80
    assert lab_test_type("2160-0") == "creatinine"
    assert lab_test_type("88293-6") == "egfr"
    assert lab_test_type("3094-0") == "bun"
    assert lab_test_type("10524-7") is None
