"""Original published results (Keras), copied from the original project README (kept locally in AKI_Dialysis_Prediction-main/).

Test-set numbers whose hyperparameters and threshold were tuned on that same test set.
Source notes explain where README and saved notebook outputs disagree (see PROVENANCE.md).
"""
ORIGINAL = {
    ("transformer", "nolabs"): dict(roc_auc=0.631, pr_auc=0.135, f1=0.205, precision=0.148, sensitivity=0.336,
                                    specificity=0.817, brier=0.094, threshold=None,
                                    note="README/report screenshot; saved notebook output differs (ROC-AUC 0.640, F1 0.009, threshold 0.34)"),
    ("transformer", "withlabs"): dict(roc_auc=0.625, pr_auc=0.157, f1=0.204, precision=0.127, sensitivity=0.513,
                                      specificity=0.662, brier=0.090, threshold=0.22,
                                      note="original used naive LOINC lab tokens, not binned labs: different input"),
    ("retain", "nolabs"): dict(roc_auc=0.638, pr_auc=0.151, f1=0.214, precision=0.134, sensitivity=0.529,
                               specificity=0.677, brier=0.113, threshold=0.29, note="matches saved notebook output"),
    ("retain", "withlabs"): dict(roc_auc=0.653, pr_auc=0.194, f1=0.254, precision=0.264, sensitivity=0.244,
                                 specificity=0.935, brier=0.118, threshold=0.36, note="binned labs; matches saved notebook output"),
}
