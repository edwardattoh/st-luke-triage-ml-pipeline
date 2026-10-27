# End-to-End Healthcare Triage ML Pipeline & Cloud Interoperability Prototype

## Project Overview
Production-minded machine learning pipeline and cloud architecture proposal for **Emergency Department (ED) Clinical Triage** at St. Luke Hospital (`High_Acuity_Triage` ESI Level 1/2 escalation).

### Key Pipeline Highlights
* **Data Cleansing & Quality Gates:** Remediates 55,500 raw records down to 49,904 validated encounters (removing 534 exact duplicates, 4,966 age-jittered near-duplicates, 96 negative billing entries, and 6 post-triage leakage columns).
* **Model Performance:** Champion `HistGradientBoostingClassifier` achieves **0.8270 ROC-AUC** (`95% CI: [0.8112, 0.8411]`), **0.6067 PR-AUC**, **0.1114 Brier Score**, and **3.79x Lift@10%**.
* **Clinical Decision Impact:** Operating at a cost-optimized threshold (`tau = 0.20`) captures **68.80% sensitivity** while reducing expected clinical deterioration/escalation costs by **$225,076 per 1,000 patients**.
* **Governance & Fairness:** Verified stable feature distributions (`PSI < 0.005`) and Equal Opportunity TPR parity across `Gender` (`< 1.5%` gap) and `Insurance Provider`.

## Repository Structure
* `StLuke_Triage_ML_Pipeline.ipynb` — End-to-end Google Colab implementation notebook.
* `healthcare_dataset.csv` — Raw St. Luke Hospital EHR admission dataset (55,500 rows).
* `triage_model_evaluation_curves.png` — Holdout ROC, Precision-Recall, and Calibration plots.
* `st_luke_triage_histgbm_v1.joblib` — Serialized champion Scikit-Learn Pipeline artifact.
* `model_metadata_v1.json` — Model contract, operational threshold (`tau = 0.20`), and training imputation medians.
* `requirements.txt` — Pinned Python package dependencies for exact reproducibility.
