# Mekong Delta Mangrove Degradation — End-to-End Pipeline

This repository contains two Google Colab notebooks that together implement the full
pipeline for predicting mangrove/salt-marsh **degradation** in the Mekong Delta from
remote-sensing and geospatial data, using Google Earth Engine (GEE), Google Drive as
shared storage, and an XGBoost classifier.

- **`1_18.ipynb`** — Data engineering: from raw GEE/Zenodo exports to a leakage-free,
  scaled train/test feature matrix.
- **`Model_ML_mangrove_degradation_FULL_CONSOLIDATED.ipynb`** — Modeling: from the
  scaled matrix to a validated, calibrated, and explained XGBoost model.

All intermediate artifacts are read from and written to a single shared folder:
`/content/drive/MyDrive/Zenodo_Mekong_Data` (Google Drive, mounted in Colab).

---

## 1. `1_18.ipynb` — Data Engineering Pipeline

Each cell is a self-contained "Task" that reads specific files from `DRIVE_FOLDER` and
writes its output back to the same folder, so the notebook is designed to be run cell
by cell (sometimes across sessions) rather than top-to-bottom in one pass. Cell order
in the file does **not** match numerical task order — the dependency order is below.

| Step | Task | What it does | Key inputs | Key outputs |
|---|---|---|---|---|
| 1 | 3.0 | Merge 11 yearly NDVI CSVs (2015–2025) into one historical baseline per point, drop years with no valid scene, compute per-point mean/stddev/valid-year count, sanity-check NDVI ranking by land-cover stratum | `NDVI_Yearly_*.csv` | `NDVI_Historical_Stats_Final.csv` |
| 2 | 4.0 | QA/clean the NDVI baseline: drop physically invalid values, flag (not delete) per-stratum IQR outliers and weak baselines (<3/11 valid years) | `NDVI_Historical_Stats_Final.csv` | `NDVI_Historical_Stats_Clean.csv` |
| 3 | 8.0 | Compute the Index of Inundation Frequency (IFI) from Otsu-thresholded inundation points and flag anomalies above the P90 threshold (auto-detects wide/long input format) | `Inundation_Otsu_Points*.csv` | `IFI_Anomaly_Points.csv` |
| 4 | 1.2 | Attach a "true mangrove" reference mask by sampling the Giri et al. (2011) Global Mangrove Watch layer (`LANDSAT/MANGROVE_FORESTS/2000`) on GEE for all 12,000 sample points | `Mekong_Sample_Points_Master_SHP.shp`, GEE | `GMW_Mangrove_Reference_Mask.csv` |
| 5 | 13.0 | Harmonize all static + yearly point-level files (NDVI, Sentinel-1, inundation, IFI, land-use change, population, climate, distance rasters) into one master table via inner joins on `point_id` | `NDVI_Historical_Stats_Final.csv`, `S1_Filtered_Points.csv`, `Inundation_Otsu_Points_Annual.csv`, `IFI_Anomaly_Points_Annual.csv`, `LUCR_Points_Annual.csv`, `Population_Points_Annual.csv`, `Climate_Points_Annual.csv`, `Distance_Points_Annual.csv`, yearly NDVI files | `Final_Harmonized_Points.parquet` (12,000 rows) |
| 6 | — | Merge the GMW mask into the harmonized table as a grouping column (does not drop points or change training scope) | `Final_Harmonized_Points.parquet`, `GMW_Mangrove_Reference_Mask.csv` | `Final_Harmonized_Points.parquet` (updated in place) |
| 7 | 8.0 patch | Fix the IFI anomaly comparison operator (`>=` instead of `>`) and add `IFI_seasonal_amplitude` (std-dev of yearly inundation) as a leakage-safe hydrology feature | `Final_Harmonized_Points.parquet` | `Final_Harmonized_Points.parquet` (updated in place) |
| 8 | 15.0 | Build a 5×5 km spatial grid, assign `spatial_block_id`, check spatial autocorrelation of the label with Moran's I, then create a stratified (by block degradation-density tier) 80/20 spatial-block train/test split (seed=42) | `Final_Harmonized_Points.parquet`, `Target_Labels_Points.parquet` (provisional) | `Point_Split_Lookup.parquet`, `Mekong_Train_Dataset.parquet`, `Mekong_Test_Dataset.parquet` |
| 9 | 14.0 | Compute the final binary label `degradation_label`: within scope (Trees + Flooded Vegetation strata only), flag points where ≥2 of 3 conditions hold — 2025 NDVI anomalously low vs. each point's own 2015–2024 baseline (P25 threshold), IFI anomaly (P90), or a forest-loss flag. **All thresholds are computed from the train split only** (via `Point_Split_Lookup.parquet`) and then applied to both splits, avoiding train/test leakage. Includes a sensitivity analysis (P10/P25/P40) and a 300-point manual-verification sample | `Final_Harmonized_Points.parquet`, `Point_Split_Lookup.parquet` | `Target_Labels_Points.parquet`, `Sensitivity_Analysis_Report.csv`, `Visual_Validation_Sample_300.csv` |
| 10 | 16.0 | QA the train/test tables: dtypes, nulls, confirms leakage-prone columns (e.g. `IFI`, `NDVI_Historical_Mean`, `NDVI_2025_Anomaly_Z`) are still present so Task 17 can drop them, confirms label balance and no duplicate points | `Mekong_Train_Dataset.parquet`, `Mekong_Test_Dataset.parquet` | console report only |
| 11 | 16–17 | Reload the corrected label from Task 14.0, engineer `NDVI_Trend_2019_2024` (2024 vs. 2019, avoiding the 2025 label year), restrict to the **locked 8-feature allowlist**, impute remaining nulls with train medians, and confirm via VIF (expected 1.03–1.50) that no multicollinearity removal is needed | `Mekong_Train/Test_Dataset.parquet`, `Target_Labels_Points.parquet` | `X_train_no_multico.parquet`, `X_test_no_multico.parquet`, `y_train.parquet`, `y_test.parquet` |
| 12 | — | Export a positional `point_id`/`lon`/`lat`/`spatial_block_id`/GMW-flag index for `X_train`/`X_test`, with a row-order cross-check, since the feature matrices no longer carry `point_id` | `Mekong_Train/Test_Dataset.parquet`, `X_train/X_test_no_multico.parquet` | `Train_Point_ID_Index.parquet`, `Test_Point_ID_Index.parquet` |
| 13 | 18.0 | Fit a `StandardScaler` on `X_train` only, apply to both splits, and persist the scaler — this is the final handoff to modeling | `X_train/X_test_no_multico.parquet` | `Mekong_Train_Dataset_Final.parquet`, `Mekong_Test_Dataset_Final.parquet`, `scaler.pkl` |

**Final locked feature set (8 predictors):** `DEM_Elevation`, `loss_to_shrimp`,
`loss_to_built`, `pop_latest`, `Distance_To_Infra`, `Distance_To_Tidal_Channel`,
`IFI_seasonal_amplitude`, `NDVI_Trend_2019_2024`.

**Anti-leakage design points:**
- Label thresholds (NDVI P25 anomaly, IFI P90) are computed from the **train** spatial
  blocks only, then frozen and applied to test.
- Train/test split is **spatial-block-based** (5×5 km blocks), not random, to prevent
  nearby points from leaking across the split.
- Feature scaling is fit on train only.
- `NDVI_Trend_2019_2024` deliberately uses 2019–2024 to avoid overlapping with the 2025
  label-derivation year.

---

## 2. `Model_ML_mangrove_degradation_FULL_CONSOLIDATED.ipynb` — Modeling Pipeline

Consumes the outputs of `1_18.ipynb` and produces the final trained model, validation
metrics, and publication-quality figures. Cells are numbered sequentially with markdown
headers; later cells generally require earlier cells' outputs (loaded from Drive).

| Cell | Task | What it does | Key outputs |
|---|---|---|---|
| 1 | Setup | Mounts Drive, sets `DRIVE_FOLDER`, configures logging and a shared publication-quality matplotlib style/color palette reused by every figure | — |
| 2 | 19.1–19.2 | Computes train-set class balance and `scale_pos_weight` for XGBoost | `class_balance_stats.csv`, `scale_pos_weight_value.csv`, `Class_Distribution_Bar.png` |
| 3 | 19.3–19.4 | Assigns 5-fold `GroupKFold` spatial cross-validation folds grouped by `spatial_block_id` (no block ever spans train/validation within a fold); runs a full pairwise (10-comparison) leakage audit | `cv_fold_assignment.csv`, `fold_leakage_check.csv`, `Spatial_CV_Fold_Map.png` |
| 4 | 19.5 | SMOTE oversampling ablation study — applies SMOTE only to each fold's training partition (never validation) and compares metrics with/without it to decide whether to use SMOTE downstream | `SMOTE_Ablation_Metrics.csv`, `SMOTE_Decision_Note.csv` |
| 5 | 20.1 | Baseline models — Logistic Regression and Random Forest (both class-balanced), evaluated with the locked 5-fold spatial CV | baseline metrics/PR curve |
| 6 | 20.2 | Default-hyperparameter `XGBClassifier` baseline under the same 5-fold spatial CV, using `scale_pos_weight` | `XGB_Default_Metrics.csv`, `Baseline_Metrics_Table.csv`, `Baseline_PR_Curve.png` |
| 7 | 20.3 | Optuna hyperparameter search (100 trials) for XGBoost over the spatial CV folds | `Optuna_Trials.csv`, `Best_Hyperparameters.csv`, `Optimization_History.png` |
| 8 | 20.4 | Trains and persists the final XGBoost model on the full train set using the best Optuna hyperparameters | saved final model file |
| 9 | 20.5 | Out-of-fold (OOF) decision-threshold optimization, then evaluates the model on the held-out test set at the frozen threshold | `Threshold_OOF_vs_Default_Comparison.csv`, `Figure_OOF_Threshold_Optimization.png/.pdf` |
| — | 20.8 | Independent validation against the GMW reference mangrove mask with the retrained model | `Independent_Validation_Report_FINAL.csv` |
| 10 | 20.6–20.8 | Test-set figures: (a) PR/ROC curves, (b) SHAP summary plot and per-point SHAP values via `TreeExplainer`, (c) feature importance comparison (XGBoost gain vs. mean \|SHAP\|), (d) feature-group ablation, (e) bootstrap confidence-interval forest plot | `Figure4_PR_ROC_Curve_Test.png/.pdf`, `shap_values.csv`, `Figure5_SHAP_Summary.png/.pdf`, `Feature_Importance_Comparison.csv/.png/.pdf`, `Feature_Group_Ablation_Bar.png/.pdf`, `Bootstrap_CI_Forest_Plot.png/.pdf` |
| 11 | 20.9 | Probability calibration via Platt scaling (`CalibratedClassifierCV`, sigmoid method) using out-of-fold predictions, compared before/after | `Calibration_Fix_Metrics.csv`, `Calibration_Table_After.csv`, `Figure_Calibration_Before_After.png/.pdf` |
| 12 | — | Compares model behavior between the GMW "true mangrove" group vs. other forested wetland points, including a per-group SHAP driver ranking | `Group_Comparison_GMW_vs_Other.csv`, `SHAP_Driver_Comparison_GMW_vs_Other.csv`, `Figure_SHAP_Group_Comparison.png/.pdf` |
| 13 | — | Regenerates bootstrap CI metrics/figure for the retrained model (supersedes an earlier version) | `Bootstrap_CI_Test_Metrics.csv`, `Bootstrap_CI_Forest_Plot.png/.pdf` |
| 14 | 20.8 (post-validation) | Manual correction of `degradation_label` for two specific points (`MP_01716`, `MP_01766`), 0→1, based on visual confirmation, with re-export of the label and test-target files | `Target_Labels_Points.parquet`, `y_test.parquet` (corrected) |

**Modeling design points:**
- Cross-validation is always the same **5-fold spatial `GroupKFold`** (by
  `spatial_block_id`), reused across baseline models, Optuna tuning, threshold
  optimization, and calibration — ensuring every reported metric is spatially
  leakage-free.
- Class imbalance is handled two ways in parallel: `scale_pos_weight` (always used by
  XGBoost) and an ablation study on whether SMOTE adds value on top of it.
- Model selection order: Logistic Regression / Random Forest baselines → default
  XGBoost → Optuna-tuned XGBoost (100 trials) → final model, trained once and reused for
  all downstream evaluation, SHAP explanation, and calibration.
- Decision threshold is optimized out-of-fold, then frozen before touching the test set.
- Final interpretability is provided via SHAP (`TreeExplainer`), with a breakdown by
  mangrove-reference group (GMW true mangrove vs. other forested wetland).

---

## End-to-End Data Flow

```
GEE / Zenodo raw exports (NDVI, inundation, S1, LUCR, population, climate, distance, shapefile)
        │
        ▼
1_18.ipynb  (Tasks 3.0 → 4.0 → 8.0 → 1.2 → 13.0 → GMW merge → IFI patch
             → 15.0 spatial split → 14.0 labeling → 16.0 QA → 16-17 VIF/allowlist
             → point_id index export → 18.0 scaling)
        │
        ▼
Mekong_Train_Dataset_Final.parquet, Mekong_Test_Dataset_Final.parquet,
y_train.parquet, y_test.parquet, scaler.pkl,
Train/Test_Point_ID_Index.parquet
        │
        ▼
Model_ML_mangrove_degradation_FULL_CONSOLIDATED.ipynb
   (class balance → spatial CV folds → SMOTE ablation → baselines
    → XGBoost default → Optuna tuning → final model → OOF threshold
    → test evaluation → SHAP/importance/ablation/bootstrap CI figures
    → calibration → GMW group comparison → label correction)
        │
        ▼
Final calibrated XGBoost model + full set of manuscript-ready
metrics tables and publication figures in DRIVE_FOLDER
```

## Requirements

- Google Colab with Google Drive mounted at `/content/drive/MyDrive/Zenodo_Mekong_Data`
- Python packages: `pandas`, `numpy`, `geopandas`, `earthengine-api` (`ee`), `scikit-learn`,
  `xgboost`, `optuna`, `imbalanced-learn`, `esda`, `libpysal`, `shap`, `matplotlib`,
  `statsmodels`
- A Google Earth Engine project (used as `project='mangrove-research-505113'` in
  `1_18.ipynb`) for the GMW reference-mask sampling step

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
