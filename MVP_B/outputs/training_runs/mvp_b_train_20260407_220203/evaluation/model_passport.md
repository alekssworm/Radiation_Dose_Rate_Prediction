# MVP-B model passportGenerated at UTC: 2026-04-07T22:02:20.304129+00:00## OverviewMVP-B is the current spatially evaluated and reproducibly trained working model package for `target_dose_rate` prediction.## Selection policyModels are ranked by:1. highest `cv_test_r2_mean`2. lowest `gap_r2`3. lowest `cv_test_r2_std`4. lowest `cv_test_rmse_mean`## Env reference model- Role: `env_reference`- Mode: `env_mode`- Feature set: `env_only`- Model: `random_forest`- Candidate ID: `random_forest__max_depth=8__max_features=0.5__min_samples_leaf=10__n_estimators=500__n_jobs=-1__random_state=42`- Rows used: `3376`- Number of features: `13`- Spatial CV test R² mean: `0.104622`- Spatial CV test R² std: `0.024971`- Spatial CV train R² mean: `0.304534`- Gap R²: `0.199911`- Spatial CV test RMSE mean: `0.012452`- Spatial CV test MAE mean: `0.009210`Parameters:```json{
  "max_depth": 8,
  "max_features": 0.5,
  "min_samples_leaf": 10,
  "n_estimators": 500,
  "n_jobs": -1,
  "random_state": 42
}```## Primary model- Role: `primary_candidate`- Mode: `nuclide_mode`- Feature set: `env_plus_no_ratio`- Model: `extra_trees`- Candidate ID: `extra_trees__max_depth=6__max_features=1.0__min_samples_leaf=2__n_estimators=500__n_jobs=-1__random_state=42`- Rows used: `545`- Number of features: `18`- Spatial CV test R² mean: `0.269798`- Spatial CV test R² std: `0.056410`- Spatial CV train R² mean: `0.590620`- Gap R²: `0.320822`- Spatial CV test RMSE mean: `0.011657`- Spatial CV test MAE mean: `0.008884`Parameters:```json{
  "max_depth": 6,
  "max_features": 1.0,
  "min_samples_leaf": 2,
  "n_estimators": 500,
  "n_jobs": -1,
  "random_state": 42
}```## Known limitations- Spatial robustness is prioritized over ordinary random-split score.- Contamination-related features remain the strongest predictive source.- Env-only performance is expected to be weaker and should be treated as contextual.- This package is intended for tabular prediction and controlled scenario comparison.