# Air Pollution Forecasting Report

## Project Goal
Forecast next-hour PM2.5 concentration using current pollutant readings, station metadata, calendar signals, and recent station history.

## Dataset Summary
- Rows used for modeling: 630904
- Stations: 25
- Time span: 2017-01-02 00:00:00 to 2019-12-31 22:00:00
- Training period: 2017-01-01 to 2018-12-31
- Test period: 2019-01-01 to 2019-12-31

## Baseline
- Persistence baseline MAE: 7.9932
- Persistence baseline RMSE: 67.1801
- Persistence baseline R2: -0.3217

## Model Family Comparison
- hist_gradient_boosting: MAE 6.8644, RMSE 49.896, R2 0.2709
- ridge_regression: MAE 9.6181, RMSE 52.1731, R2 0.2028

## Selected Final Model
- Champion model: HistGradientBoostingRegressor
- MAE: 6.8644
- RMSE: 49.896
- R2: 0.2709
- RMSE improvement vs baseline: 25.73%

## Time-Series Features Added
- Recent PM2.5 lags: 1 hour, 2 hours, 3 hours, and 24 hours
- PM2.5 rolling signals: 3-hour mean, 24-hour mean, and 24-hour volatility
- PM2.5 trend signals: current minus 1-hour lag, current minus 24-hour lag
- PM10 memory signals: 1-hour lag, 24-hour lag, and 24-hour rolling mean
- Cyclical calendar encoding for hour of day and month of year

## Quarterly Walk-Forward Backtesting
- 2019-Q1: RMSE 50.5493, baseline RMSE 68.0542, R2 0.3153
- 2019-Q2: RMSE 46.871, baseline RMSE 65.0153, R2 0.1053
- 2019-Q3: RMSE 56.0546, baseline RMSE 72.7432, R2 0.3427
- 2019-Q4: RMSE 45.0141, baseline RMSE 62.9625, R2 0.1931

Average across quarters:
- Mean MAE: 6.8445
- Mean RMSE: 49.6223
- Mean R2: 0.2391

## Monthly Residual Analysis
- Jan: bias 1.7586, MAE 8.7947, RMSE 54.6394, coverage 0.77, width 13.5168
- Feb: bias 2.0158, MAE 7.8211, RMSE 51.8528, coverage 0.7817, width 12.6468
- Mar: bias 1.0414, MAE 8.4697, RMSE 41.9438, coverage 0.7334, width 15.5969
- Apr: bias 1.156, MAE 6.0875, RMSE 45.3704, coverage 0.8402, width 12.1038
- May: bias 1.2019, MAE 6.6919, RMSE 44.6378, coverage 0.7788, width 11.4614
- Jun: bias 1.9055, MAE 6.5882, RMSE 50.1083, coverage 0.8432, width 12.3097
- Jul: bias 3.0967, MAE 7.6905, RMSE 59.3714, coverage 0.7822, width 9.3672
- Aug: bias 2.1732, MAE 6.033, RMSE 51.2475, coverage 0.8034, width 8.6549
- Sep: bias 2.5696, MAE 7.4331, RMSE 60.1458, coverage 0.7944, width 7.0247
- Oct: bias 0.8743, MAE 5.594, RMSE 46.7872, coverage 0.8138, width 8.3568
- Nov: bias 0.8317, MAE 5.8335, RMSE 46.2063, coverage 0.8909, width 12.6233
- Dec: bias 1.2758, MAE 5.8381, RMSE 42.5212, coverage 0.8168, width 10.8648

## Uncertainty Bands
- Interval level: 80%
- Empirical holdout coverage: 0.8059
- Average interval width: 11.1899
- Calibration source: 2018 quarter-by-quarter walk-forward residuals grouped by calendar month

## Most Influential Features
- PM2.5: 7.9235
- pm25_lag_1: 2.5292
- pm25_lag_2: 0.7793
- pm25_roll_std_24: 0.2819
- PM10: 0.2208
- Station code: 0.2154
- O3: 0.2031
- pm10_lag_1: 0.1616
- hour_cos: 0.1089
- NO2: 0.0860

## Interpretation
The evaluation is now stronger in three ways. First, a second model family is trained on the same feature set to test whether the gains come from the model design or only from the features. Second, quarterly walk-forward backtesting measures how stable the selected model is through time instead of relying on one aggregate holdout score. Third, monthly residual diagnostics and empirical uncertainty bands reveal when the model is biased, how error magnitude shifts through the year, and how reliable the prediction interval is.

## Artifacts
- Model: artifacts/model.joblib
- Metrics: artifacts/metrics.json
- Model comparison: artifacts/model_comparison.csv
- Quarterly backtest: artifacts/quarterly_backtest.csv
- Monthly residual analysis: artifacts/monthly_residual_analysis.csv
- Monthly uncertainty profile: artifacts/monthly_uncertainty_profile.csv
- Station metadata: artifacts/stations.csv
- Prediction plot: visuals/predicted_vs_actual.png
- Feature importance plot: visuals/feature_importance.png
- Model comparison plot: visuals/model_comparison.png
- Quarterly backtest plot: visuals/quarterly_backtest_rmse.png
- Monthly residual plot: visuals/monthly_residual_analysis.png
- Monthly uncertainty plot: visuals/monthly_uncertainty_coverage.png
- Uncertainty band example: visuals/uncertainty_band_example.png
