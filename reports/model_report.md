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
The evaluation is now stronger in two ways. First, a second model family is trained on the same feature set to test whether the gains come from the model design or only from the features. Second, quarterly walk-forward backtesting measures how stable the selected model is through time instead of relying on one aggregate holdout score.

## Artifacts
- Model: artifacts/model.joblib
- Metrics: artifacts/metrics.json
- Model comparison: artifacts/model_comparison.csv
- Quarterly backtest: artifacts/quarterly_backtest.csv
- Station metadata: artifacts/stations.csv
- Prediction plot: visuals/predicted_vs_actual.png
- Feature importance plot: visuals/feature_importance.png
- Model comparison plot: visuals/model_comparison.png
- Quarterly backtest plot: visuals/quarterly_backtest_rmse.png
