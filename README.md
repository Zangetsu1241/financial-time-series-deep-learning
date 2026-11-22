# Financial Time-Series Forecasting: Deep Learning vs Statistical Baselines

A reproducible quantitative research benchmark comparing **Zero forecast, Ridge, ARIMA(1,0,0), LSTM, TCN, and Transformer** models on next-day financial log returns.

## Research question
Do deep sequence models consistently improve out-of-sample forecasts over simpler statistical baselines? No result is assumed.

## Setup
Python 3.10+:
```bash
python -m pip install -e '.[test]'
python -m pytest -q
finforecast --days 900 --epochs 8 --output results
```
Fast smoke experiment:
```bash
finforecast --days 260 --window 12 --epochs 1 --hidden 8 --output quick-results
```

With your own daily adjusted closes, use a CSV with `date,close` columns, positive strictly ordered prices (at least 150 rows):
```bash
finforecast --csv adjusted_prices.csv --epochs 12 --output market-results
```

## Dataset and information boundaries
- Default data is **synthetic**, autoregressive returns with a switching volatility regime. Synthetic results are not evidence of performance on real securities.
- Features at decision time t: log return, 10-day rolling return volatility, 5-day momentum, calculated using information available at or before t.
- A trailing window of features forecasts next-day log return.
- 65% training, 15% validation, 20% test in chronological order.
- Scaler fits only training-era observations. Ridge fits training rows only. ARIMA is estimated on training observations and updated with realized returns sequentially **after predicting them**; parameters are not retrained on validation or test.
- Neural models select the best checkpoint by validation MSE; test observations are never used for fitting/checkpoint selection.
- Reports test MSE, MAE, directional accuracy, model parameters, and illustrative after-cost sign-strategy returns.

## Outputs
- `metrics.json`: per-model validation/test metrics and parameter counts.
- `predictions.csv`: dated validation/test actuals and forecasts.
- `test_mse.png`: held-out mean squared error comparison.

## Limitations
Close-t features targeting close-to-close t+1 returns do **not** imply executable fill at close t. Transaction costs are simple proportional turnover penalties, not market-impact or real execution estimates. Corporate actions must be adjusted in input data. This is an experimental platform, not an investment recommendation. To validate a research hypothesis, use point-in-time market data, multiple securities and eras, and walk-forward repeated tests.

A synthetic 400-day example with 3 epochs ranked ARIMA first by test MSE; the experiment does **not** show deep learning dominance.

MIT licensed. CI validates Python 3.10/3.12.
