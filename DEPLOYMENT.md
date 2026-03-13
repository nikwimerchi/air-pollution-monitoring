# Deployment Guide

## Streamlit Community Cloud

1. Create a GitHub repository and push the full project.
2. Include these folders if you want the deployed app to open with the trained model already available:
   - `artifacts/`
   - `visuals/`
3. Sign in to Streamlit Community Cloud.
4. Click **Create app**.
5. Choose the repository, branch, and main file `app.py`.
6. Deploy.

## Local Deployment

```bash
pip install -r requirements.txt
python scripts/train_model.py
streamlit run app.py
```

## What the App Shows

- Forecast tab: live next-hour PM2.5 prediction
- Validation tab: model-family comparison and quarterly walk-forward backtesting
- Visuals tab: report-ready plots
- Deployment tab: quick setup instructions