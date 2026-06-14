# Logistic Regression Cancer Prediction

A Flask-based demo app that predicts breast cancer risk using a trained logistic regression model.

## Run locally

1. Create a Python virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Start the app: `python app.py`
4. Open `http://localhost:5000`

## Deploy on Render

This repository includes a `requirements.txt` and `Procfile` so Render can detect and deploy the app automatically.

## Features

- Uses scikit-learn's breast cancer dataset
- Trains a logistic regression model on 6 selected features
- Displays a clean form UI to submit measurements
- Predicts cancer vs. non-cancer with confidence score
