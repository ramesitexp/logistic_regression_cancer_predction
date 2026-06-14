# Logistic Regression Cancer Prediction

A Streamlit app that predicts cancer versus normal using a logistic regression model loaded from pickle.

## Run locally

1. Create a Python virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model and create the pickle file:
   ```bash
   python train_model.py
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
5. Open the URL shown in the terminal.

## Deploy on Render

This repository includes a `requirements.txt` and `Procfile` so Render can detect and deploy the app.

## Features

- Trains a logistic regression model on the breast cancer dataset
- Saves the model to `model.pkl`
- Loads the pickled model in Streamlit for prediction
- Provides a clean form UI to enter tumor measurements
- Predicts cancer or non-cancer with a confidence score
