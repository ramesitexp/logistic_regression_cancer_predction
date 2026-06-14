import pickle
from pathlib import Path

import numpy as np
import streamlit as st
from sklearn.datasets import load_breast_cancer

st.set_page_config(page_title="Breast Cancer Predictor", page_icon="🩺", layout="centered")

MODEL_PATH = Path(__file__).parent / "model.pkl"
FEATURE_LABELS = [
    "Average radius",
    "Texture variation",
    "Perimeter length",
    "Tumor area",
    "Border smoothness",
    "Concavity depth",
]
FEATURE_NAMES = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean concavity",
]
FEATURE_HELP = {
    "Average radius": "The average size of the tumor measured from center to edge.",
    "Texture variation": "How irregular the tumor's interior texture appears.",
    "Perimeter length": "The average outline length of the tumor shape.",
    "Tumor area": "The average cross-sectional area of the tumor.",
    "Border smoothness": "How smooth or jagged the tumor border is.",
    "Concavity depth": "The average depth of any inward curves on the tumor border.",
}

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)

@st.cache_data
def get_defaults():
    cancer = load_breast_cancer()
    X = cancer.data[:, [list(cancer.feature_names).index(name) for name in FEATURE_NAMES]]
    return {label: float(np.round(X[:, idx].mean(), 4)) for idx, label in enumerate(FEATURE_LABELS)}

model = load_model()

def predict(values):
    values_array = np.array([values[label] for label in FEATURE_LABELS]).reshape(1, -1)
    prediction_class = model.predict(values_array)[0]
    probability = float(model.predict_proba(values_array)[0].max())
    return prediction_class, probability

st.title("Cancer Prediction")
st.markdown(
    "Enter six tumor measurements below. Each field uses a friendly label and a quick explanation so the attributes are easy to understand. "
    "The model is loaded from `model.pkl` and predicts whether the sample is cancer or normal."
)

defaults = get_defaults()
cols = st.columns(2)
user_values = {}
for idx, label in enumerate(FEATURE_LABELS):
    column = cols[idx % 2]
    user_values[label] = column.number_input(
        label=label,
        value=defaults[label],
        format="%.4f",
        step=0.1,
        help=FEATURE_HELP.get(label, ""),
    )

if st.button("Predict"):
    prediction_class, probability = predict(user_values)
    label = "Cancer" if prediction_class == 0 else "Normal"
    st.success(f"Prediction: {label}")
    st.write(f"Confidence: {probability * 100:.1f}%")

    if prediction_class == 0:
        st.warning(
            "The model predicts a cancer diagnosis. This is a data-driven prediction and not a medical diagnosis."
        )
    else:
        st.info("The model predicts a normal diagnosis.")
