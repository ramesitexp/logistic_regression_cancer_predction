import pickle
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean concavity",
]

MODEL_PATH = Path(__file__).parent / "model.pkl"

if __name__ == "__main__":
    cancer = load_breast_cancer()
    X = cancer.data[:, [list(cancer.feature_names).index(name) for name in FEATURE_NAMES]]
    y = cancer.target
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000, solver="liblinear"))
    model.fit(X, y)

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    print(f"Model trained and saved to {MODEL_PATH}")
