from flask import Flask, render_template, request
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "concavity_mean",
]

app = Flask(__name__)

# Train a simple logistic regression model on the breast cancer dataset
cancer = load_breast_cancer()
X = cancer.data[:, [list(cancer.feature_names).index(name) for name in FEATURE_NAMES]]
y = cancer.target
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000, solver="liblinear"))
model.fit(X, y)

feature_defaults = {
    name: float(X[:, idx].mean()) for idx, name in enumerate(FEATURE_NAMES)
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    values = feature_defaults.copy()

    if request.method == "POST":
        try:
            values = {
                name: float(request.form.get(name, feature_defaults[name]))
                for name in FEATURE_NAMES
            }
            input_vector = [[values[name] for name in FEATURE_NAMES]]
            prediction_class = model.predict(input_vector)[0]
            probability = float(model.predict_proba(input_vector)[0].max())
            prediction = "Cancer" if prediction_class == 0 else "Non-cancer"
        except ValueError:
            prediction = "Invalid input. Please enter numeric values for all fields."

    return render_template(
        "index.html",
        feature_names=FEATURE_NAMES,
        values=values,
        prediction=prediction,
        probability=probability,
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(__import__("os").environ.get("PORT", 5000)))
