"""Train an educational diabetes classifier from the local CSV dataset."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset" / "diabetes.csv"
MODEL_PATH = ROOT / "diabetes_model.joblib"
COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Keep diabetes.csv inside the dataset folder."
        )
    data = pd.read_csv(DATA_PATH, header=None, names=COLUMNS)
    X = data.drop(columns="Outcome")
    y = data["Outcome"].astype(int)
    X[ZERO_AS_MISSING] = X[ZERO_AS_MISSING].mask(
        X[ZERO_AS_MISSING].eq(0), np.nan
    ).astype(float)
    return X, y


def main():
    X, y = load_data()
    numeric = X.columns.tolist()
    prep = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
    ])
    model = Pipeline([
        ("prep", prep),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    print(f"Held-out ROC-AUC: {roc_auc_score(y_test, probabilities):.3f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions, target_names=["class 0", "class 1"]))
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
