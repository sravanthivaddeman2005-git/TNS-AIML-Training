"""Run a single illustrative prediction using the saved model."""
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / "diabetes_model.joblib"


def main():
    if not MODEL_PATH.exists():
        raise SystemExit("Model not found. Run `python train.py` first.")
    # Example input only; replace values with one row of appropriately collected data.
    example = pd.DataFrame([{
        "Pregnancies": 2, "Glucose": 120, "BloodPressure": 70,
        "SkinThickness": 25, "Insulin": 80, "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.45, "Age": 35,
    }])
    probability = joblib.load(MODEL_PATH).predict_proba(example)[0, 1]
    print(f"Model score for the example: {probability:.1%}")
    print("This is an uncalibrated dataset-model score, not a personal medical risk or diagnosis.")


if __name__ == "__main__":
    main()
