# utils/ml_model.py

import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


# Use absolute path to be safe
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")


def train_model():
    csv_path = os.path.join(os.path.dirname(__file__), "quiz_data.csv")
    df = pd.read_csv(csv_path)

    # Features and target
    X = df[["quiz_score", "quiz_time_sec", "num_attempts"]]
    y = df["label"].map({"pass": 1, "fail": 0})

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model with balanced classes
    model = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("Model evaluation on test data:")
    print(classification_report(y_test, y_pred, target_names=["fail", "pass"]))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model trained and saved to {MODEL_PATH}")


def predict_performance(score, time_sec, attempts):
    # Load the trained model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please train the model first.")

    model = joblib.load(MODEL_PATH)

    # Create dataframe for input
    X = pd.DataFrame(
        [{"quiz_score": score, "quiz_time_sec": time_sec, "num_attempts": attempts}]
    )

    # Predict probability and class
    proba = model.predict_proba(X)[0]
    print(f"Prediction probabilities (fail, pass): {proba}")

    prediction = model.predict(X)[0]

    return "pass" if prediction == 1 else "fail"
