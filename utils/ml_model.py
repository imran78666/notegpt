# utils/ml_model.py

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_model():
    csv_path = "quiz_data.csv"
    df = pd.read_csv(csv_path)

    # Select features and target
    X = df[["quiz_score", "quiz_time_sec", "num_attempts"]]
    y = df["label"].map({"pass": 1, "fail": 0})  # Convert labels to numeric

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "utils/model.pkl")

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Model trained. Accuracy: {acc:.2f}")


def predict_performance(score, time_sec, attempts):
    # Load the trained model
    model = joblib.load("utils/model.pkl")

    # Prepare input features
    X = pd.DataFrame([{
        "quiz_score": score,
        "quiz_time_sec": time_sec,
        "num_attempts": attempts
    }])

    prediction = model.predict(X)[0]
    return "pass" if prediction == 1 else "fail"
