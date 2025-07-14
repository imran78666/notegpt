import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler


def train_model():
    csv_path = "utils/quiz_data.csv"
    df = pd.read_csv(csv_path)

    # Select features and target
    X = df[["quiz_score", "quiz_time_sec", "num_attempts"]]
    y = df["label"].map({"pass": 1, "fail": 0})  # Convert labels to numeric

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model with balanced class weights
    model = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    )
    model.fit(X_train_scaled, y_train)

    # Predict on test set
    y_pred = model.predict(X_test_scaled)

    # Print classification report and accuracy
    print("Model evaluation on test data:")
    print(classification_report(y_test, y_pred, target_names=["fail", "pass"]))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    # Save model and scaler to utils folder
    joblib.dump(model, "utils/model.pkl")
    joblib.dump(scaler, "utils/scaler.pkl")
    print("✅ Model and scaler saved to utils/model.pkl and utils/scaler.pkl")


def predict_performance(score, time_sec, attempts):
    # Load model and scaler
    model = joblib.load("utils/model.pkl")
    scaler = joblib.load("utils/scaler.pkl")

    # Prepare input features as dataframe
    X = pd.DataFrame(
        [
            {
                "quiz_score": score,
                "quiz_time_sec": time_sec,
                "num_attempts": attempts,
            }
        ]
    )

    # Scale features
    X_scaled = scaler.transform(X)

    # Debug print probabilities
    proba = model.predict_proba(X_scaled)[0]
    print(f"Prediction probabilities (fail, pass): {proba}")

    # Predict class
    prediction = model.predict(X_scaled)[0]

    return "pass" if prediction == 1 else "fail"
