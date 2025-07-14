import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Sample dummy data
X_train = np.array([
    [50, 150, 1],
    [70, 250, 2],
    [30, 100, 4],
    [90, 300, 1],
    [60, 200, 3]
])
y_train = ['fail', 'pass', 'fail', 'pass', 'fail']

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model.pkl')

print("✅ Model retrained and saved as model.pkl")
