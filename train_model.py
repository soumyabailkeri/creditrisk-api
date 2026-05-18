
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Create sample loan data
data = {
    'age': [25, 45, 35, 28, 52, 23, 40, 31, 48, 36,
            27, 43, 33, 29, 51, 24, 41, 32, 47, 37],
    'income': [30000, 80000, 55000, 35000, 90000, 25000, 70000,
               45000, 85000, 60000, 32000, 75000, 52000, 38000,
               95000, 28000, 68000, 48000, 82000, 63000],
    'loan_amount': [10000, 20000, 15000, 12000, 25000, 8000, 18000,
                    13000, 22000, 16000, 11000, 19000, 14000, 12500,
                    24000, 9000, 17000, 13500, 21000, 16500],
    'employment_years': [2, 10, 6, 3, 15, 1, 8, 5, 12, 7,
                         2, 9, 6, 3, 14, 1, 8, 4, 11, 7],
    'defaulted': [1, 0, 0, 1, 0, 1, 0, 0, 0, 0,
                  1, 0, 0, 1, 0, 1, 0, 0, 0, 0]
}

df = pd.DataFrame(data)

# Features and target
X = df[['age', 'income', 'loan_amount', 'employment_years']]
y = df['defaulted']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Check accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy: {accuracy * 100:.1f}%")

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")