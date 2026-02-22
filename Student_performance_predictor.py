import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ----------------------------
# 1. Load Dataset
# ----------------------------
data = pd.read_csv("e:\Healthproject\student_data.csv")   # use your file name

print("Dataset Loaded Successfully ✅")

# ----------------------------
# 2. Create Total Marks
# ----------------------------
data["Total_Marks"] = data["G1"] + data["G2"] + data["G3"]

# ----------------------------
# 3. Select Features (Numeric Only)
# ----------------------------
X = data[["G1", "G2", "studytime", "absences", "failures"]]
y = data["Total_Marks"]

# ----------------------------
# 4. Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 5. Train Model
# ----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------
# 6. Predictions
# ----------------------------
y_pred = model.predict(X_test)

# ----------------------------
# 7. Model Evaluation
# ----------------------------
print("\nModel Performance 📊")
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# ----------------------------
# 8. Compare Actual vs Predicted
# ----------------------------
results = pd.DataFrame({
    "Actual_Total": y_test.values,
    "Predicted_Total": y_pred
})

print("\nSample Predictions:")
print(results.head())

import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("Actual vs Predicted")
plt.show()