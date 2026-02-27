import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ===============================
# 1. Load dataset
# ===============================
data = pd.read_csv("data/city_day.csv")

print("Original shape:", data.shape)

# ===============================
# 2. Select required columns
# ===============================
data = data[["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "AQI"]]

# ===============================
# 3. Handle missing values
# ===============================
data = data.dropna()

print("After cleaning:", data.shape)

# ===============================
# 4. Seaborn Heatmap (IMPORTANT)
# ===============================
plt.figure(figsize=(8,6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# ===============================
# 5. Features and Target
# ===============================
X = data[["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]]
y = data["AQI"]

# ===============================
# 6. Train-test split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 7. Model + GridSearchCV
# ===============================
rf = RandomForestRegressor(random_state=42)

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [None, 10]
}

grid = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("Best Parameters:", grid.best_params_)

# ===============================
# 8. Evaluation
# ===============================
y_pred = best_model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# ===============================
# 9. Feature Importance (BONUS MARKS)
# ===============================
importances = best_model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(6,4))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# ===============================
# 10. Save model
# ===============================
joblib.dump(best_model, "model/aqi_model.pkl")

print("✅ Model trained and saved!")