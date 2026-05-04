import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# LOAD DATA
train_df = pd.read_csv("data/training_data.csv")
new_df = pd.read_csv("data/new_data.csv")

# counts
original_rows = len(train_df)
new_rows = len(new_df)

# combine
combined_df = pd.concat([train_df, new_df], ignore_index=True)
combined_rows = len(combined_df)

# SPLIT DATA
X = combined_df.drop("cooling_power_kw", axis=1)
y = combined_df["cooling_power_kw"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# CHAMPION MODEL (from Task 1 → Linear Regression)
champion_model = LinearRegression()
champion_model.fit(X_train, y_train)

champion_pred = champion_model.predict(X_test)
champion_mae = mean_absolute_error(y_test, champion_pred)


# RETRAINED MODEL (same type)
retrained_model = LinearRegression()
retrained_model.fit(X_train, y_train)

retrained_pred = retrained_model.predict(X_test)
retrained_mae = mean_absolute_error(y_test, retrained_pred)

improvement = champion_mae - retrained_mae
threshold = 0.5

if improvement >= threshold:
    action = "promoted"
else:
    action = "kept_champion"

output = {
    "original_data_rows": original_rows,
    "new_data_rows": new_rows,
    "combined_data_rows": combined_rows,
    "champion_mae": champion_mae,
    "retrained_mae": retrained_mae,
    "improvement": improvement,
    "min_improvement_threshold": threshold,
    "action": action,
    "comparison_metric": "mae"
}

with open("results/step4_s4.json", "w") as f:
    json.dump(output, f, indent=4)

print("Champion MAE:", champion_mae)
print("Retrained MAE:", retrained_mae)
print("Improvement:", improvement)
print("Action:", action)