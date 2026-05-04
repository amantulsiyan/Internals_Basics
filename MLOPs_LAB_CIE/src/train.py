import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

import mlflow
import mlflow.sklearn
import json

mlflow.set_experiment("rackcool-cooling-power-kw")

with mlflow.start_run() as run:

    df = pd.read_csv("data/training_data.csv")

    X = df.drop("cooling_power_kw", axis=1)
    y = df["cooling_power_kw"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.2
    )

    # Linear Regression
    baselineModel = LinearRegression()
    baselineModel.fit(X_train, y_train)
    baseline_pred = baselineModel.predict(X_test)

    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
    baseline_mae = mean_absolute_error(y_test, baseline_pred)

    # Gradient Boosting
    gbr = GradientBoostingRegressor(random_state=42)
    gbr.fit(X_train, y_train)
    gbr_pred = gbr.predict(X_test)

    gbr_rmse = np.sqrt(mean_squared_error(y_test, gbr_pred))
    gbr_mae = mean_absolute_error(y_test, gbr_pred)

    # MLflow logging
    mlflow.log_param("gbr_random_state", 42)

    mlflow.log_metric("lr_rmse", baseline_rmse)
    mlflow.log_metric("lr_mae", baseline_mae)
    mlflow.log_metric("gbr_rmse", gbr_rmse)
    mlflow.log_metric("gbr_mae", gbr_mae)

    mlflow.set_tag("priority", "high")

    # Best model selection
    if gbr_rmse < baseline_rmse:
        best_model = gbr
        best_model_name = "GradientBoosting"
        best_rmse = gbr_rmse
    else:
        best_model = baselineModel
        best_model_name = "LinearRegression"
        best_rmse = baseline_rmse

    # Log model
    model_info = mlflow.sklearn.log_model(
        sk_model=best_model,
        name="model"
    )

    print("MODEL URI:", model_info.model_uri)

    run_id = run.info.run_id
    print("RUN ID:", run_id)

    print("Baseline RMSE:", baseline_rmse)
    print("GBR RMSE:", gbr_rmse)

    # JSON output (Task 1)
    output = {
        "experiment_name": "rackcool-cooling-power-kw",
        "models": [
            {
                "name": "LinearRegression",
                "mae": float(baseline_mae),
                "rmse": float(baseline_rmse)
            },
            {
                "name": "GradientBoosting",
                "mae": float(gbr_mae),
                "rmse": float(gbr_rmse)
            }
        ],
        "best_model": best_model_name,
        "best_metric_name": "rmse",
        "best_metric_value": float(best_rmse)
    }

    with open("results/step1_s1.json", "w") as f:
        json.dump(output, f, indent=4)