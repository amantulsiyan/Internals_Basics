import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

MODEL_NAME = "rackcool-cooling-power-kw-predictor"

df = pd.read_csv("data/training_data.csv")

X = df.drop("cooling_power_kw", axis=1)
y = df["cooling_power_kw"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


champion_model = LinearRegression()
champion_model.fit(X_train, y_train)

champion_pred = champion_model.predict(X_test)
champion_rmse = np.sqrt(mean_squared_error(y_test, champion_pred))

challenger_model = GradientBoostingRegressor(random_state=99)
challenger_model.fit(X_train, y_train)

challenger_pred = challenger_model.predict(X_test)
challenger_rmse = np.sqrt(mean_squared_error(y_test, challenger_pred))

with mlflow.start_run() as run:
    mlflow.sklearn.log_model(
        sk_model=challenger_model,
        artifact_path="model"
    )
    run_id = run.info.run_id

result = mlflow.register_model(
    f"runs:/{run_id}/model",
    MODEL_NAME
)

challenger_version = int(result.version)

client = MlflowClient()

versions = client.get_latest_versions(MODEL_NAME)

champion_version = int(versions[0].version)

client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="live",
    version=champion_version
)

if challenger_rmse < champion_rmse:
    # promote challenger
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="live",
        version=challenger_version
    )
    action = "promoted"
    final_live = challenger_version
else:
    action = "kept"
    final_live = champion_version

output = {
    "registered_model_name": MODEL_NAME,
    "alias_name": "live",
    "champion_version": champion_version,
    "challenger_version": challenger_version,
    "action": action
}

with open("results/step3_s3.json", "w") as f:
    json.dump(output, f, indent=4)

print("Champion RMSE:", champion_rmse)
print("Challenger RMSE:", challenger_rmse)
print("Champion Version:", champion_version)
print("Challenger Version:", challenger_version)
print("Action:", action)