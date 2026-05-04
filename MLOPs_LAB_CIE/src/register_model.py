import mlflow
from mlflow.tracking import MlflowClient
import json

run_id = "296c4eda54e64fce8058ab2e331628af"
model_uri = "models:/m-ea186cdcc3ca4b28a990c820fb294496"
model_name = "rackcool-cooling-power-kw-predictor"

result = mlflow.register_model(model_uri, model_name)

version = result.version
print("Model version:", version)

client = MlflowClient()

client.set_registered_model_alias(
    name=model_name,
    alias="candidate",
    version=version
)

best_rmse =  21.34067000513169

output = {
    "registered_model_name": model_name,
    "version": int(version),
    "run_id": run_id,
    "source_metric": "rmse",
    "source_metric_value": best_rmse
}

with open("results/step2_s2.json", "w") as f:
    json.dump(output, f, indent=4)