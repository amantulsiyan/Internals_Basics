import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import mlflow
import mlflow.sklearn
mlflow.setexperiments("rackcooling-power-kw")
with mlflow.start_run() as run:
    df = pd.read_csv("data\training_data.csv")
    X=df.drop("cooling_power_kw",axis=1)
    y=df["cooling_power_kw"]
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size = 0.2)
    baselineModel= LinearRegression()
    baselineModel.fit(X_train, y_train)
    baseline_pred = baselineModel.predict(X_test)
    baseline_rmse = root_mean_squared_error(y_test, baseline_pred)
    baseline_mae =  mean_absolute_error(y_test, baseline_pred)
    gbr = GradientBoostingRegressor(random_state=42)
    gbr.fit(X_train, y_train)
    gbr_pred = gbr.predict(X_test)
    gbr_rmse = root_mean_squared_error(y_test, gbr_pred)
    gbr_mae = mean_absolute_error(y_test, gbr_pred)
    mlflow.log_param("gbr_random_state",42)
    mlflow.log_metric("Baseline RMSE", baseline_rmse)
    mlflow.log_metric("Baseline MAE", baseline_mae)
    mlflow.log_metric("GBR RMSE", gbr_rmse)
    mlflow.log_metric("GBR MAE", gbr_mae)
    mlflow.set_tag("priority", "high")
    
    if gbr_rmse<baseline_rmse:
        best_model = gbr
        best_model_name = "GradientBoostingRegressor"
        best_rmse = gbr_rmse
    else:
        best_model = baselineModel
        best_model_name = "Baseline Linear Regression"
        best_rmse = baseline_rmse
    
    mlflow.sklearn.log_model(best_model, "model")
    run_id = run.info.run_id
    print("RUN ID: ", run_id)


