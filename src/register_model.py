import mlflow
from mlflow.tracking import MlflowClient

client   = MlflowClient()
SYMBOLS  = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
HORIZONS = ["1h", "6h", "24h"]

for symbol in SYMBOLS:
    for horizon in HORIZONS:
        experiment = client.get_experiment_by_name(f"mlops-crypto-{symbol}-{horizon}")
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="params.model = 'LinearRegression'",
            order_by=["metrics.r2 DESC"],
            max_results=1
        )

        best_run   = runs[0]
        run_id     = best_run.info.run_id
        r2         = best_run.data.metrics["r2"]
        mae        = best_run.data.metrics["mae"]

        model_name = f"crypto-predictor-{symbol}-{horizon}"
        model_uri  = f"runs:/{run_id}/model"
        result     = mlflow.register_model(model_uri, model_name)

        print(f"{symbol} {horizon} → R²: {r2:.4f} | MAE: {mae:,.2f} | registered as '{model_name}' v{result.version}")
