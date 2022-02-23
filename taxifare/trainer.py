import joblib

from taxifare.data import get_data, clean_data, holdout, save_model_to_gcp
from taxifare.model import get_model
from taxifare.pipeline import get_pipeline
from taxifare.metrics import compute_rmse
from taxifare.mlflow import MLFlowBase

class Trainer(MLFlowBase):
    def __init__(self) -> None:
        super().__init__(
            "[DE] [Berlin] [bruncky] taxifare  v1",
            "https://mlflow.lewagon.co/"
        )

    def train(self):
        # Defining line_count for get_data
        # and model_name for get_model
        line_count = 1000
        model_name = 'random_forest'

        # Create run
        self.mlflow_create_run()

        # Log model
        self.mlflow_log_param('model', 'random forest regressor')

        # Get data
        data = get_data(line_count = line_count)

        # Clean data
        data = clean_data(data)

        # Holdout
        X_train, X_test, y_train, y_test = holdout(data)

        # Create Model
        model = get_model(model_name)

        # Create Pipeline
        pipeline = get_pipeline(model)

        # Train pipeline
        pipeline.fit(X_train, y_train)

        # Compute y_pred
        y_pred = pipeline.predict(X_test)

        # Compute RMSE for y_pred
        rmse = compute_rmse(y_pred, y_test)

        # Log RMSE
        self.mlflow_log_metric('rmse', rmse)

        # Save pipeline
        joblib.dump(pipeline, f'{model_name}.joblib')

        # Save it to GCP
        save_model_to_gcp('random_forest_regressor', f'{model_name}.joblib')

        return pipeline

def main():
    trainer = Trainer()
    trainer.train()

if __name__ == '__main__':
    main()
