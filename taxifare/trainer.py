import joblib

from taxifare.data import get_data, clean_data, holdout
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
        line_count = 1000

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
        model = get_model('random_forest')

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

        # GridSearch
        # grid_search = ParamTrainer().get_grid_search(pipeline, X_train, y_train)

        # Save pipeline
        joblib.dump(pipeline, 'model.joblib')

        return pipeline
