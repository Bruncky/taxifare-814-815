# THIS IS ESSENTIALLY TRAINER, BUT ADDING GRIDSEARCH

import joblib

from taxifare.data import get_data, clean_data, holdout
from taxifare.model import get_model
from taxifare.pipeline import get_pipeline
from taxifare.mlflow import MLFlowBase

from sklearn.model_selection import GridSearchCV

class ParamTrainer(MLFlowBase):

    def __init__(self) -> None:
        super().__init__(
            '[DE] [Berlin] [bruncky] taxifare  v1',
            'https://mlflow.lewagon.co/'
        )

    def train(self, params):
        # Results
        models = {}

        # Iterate over models
        for model_name, model_params in params.items():
            line_count = model_params['line_count']
            hyper_params = model_params['hyper_params']

            # Create run
            self.mlflow_create_run()

            # Log params
            self.mlflow_log_param('model_name', model_name)
            self.mlflow_log_param('line_count', line_count)

            for key, value in hyper_params.items():
                self.mlflow_log_param(key, value)

            # Get data
            data = get_data(line_count)
            data = clean_data(data)

            # Holdout
            X_train, X_test, y_train, y_test = holdout(data)

            # Log params
            self.mlflow_log_param('model', model_name)

            # Create model
            model = get_model(model_name)

            # Create pipeline
            pipeline = get_pipeline(model)

            # Create GridSearch
            grid_search = GridSearchCV(
                pipeline,
                param_grid = hyper_params,
                cv = 5
            )

            # Train with GridSearch
            grid_search.fit(X_train, y_train)

            # score GridSearch
            score = grid_search.score(X_test, y_test)

            # Save the trained model
            joblib.dump(pipeline, f'{model_name}.joblib')

            # Push metrics to mlflow
            self.mlflow_log_metric('score', score)

            # Return the GridSearch in order to identify the best estimators and params
            models[model_name] = grid_search

        return models
