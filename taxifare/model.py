from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

def get_model(model_name):
    if model_name == 'random_forest':
        model_params = {
            'n_estimators': 100,
            'max_depth': 1
        }

        model = RandomForestRegressor()
        model.set_params(**model_params)

        return model

    elif model_name == 'linear_regression':
        model = LinearRegression()

        return model
