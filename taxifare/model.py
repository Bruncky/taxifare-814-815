from sklearn.ensemble import RandomForestRegressor

def get_model():
    model_params = {
        'n_estimators': 100,
        'max_depth': 1
    }

    model = RandomForestRegressor()
    model.set_params(**model_params)

    return model