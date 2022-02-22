import pandas as pd

from sklearn.model_selection import train_test_split

def get_data():
    url = 's3://wagon-public-datasets/taxi-fare-train.csv'
    data = pd.read_csv(url, nrows = 100)

    return data

def clean_data(data):
    # Removing any rows containing NaN
    data = data.dropna(axis = 'rows')

    # Keeping lat/lon that are not zero
    condition = (data.dropoff_latitude != 0) | (data.dropoff_longitude != 0)
    data = data[condition]

    condition = (data.pickup_latitude != 0) | (data.pickup_longitude != 0)
    data = data[condition]

    # If fare_amount is in the columns, keep only the ones
    # between 0 and 4000
    if 'fare_amount' in list(data):
        condition = data.fare_amount.between(0, 4000)
        data = data[condition]
        
    # Keep only rows where passenger_count is strictly below 8
    # and above 1
    condition = data.passenger_count < 8
    data = data[condition]

    condition = data.passenger_count >= 1
    data = data[condition]

    # Limiting the coordinates
    condition = data['pickup_latitude'].between(left = 40, right = 42)
    data = data[condition]

    condition = data['pickup_longitude'].between(left = -74.3, right = -72.9)
    data = data[condition]

    condition = data['dropoff_latitude'].between(left = 40, right = 42)
    data = data[condition]

    condition = data['dropoff_longitude'].between(left = -74, right = -72.9)
    data = data[condition]

    return data

def holdout(data):
    X = data.drop('fare_amount', axis = 1)
    y = data['fare_amount']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .1)

    return X_train, X_test, y_train, y_test