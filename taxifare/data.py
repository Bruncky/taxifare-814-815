import pandas as pd
import datetime

from google.cloud import storage

from sklearn.model_selection import train_test_split

def get_data(line_count, method = 's3'):
    url = ''

    if method == 's3':
        url = 's3://wagon-public-datasets/taxi-fare-train.csv'
    elif method == 'gs':
        url = 'gs://le-wagon-data/data/train_1k.csv'
    elif method == 'blob':
        return get_data_using_blob(line_count)

    data = pd.read_csv(url, nrows = line_count)

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

def save_model_to_gcp(cloud_model_name, local_model_path):
    BUCKET_NAME = 'batch-814-815-berlin-bucket'

    # Defining some variables for the cloud name
    today = datetime.datetime.today()
    date = f'{today.day}_{today.month}_{today.year}'
    time = f'{today.hour}_{today.minute}_{today.second}'

    # Name of the file ON THE CLOUD
    storage_location = f'models/{cloud_model_name}_{date}_{time}.joblib'

    # Name of LOCAL file to be uploaded
    local_model_filename = local_model_path

    # IMPORTANT: this verifies $GOOGLE_APPLICATION_CREDENTIALS
    client = storage.Client()

    # Load the bucket
    bucket = client.bucket(BUCKET_NAME)

    # Create blob to upload
    blob = bucket.blob(storage_location)
    blob.upload_from_filename(local_model_filename)

# -------------------- HELPER METHODS -------------------- #
def get_data_using_blob(line_count):
    # Get data from MY OWN Google Storage bucket
    BUCKET_NAME = 'batch-814-815-berlin-bucket'
    BUCKET_TRAIN_DATA_PATH = 'data/train_1k.csv'

    data_file = 'train_1k.csv'

    # IMPORTANT: this verifies $GOOGLE_APPLICATION_CREDENTIALS
    client = storage.Client()

    # Load the bucket
    bucket = client.bucket(BUCKET_NAME)

    # Create a blob to download
    blob = bucket.blob(BUCKET_TRAIN_DATA_PATH)
    blob.download_to_filename(data_file)

    # Load downloaded data into a DataFrame
    data = pd.read_csv(data_file, nrows = line_count)

    return data

def main():
    cloud_model_name = 'random_forest_regressor'
    local_model_path = '/Users/bruncky/lecture_packages/taxifare/taxifare/models/random_forest.joblib'

    save_model_to_gcp(cloud_model_name, local_model_path)

if __name__ == '__main__':
    main()
