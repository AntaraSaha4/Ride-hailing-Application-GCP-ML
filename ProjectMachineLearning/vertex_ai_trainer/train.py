from google.cloud import storage
import pandas as pd
import math
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from hypertune import HyperTune
import argparse
import os
import matplotlib.path as mpltPath

# ==========================
# ==== Define Variables ====
# ==========================
# When dealing with a large dataset, it is practical to randomly sample
# a smaller proportion of the data to reduce the time and money cost per iteration.
#
# When you are testing, start with 0.2. You need to change it to 1.0 when you make submissions.
# TODO: Set SAMPLE_PROB to 1.0 when you make submissions
SAMPLE_PROB = 0.2   # Sample 20% of the whole dataset
random.seed(15619)  # Set the random seed to get deterministic sampling results

# TODO: Update the value using the ID of the GS bucket
# For example, if the GS path of the bucket is gs://my-bucket the OUTPUT_BUCKET_ID will be "my-bucket"
OUTPUT_BUCKET_ID = 'ml-fare-prediction-120818'

# DO NOT CHANGE IT
DATA_BUCKET_ID = 'cmucc-public'
# DO NOT CHANGE IT
TRAIN_FILE = 'dataset/nyc-taxi-fare/cc_nyc_fare_train_small.csv'


# =========================
# ==== Utility Methods ====
# =========================
def haversine_distance(origin, destination):
    """
    Calculate the spherical distance from coordinates

    :param origin: tuple (lat, lng)
    :param destination: tuple (lat, lng)
    :return: Distance in km
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) * math.cos(
        math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d


# =====================================
# ==== Define data transformations ====
# =====================================

def process_train_data(raw_df):
    """
    TODO: Copy your feature engineering code from task 1 here

    :param raw_df: the DataFrame of the raw training data
    :return:  a DataFrame with the predictors created
    """
    # Create Boundary Polygon for NYC
    boundary = [[-74.245423302,40.4982316953],[-74.199313916,40.6451659209],[-74.1467405959,40.6436015954],
             [-74.1158151042,40.6467302396],[-74.0725194473,40.652987118],[-74.0694268667,40.6310854621],
             [-74.0601492034,40.6130893818],[-74.0508715402,40.6083940404],[-74.0432000791,40.6397715688],
             [-74.0240694632,40.652987118],[-74.0096375513,40.6631533277],[-74.0323162924,40.6842627394],
             [-74.0261312098,40.6983320071],[-74.0271620438,40.6959873155],[-74.0137609659,40.7522365679],
             [-73.9086143257,40.923814614],[-73.749863374,40.9222567933],[-73.7867377248,40.8724687301],
             [-73.7809315771,40.8566613106],[-73.7762866412,40.8294286083],[-73.7937050843,40.8443640202],
             [-73.8192520986,40.8513913323],[-73.8064785914,40.8215202268],[-73.8180908867,40.7986685462],
             [-73.7786090649,40.7933939407],[-73.756545686,40.7652557343],[-73.6950005739,40.7265462696],
             [-73.742610967,40.6393675514],[-73.7646743459,40.6314365811],[-73.7693192818,40.6173346961],
             [-73.7368048194,40.6323177681],[-73.7228701004,40.6173346961],[-73.7495783265,40.5952945721],
             [-73.9435037118,40.5414861827],[-73.9295689043,40.5776572399],[-73.9795018097,40.5697189432],
             [-74.0154999075,40.5732471801],[-74.005048824,40.5917674979],[-74.0352407741,40.6085195187],
             [-74.0631103007,40.5926492082],[-74.2431008784,40.4991147051],[-74.245423302,40.4982316953],
             [-74.245423302,40.4982316953]]

    df_boundary = pd.DataFrame(boundary, columns=['lon','lat'])
    path = mpltPath.Path(boundary)
    inside = path.contains_points(raw_df[['pickup_longitude','pickup_latitude']]) # Check the points which are inside the polygon
    data_clean_train = raw_df[inside]

    # Drop rows where any column value is null
    data_clean_train = data_clean_train.dropna()

    # Remove Outlier of high taxi fare:
    data_clean_train = data_clean_train[data_clean_train['fare_amount'] < data_clean_train['fare_amount'].quantile(0.999)]
    
    # Adding new feature 'distance' between pick up and drop off location:
    data_clean_train['distance'] = data_clean_train.apply(lambda x: haversine_distance((x['pickup_latitude'], x['pickup_longitude']),
                                                                                       (x['dropoff_latitude'], x['dropoff_longitude'])),axis=1)   
    # Extract year, month,hour and weekday from the pickup time.
    data_clean_train['year'] = pd.to_datetime(data_clean_train['pickup_datetime'], '%y/%m/%d %H:%M:%S').dt.year
    data_clean_train['month'] = pd.to_datetime(data_clean_train['pickup_datetime'], '%y/%m/%d %H:%M:%S').dt.month
    data_clean_train['hour'] = pd.to_datetime(data_clean_train['pickup_datetime'], '%y/%m/%d %H:%M:%S').dt.hour
    data_clean_train['weekday'] = pd.to_datetime(data_clean_train['pickup_datetime'], '%y/%m/%d %H:%M:%S').dt.weekday
    
    clean_train_df = data_clean_train.loc[:, ['key','pickup_datetime','fare_amount','distance'
                                              ,'year','month','hour','weekday',
                                              'pickup_latitude','pickup_longitude',
                                              'dropoff_latitude','dropoff_longitude'
                                              ]]
    return clean_train_df


def process_test_data(raw_df):
    """
    TODO: Copy your feature engineering code from task 1 here

    :param raw_df: the DataFrame of the raw test data
    :return: a DataFrame with the predictors created
    """
    data_clean_test = raw_df
   
    # Adding new feature 'distance' between pick up and drop off location:
    data_clean_test['distance'] = data_clean_test.apply(lambda x: haversine_distance((x['pickup_latitude'], x['pickup_longitude']),
                                                                                       (x['dropoff_latitude'], x['dropoff_longitude'])),axis=1)
    
    # Extract year, month,hour and weekday from the pickup time.
    data_clean_test['year'] = pd.to_datetime(data_clean_test['pickup_datetime'], '%y/%m/%d %H:%M:%S').dt.year
    data_clean_test['month'] = pd.to_datetime(data_clean_test['pickup_datetime'], '%y/%m/%d %H:%M:%S').dt.month
    data_clean_test['hour'] = pd.to_datetime(data_clean_test['pickup_datetime'], '%y/%m/%d %H:%M:%S').dt.hour
    data_clean_test['weekday'] = pd.to_datetime(data_clean_test['pickup_datetime'], '%y/%m/%d %H:%M:%S').dt.weekday
  
    clean_test_df = data_clean_test.loc[:, ['key','pickup_datetime','distance'
                                              ,'year','month','hour','weekday',
                                              'pickup_latitude','pickup_longitude',
                                              'dropoff_latitude','dropoff_longitude']]
    return clean_test_df


if __name__ == '__main__':
    # ===========================================
    # ==== Download data from Google Storage ====
    # ===========================================
    print('Downloading data from google storage')
    print('Sampling {} of the full dataset'.format(SAMPLE_PROB))
    input_bucket = storage.Client().bucket(DATA_BUCKET_ID)
    output_bucket = storage.Client().bucket(OUTPUT_BUCKET_ID)
    input_bucket.blob(TRAIN_FILE).download_to_filename('train.csv')

    raw_train = pd.read_csv('train.csv', parse_dates=["pickup_datetime"],
                            skiprows=lambda i: i > 0 and random.random() > SAMPLE_PROB)

    print('Read data: {}'.format(raw_train.shape))

    # =============================
    # ==== Data Transformation ====
    # =============================
    df_train = process_train_data(raw_train)

    # Prepare feature matrix X and labels Y
    X = df_train.drop(['key', 'fare_amount', 'pickup_datetime'], axis=1)
    Y = df_train['fare_amount']
    X_train, X_eval, y_train, y_eval = train_test_split(X, Y, test_size=0.33)
    print('Shape of feature matrix: {}'.format(X_train.shape))

    # ======================================================================
    # ==== Improve model performance with hyperparameter tuning ============
    # ======================================================================
    # You are provided with the code that creates an argparse.ArgumentParser
    # to parse the command line arguments and pass these parameters to Google Vertex AI
    # to be tuned by HyperTune.
    # TODO: Your task is to add at least 3 more arguments.
    # You need to update both the code below and config.yaml.

    parser = argparse.ArgumentParser()

    # the 5 lines of code below parse the --max_depth option from the command line
    # and will convert the value into "args.max_depth"
    # "args.max_depth" will be passed to XGBoost training through the `params` variables
    # i.e., xgb.train(params, ...)
    #
    # the 5 lines match the following YAML entry in `config.yaml`:
    # - parameterId: max_depth
    #   integerValueSpec:
    #       minValue: 4
    #       maxValue: 10
    # "- parameterId: max_depth" matches "--max_depth"
    # "minValue: 4" and "maxValue: 10" match "default=6"
    parser.add_argument(
        '--max_depth',
        default=6,
        type=int
    )

    # TODO: Create more arguments here, similar to the "max_depth" example
    # parser.add_argument(
    #     '--param2',
    #     default=...,
    #     type=...
    # )
    parser.add_argument(
        '--learning_rate',
        default=0.3,
        type=float
    )
    parser.add_argument(
        '--subsample',
        default=1.0,
        type=float
    )
    parser.add_argument(
        '--n_estimators',
        default=100,
        type=int
    )

    args = parser.parse_args()
    params = {
        'max_depth': args.max_depth,
        # TODO: Add the new parameters to this params dict, e.g.,
        # 'param2': args.param2
        'learning_rate': args.learning_rate,
        'subsample': args.subsample,
        'n_estimators': args.n_estimators
    }

    """
    DO NOT CHANGE THE CODE BELOW
    """
    # ===============================================
    # ==== Evaluate performance against test set ====
    # ===============================================
    # Create DMatrix for XGBoost from DataFrames
    d_matrix_train = xgb.DMatrix(X_train, y_train)
    d_matrix_eval = xgb.DMatrix(X_eval)
    model = xgb.train(params, d_matrix_train)
    y_pred = model.predict(d_matrix_eval)
    rmse = math.sqrt(mean_squared_error(y_eval, y_pred))
    print('RMSE: {:.3f}'.format(rmse))

    # Return the score back to HyperTune to inform the next iteration
    # of hyperparameter search
    hpt = HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='nyc_fare',
        metric_value=rmse)

    # ============================================
    # ==== Upload the model to Google Storage ====
    # ============================================
    JOB_NAME = os.environ['CLOUD_ML_JOB_ID']
    TRIAL_ID = os.environ['CLOUD_ML_TRIAL_ID']
    model_name = 'model.bst'
    model.save_model(model_name)
    blob = output_bucket.blob('{}/{}_rmse{:.3f}_{}'.format(
        JOB_NAME,
        TRIAL_ID,
        rmse,
        model_name))
    blob.upload_from_filename(model_name)
