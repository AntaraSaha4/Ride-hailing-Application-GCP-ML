import json
import logging
import os

import pandas as pd
import math
import base64
from flask import Flask, request
import json
from datetime import datetime
from datetime import datetime, timezone
import requests

from clients.vertex_ai import VertexAIClient
from clients.google_maps import GoogleMapsClient
from clients.natural_language import NaturalLanguageClient
from clients.text_to_speech import TextToSpeechClient
from clients.speech_to_text import SpeechToTextClient
from clients.cloud_vision import CloudVisionClient
from clients.vertex_ai_auto_ml import VertexAIAutoMLClient


app = Flask(__name__)

project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
endpoint_name = os.getenv("VERTEX_AI_MODEL_ENDPOINT")
location = os.getenv("LOCATION")
AutoML_endpoint_name = os.environ['AUTO_ML_MODEL_ENDPOINT_ID']

vertex_ai_client = VertexAIClient(project_id, endpoint_name, location)
speech_to_text_client = SpeechToTextClient()
text_to_speech_client = TextToSpeechClient()
natural_language_client = NaturalLanguageClient()
google_maps_client = GoogleMapsClient()
cloud_vision_client = CloudVisionClient()
vertex_ai_automl_client = VertexAIAutoMLClient(project_id, AutoML_endpoint_name)

#endpoint = "https://ml-fare-prediction-120818.ue.r.appspot.com"

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
    data_clean_test['year'] = pd.to_datetime(data_clean_test['pickup_datetime'], format='%y/%m/%d %H:%M:%S').dt.year
    data_clean_test['month'] = pd.to_datetime(data_clean_test['pickup_datetime'], format='%y/%m/%d %H:%M:%S').dt.month
    data_clean_test['hour'] = pd.to_datetime(data_clean_test['pickup_datetime'], format='%y/%m/%d %H:%M:%S').dt.hour
    data_clean_test['weekday'] = pd.to_datetime(data_clean_test['pickup_datetime'], format='%y/%m/%d %H:%M:%S').dt.weekday
  
    clean_test_df = data_clean_test.loc[:, ['distance'
                                              ,'year','month','hour','weekday',
                                              'pickup_latitude','pickup_longitude',
                                              'dropoff_latitude','dropoff_longitude']]
    return clean_test_df


@app.route('/')
def index():
    return "Hello"


@app.route('/predict', methods=['POST'])
def predict():
    raw_data_df = pd.read_json(request.data.decode('utf-8'),
                               convert_dates=["pickup_datetime"])
    predictors_df = process_test_data(raw_data_df)

    # return the predictions in the response in json format
    return json.dumps(vertex_ai_client.predict(predictors_df.values.tolist()))

@app.route('/farePrediction', methods=['POST'])
def fare_prediction():
    user_input = request.data

    # Convert the Speech To Text
    audio_content = user_input.decode('utf-8')
    text_result = speech_to_text_client.recognize(content_bytes= audio_content)

    # Convert the text to Entities
    response = natural_language_client.analyze_entities(text_result)
    final_entities = []
    for entity in response:
        final_entities.append(entity.name)
    origin = final_entities[0]
    destination = final_entities[1]

    # Directions
    directions_result = google_maps_client.directions(origin,destination)
    start_location = {
        "lng": directions_result[0]['legs'][0]['start_location']['lng'],
        "lat": directions_result[0]['legs'][0]['start_location']['lat']
    }

    end_location = {
        "lng": directions_result[0]['legs'][0]['end_location']['lng'],
        "lat": directions_result[0]['legs'][0]['end_location']['lat']
    }
    dir_data = {"start_location": start_location, "end_location": end_location}
    
    
    # Create Pick up and Drop off Lat and Long
    pickup_lng = dir_data["start_location"]["lng"]
    pickup_lat = dir_data["start_location"]["lat"]
    dropoff_lng = dir_data["end_location"]["lng"]
    dropoff_lat = dir_data["end_location"]["lat"]
    
    # Create Pick up datetime
    pickup_time = datetime.now(timezone.utc).strftime("%y/%m/%d %H:%M:%S")

    # Create dataframe for fare predictions
    raw_df = pd.DataFrame([{'pickup_latitude':pickup_lat, 'pickup_longitude':pickup_lng, 'dropoff_latitude':dropoff_lat,
                           'dropoff_longitude':dropoff_lng,'pickup_datetime':pickup_time}])
    
    # Send raw df to get predictors variables
    predictors_df = process_test_data(raw_df)

    # Get Fareprediction
    fare_response = vertex_ai_client.predict(predictors_df.values.tolist())

    # Ouput Response
    predicted_fare = "{:.2f}".format(fare_response[0])
    output_text = "Your expected fare from "+final_entities[0]+" to "+final_entities[1]+" is $"+predicted_fare

    # Text To Speech Conversion
    response = text_to_speech_client.synthesize_speech(output_text)
    speech = base64.b64encode(response).decode('utf-8')
    
    response_data = {"predicted_fare": predicted_fare, 
                     "entities": final_entities, 
                     "text": output_text, 
                     "speech": speech}

    return json.dumps(response_data)
    

@app.route('/speechToText', methods=['POST'])
def speech_to_text():
    audio_content = request.data.decode('utf-8')
    response_text = speech_to_text_client.recognize(content_bytes= audio_content)
    response_data = {'text':response_text}
    return json.dumps(response_data)

@app.route('/textToSpeech', methods=['GET'])
def text_to_speech():
    text = request.args.get('text')
    response = text_to_speech_client.synthesize_speech(text)
    speech = base64.b64encode(response).decode('utf-8')
    response_data = {'speech':speech}
    return json.dumps(response_data)

@app.route('/farePredictionVision', methods=['POST'])
def fare_prediction_vision():
    label_mapping = {"Jing_Fong": "Jing Fong", "Bamonte": "Bamonte's", "Katz_Deli": "Katz's Delicatessen", "ACME": "ACMENYC"}
    
    vision_ori_data = request.form.get('source')
    vision_dest_data = request.form.get('destination')

    # Origin location
    vision_ori_response = cloud_vision_client.get_landmarks(vision_ori_data)
    if vision_ori_response is not None:
        origin = vision_ori_response.description
    else:
        vertex_ori_name = vertex_ai_automl_client.predict_image(vision_ori_data)
        origin = label_mapping.get(vertex_ori_name,vertex_ori_name)

    # Destination location
    vision_dest_response = cloud_vision_client.get_landmarks(vision_dest_data)
    if vision_dest_response is not None:
        destination = vision_dest_response.description
    else:
        vertex_dest_name = vertex_ai_automl_client.predict_image(vision_dest_data)
        destination = label_mapping.get(vertex_dest_name,vertex_dest_name)

    # Directions
    directions_result = google_maps_client.directions(origin,destination)
    start_location = {
        "lng": directions_result[0]['legs'][0]['start_location']['lng'],
        "lat": directions_result[0]['legs'][0]['start_location']['lat']
    }
    end_location = {
        "lng": directions_result[0]['legs'][0]['end_location']['lng'],
        "lat": directions_result[0]['legs'][0]['end_location']['lat']
    }

    dir_data = {"start_location": start_location, "end_location": end_location}
    
    # Create Pick up and Drop off Lat and Long
    pickup_lng = dir_data["start_location"]["lng"]
    pickup_lat = dir_data["start_location"]["lat"]
    dropoff_lng = dir_data["end_location"]["lng"]
    dropoff_lat = dir_data["end_location"]["lat"]
    
    # Create Pick up datetime
    pickup_time = datetime.now(timezone.utc).strftime("%y/%m/%d %H:%M:%S")

    # Create dataframe for fare predictions
    raw_df = pd.DataFrame([{'pickup_latitude':pickup_lat, 'pickup_longitude':pickup_lng, 'dropoff_latitude':dropoff_lat,
                           'dropoff_longitude':dropoff_lng,'pickup_datetime':pickup_time}])
    
    # Send raw df to get predictors variables
    predictors_df = process_test_data(raw_df)

    # Get Fareprediction
    fare_response = vertex_ai_client.predict(predictors_df.values.tolist())

    # Ouput Response
    predicted_fare = "{:.2f}".format(fare_response[0])
    output_text = "Your expected fare from "+origin+" to "+destination+" is $"+predicted_fare

    # Text To Speech Conversion
    response = text_to_speech_client.synthesize_speech(output_text)
    speech = base64.b64encode(response).decode('utf-8')
    
    response_data = {"predicted_fare": predicted_fare, 
                     "entities": [origin,destination], 
                     "text": output_text, 
                     "speech": speech}

    return json.dumps(response_data)

@app.route('/namedEntities', methods=['GET'])
def named_entities():
    text = request.args.get('text')
    response = natural_language_client.analyze_entities(text)
    entities_list = []
    for entity in response:
        entities_list.append(entity.name)
    response_data = {"entities": entities_list}
    return json.dumps(response_data)

@app.route('/directions', methods=['GET'])
def directions():
    origin = request.args.get('origin')
    destination = request.args.get('destination')

    directions_result = google_maps_client.directions(origin,destination)
    start_location = {
        "lng": directions_result[0]['legs'][0]['start_location']['lng'],
        "lat": directions_result[0]['legs'][0]['start_location']['lat']
    }

    end_location = {
        "lng": directions_result[0]['legs'][0]['end_location']['lng'],
        "lat": directions_result[0]['legs'][0]['end_location']['lat']
    }
    response_data = {"start_location": start_location, "end_location": end_location}
    return json.dumps(response_data)


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    app.run()
