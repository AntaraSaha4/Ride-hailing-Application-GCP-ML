"""
A short script to test the local server.
"""
import requests
import pandas as pd
import base64
import unittest
import json

endpoint = "https://ml-fare-prediction-120818.ue.r.appspot.com"

class TestAPIMethods(unittest.TestCase):

    def test_predict(self):

        record_data = pd.read_csv('./test_dataset/predict_input.csv').to_json(orient='records')
        response = requests.post('{}/predict'.format(endpoint), data=record_data)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(is_json(response.text))
        
    def test_text_to_speech(self):

        """ This functional test depends on Cloud Text-to-Speech for which the model may get updated over time, thus the base64 result is not fixed.
        Hence, this test case only validates the JSON structure and response code, but not the concrete base64 value.

        """

        text = u"Your expected fare from Charging Bull to Carnegie Hall is $23.78"
        response = requests.get('{}/textToSpeech'.format(endpoint), params={'text': text})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(is_json(response.text))

    def test_speech_to_text(self):

        with open("./test_dataset/the_cooper_union_the_juilliard_school.wav", "rb") as f:
            speech = f.read()
        speech_data = str(base64.b64encode(speech).decode("utf-8"))
        response = requests.post('{}/speechToText'.format(endpoint), data=speech_data)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(is_json(response.text))
        
        actual_text_result = json.loads(response.text)
        expected_text_result = json.loads(u"{\"text\": \"I would like to go from the Cooper Union to the Juilliard School\"}")
        self.assertEqual(actual_text_result, expected_text_result)

    def test_named_entities(self):

        test_entities = u"American Museum of Natural History and Bryant Park"
        response = requests.get('{}/namedEntities'.format(endpoint), params={'text': test_entities})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(is_json(response.text))
        
        actual_entities_result = json.loads(response.text)
        expected_entities_result = json.loads(u"{\"entities\": [\"American Museum of Natural History\", \"Bryant Park\"]}")
        self.assertEqual(actual_entities_result, expected_entities_result)
    
    def test_directions(self):

        """ This functional test depends on Google Direction APIs for which the exact results may get updated over time.
        Hence, this test case only validates the JSON structure and response code, but not the concrete values.
        """

        response = requests.get('{}/directions'.format(endpoint), params={'origin': u"Pennsylvania Station", 'destination': u"Times Square"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(is_json(response.text))

    def test_fare_prediction(self):

        with open("./test_dataset/the_cooper_union_the_juilliard_school.wav", "rb") as f:
            speech = f.read()
        speech_data = str(base64.b64encode(speech).decode("utf-8"))

        response = requests.post('{}/farePrediction'.format(endpoint), data=speech_data)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(is_json(response.text))

    def test_fare_prediction_vision(self):
        
        with open("./test_dataset/katzs_delicatessen.jpg", 'rb') as f:
            ori = f.read()
        with open("./test_dataset/bamonte.jpg", 'rb') as f:
            dest = f.read()
        vision_ori_data = str(base64.b64encode(ori).decode("utf-8"))
        vision_dest_data = str(base64.b64encode(dest).decode("utf-8"))

        response = requests.post('{}/farePredictionVision'.format(endpoint), data={'source': vision_ori_data,
                                                        'destination': vision_dest_data})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(is_json(response.text))

def is_json(res):

    """Validate JSON response.
    
    Keyword arguments:
    res -- the response string
    """
    try:
        json_object = json.loads(res)
    except:
        return False
    return True

if __name__ == '__main__':
    unittest.main()
