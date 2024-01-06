from google.cloud import aiplatform

class VertexAIAutoMLClient:
    """
    VertexAIAutoMLClient recognize specific NYC restaurants images according to your Vertex AI AutoML model.
    You should NOT change this class.

    Methods:
        predict_image(self, content): identify specific NYC restaurants
    """

    def __init__(self, project_id, endpoint_name):
        """
        Constructs all the necessary attributes for the VertexAIAutoMLClient object

        Parameters:
            project_id (str): project ID of your GCP project
            endpoint_name (str): ID of your Vertex AI AutoML model endpoint
        """
        # initialize the aiplatform client
        aiplatform.init(project=project_id, location="us-central1")

        # initialize the endpoint which the model deployed to
        self.endpoint = aiplatform.Endpoint(endpoint_name)

    def predict_image(self, content):
        """
        Use the Vertex AI AutoML model to identify specific NYC restaurants

        Arguments:
            content(str): byte stream of a specific NYC restaurant image

        Returns:
            an object that contains label of the restaurant
        """
        instances = [{"content": content}]

        prediction = self.endpoint.predict(instances=instances).predictions[0]

        # get the prediction result with highest confidence score
        highest_confidence_index = prediction['confidences'].index(max(prediction['confidences']))

        # get the displayName associated with the highest confidence score
        highest_confidence_displayName = prediction['displayNames'][highest_confidence_index]

        return highest_confidence_displayName

