from google.cloud import aiplatform

class VertexAIClient:
    """
    Make predictions using the model you uploaded to Vertex AI platform
    You should NOT change this class.

    Methods:
        predict(self, instances): make predictions using the model and return predicted fare
    """

    def __init__(self, project_id, endpoint_name, location):
        """
        Constructs all the necessary attributes for the AIPlatformClient object

        Parameters:
            project_id (str): project ID of your GCP project
            endpoint_name (str): endponint name of the model you are using
            location (str): GCP region
        """

        # initialize the aiplatform client
        aiplatform.init(project=project_id, location=location)

        # initialize the endpoint which the model deployed to
        self.endpoint = aiplatform.Endpoint(endpoint_name)

    def predict(self, instances):
        """
        Make predictions using the model you uploaded to Vertex AI platform

        Arguments:
            instances: a DataFrame with the predictors created

        Returns:
            predicted fare
        """
        # get the response from the model deployment endpoint
        response = self.endpoint.predict(instances=instances)

        return response.predictions
