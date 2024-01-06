from google.cloud import vision


class CloudVisionClient:
    """
    Use GCP Cloud Vision API to identify common NYC landmarks and their map coordinates.
    You should NOT change this class.

    Methods:
        get_landmarks(self, content): identify common NYC landmarks from images and return their information
    """

    def __init__(self):
        """
        Constructs all the necessary attributes for the GoogleMapsClient object
        """
        self.client = vision.ImageAnnotatorClient()

    def get_landmarks(self, content):
        """
        Use GCP Cloud Vision API to identify common NYC landmarks and their map coordinates


        Arguments:
            content(str): byte stream of the landmark image

        Returns:
            landmark: an object that contains label and coordinates of the landmark
        """
        image = vision.Image(content=content)

        response = self.client.landmark_detection(image=image)

        for landmark in response.landmark_annotations:
            return landmark
