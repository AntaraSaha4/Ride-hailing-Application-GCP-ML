import googlemaps
import os


class GoogleMapsClient:
    """
    Parse addresses and resolve coordinates to the address
    You should NOT change this class.

    Methods:
        directions(self, origin, destination): input origin and destination as string and
        return an Object containing latitude and longitude of the addresses
    """

    def __init__(self):
        """
        Constructs all the necessary attributes for the GoogleMapsClient object
        """
        self.client = googlemaps.Client(key=os.environ['GOOGLE_MAPS_API_KEY'])

    def directions(self, origin, destination):
        """
        Transforms byte stream and returns a text string

        Arguments:
            origin(str): the origin address
            destination(str): the destination address

        Returns:
            (list): a list object containing latitude and longitude of the addresses
        """
        directions_result = self.client.directions(origin,
                                                   destination,
                                                   mode="driving")

        return directions_result
