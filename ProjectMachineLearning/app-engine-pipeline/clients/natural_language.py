from google.cloud import language_v1 as language


class NaturalLanguageClient:
    """
    Identify NYC landmarks as entities in plain text
    You should NOT change this class.

    Methods:
        analyze_entities(self, text):  identifies NYC landmarks as entities and return entities list
    """

    def __init__(self):
        """
        Constructs all the necessary attributes for the NaturalLanguageClient object
        """
        self.client = language.LanguageServiceClient()

    def analyze_entities(self, text):
        """
        Identifies NYC landmarks as entities in plain text and return entities list

        Arguments:
            text(str): plain text string that contains NYC landmarks

        Returns:
            entities as a list
        """
        document = language.types.Document(
            content=text,
            type="PLAIN_TEXT",
        )

        response = self.client.analyze_entities(document=document, encoding_type='UTF32')
        return response.entities
