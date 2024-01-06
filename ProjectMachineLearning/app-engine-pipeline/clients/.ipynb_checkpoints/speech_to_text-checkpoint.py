from google.cloud import speech_v1 as speech


class SpeechToTextClient:
    """
    Transform speech files(bytes) into plain text(str)
    You should NOT change this class.

    Methods:
        recognize(self, content_bytes): transforms byte stream and returns a text string
    """

    def __init__(self):
        """
        Constructs all the necessary attributes for the SpeechToTextClient object
        """
        self.client = speech.SpeechClient()

        self.config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code='en-US')

    def recognize(self, content_bytes):
        """
        Transforms byte stream and returns a text string

        Arguments:
            content_bytes(bytes): byte stream

        Returns:
            transformed text as string
        """
        audio = speech.RecognitionAudio(content=content_bytes)

        response = self.client.recognize(config=self.config, audio=audio)

        for result in response.results:
            return result.alternatives[0].transcript
