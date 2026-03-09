"""
Consultation configuration settings for the HERO system.
"""


class ConsultConfig:
    """
    Configuration flags for a consultation session.

    Attributes:
        speech: Whether text-to-speech is enabled.
        text: Whether text display is enabled.
        output_lang: BCP-47 language tag for TTS output.
        input_lang: BCP-47 language tag for speech recognition input.
    """

    def __init__(self, speech=True):
        """
        Initialise configuration with default settings.

        Args:
            speech: If True, text-to-speech is enabled during the session.
        """
        self.speech = speech
        self.text = True
        self.output_lang = 'en'
        self.input_lang = 'en'
