"""
Consultation configuration settings.
"""


class ConsultConfig:
    """Configuration for the consultation session."""

    def __init__(self, speech=True):
        self.speech = speech
        self.text = True
        self.output_lang = 'en'
        self.input_lang = 'en'
