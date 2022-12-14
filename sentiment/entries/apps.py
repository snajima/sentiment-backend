from django.apps import AppConfig
from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline

emotion_model = None

class EntriesConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'entries'

    def ready(self):
        global emotion_model
        emotion_model = pipeline('sentiment-analysis', 
                            model='arpanghoshal/EmoRoBERTa')