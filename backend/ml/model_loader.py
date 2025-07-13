# Tensorflow model loading
import tensorflow as tf
from ..config import settings

_model = None

def load_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(settings.model_path)
    return _model

def get_model():
    return _model