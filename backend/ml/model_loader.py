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

CLASS_NAMES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
               "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", 
               "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]

def idx_to_label(idx: int) -> str:
    """
    Map model output index to class label using CLASS_NAMES.
    """
    if CLASS_NAMES and 0 <= idx < len(CLASS_NAMES):
        return CLASS_NAMES[idx]
    return str(idx)