# Image preprocessing pipeline
import base64
import cv2
import numpy as np

TARGET_SIZE = (64, 64) # Target input size for the model (height, width)

def decode_base64_image(image_str: str) -> np.ndarray:
    """
    Decode a base64-encoded image string to an OpenCV BGR image array.
    """
    # handle data URI prefix if present
    header, _, data = image_str.partition(',')
    img_bytes = base64.b64decode(data if data else header)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Failed to decode base64 image")
    return img

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Resize to model input size and normalize pixel values to [0,1].
    Returns a batch of one image.
    """
    resized = cv2.resize(img, TARGET_SIZE)
    normalized = resized.astype(np.float32) / 255.0

    return np.expand_dims(normalized, axis=0)   # add batch dimension