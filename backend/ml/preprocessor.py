# Image preprocessing pipeline
import base64
import cv2
import numpy as np

TARGET_SIZE = (64, 64) # Target input size for the model (height, width)

def decode_base64_image(image_str: str) -> np.ndarray:
    """
    Decode a base64-encoded image string to an OpenCV BGR image array.
    """
    # Strip data URI prefix if present (e.g., 'data:image/jpeg;base64,')
    if image_str.startswith('data:') and ',' in image_str:
        # keep only the Base64 part
        image_str = image_str.split(',', 1)[1]
    try:
        img_bytes = base64.b64decode(image_str)
    except Exception as e:
        raise ValueError(f"Invalid Base64 data: {e}")
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