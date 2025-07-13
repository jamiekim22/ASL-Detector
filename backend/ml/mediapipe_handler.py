import cv2
import mediapipe as mp
import numpy as np

_mp_hands = mp.solutions.hands
_mp_drawing = mp.solutions.drawing_utils
_hands_detector = _mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def detect_hand_roi(img: np.ndarray, padding: float = 0.1) -> np.ndarray:
    """
    Detects a single hand in the image and returns a cropped ROI with optional padding.
    Raises ValueError if no hand detected.
    """
    # Convert BGR to RGB for MediaPipe
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = _hands_detector.process(rgb)
    if not results.multi_hand_landmarks:
        raise ValueError("No hand detected")
    # Use first detected hand
    landmarks = results.multi_hand_landmarks[0]
    h, w, _ = img.shape
    xs = [lm.x for lm in landmarks.landmark]
    ys = [lm.y for lm in landmarks.landmark]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    # Apply padding in normalized coords
    x_pad = (x_max - x_min) * padding
    y_pad = (y_max - y_min) * padding
    # Convert to pixel coordinates
    x1 = max(int((x_min - x_pad) * w), 0)
    x2 = min(int((x_max + x_pad) * w), w)
    y1 = max(int((y_min - y_pad) * h), 0)
    y2 = min(int((y_max + y_pad) * h), h)
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        raise ValueError("Empty ROI extracted")
    return roi

def draw_hand_landmarks(img: np.ndarray) -> np.ndarray:
    """
    Draws hand landmarks on the image and returns annotated image.
    """
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = _hands_detector.process(rgb)
    annotated = img.copy()
    if results.multi_hand_landmarks:
        for hand_lm in results.multi_hand_landmarks:
            _mp_drawing.draw_landmarks(annotated, hand_lm, _mp_hands.HAND_CONNECTIONS)
    return annotated