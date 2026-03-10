import cv2
import numpy as np
import io
from PIL import Image

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------

TARGET_GRAY_BGR = (242, 232, 222)
COLOR_TOLERANCE = 3
TOP_ROW_CHECK_BOX = (119, 27, 123, 40)
BOTTOM_ROW_CHECK_BOX = (119, 134, 123, 147)

# -------------------------------------------------------------------------
# LAYOUT DETECTION LOGIC
# -------------------------------------------------------------------------

def is_color_in_range(avg_color, target_color, tolerance):
    """Checks if the average color is within the tolerance of the target color."""
    return all(target - tolerance <= avg <= target + tolerance
               for avg, target in zip(avg_color, target_color))

def _get_layout_logic(img) -> int:
    """Logica di base che opera su un'immagine CV2."""
    x1, y1, x2, y2 = TOP_ROW_CHECK_BOX
    top_roi = img[y1:y2, x1:x2]
    avg_color_top = np.mean(top_roi, axis=(0, 1))
    top_is_gray = is_color_in_range(avg_color_top, TARGET_GRAY_BGR, COLOR_TOLERANCE)
    cards_in_top_row = 2 if top_is_gray else 3
    
    x1, y1, x2, y2 = BOTTOM_ROW_CHECK_BOX
    bottom_roi = img[y1:y2, x1:x2]
    avg_color_bottom = np.mean(bottom_roi, axis=(0, 1))
    bottom_is_gray = is_color_in_range(avg_color_bottom, TARGET_GRAY_BGR, COLOR_TOLERANCE)
    cards_in_bottom_row = 2 if bottom_is_gray else 3
    
    return cards_in_top_row + cards_in_bottom_row

def get_layout(image_path: str) -> int:
    """
    Funzione originale: determina il layout da un PERCORSO file.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        return _get_layout_logic(img)
    except Exception:
        return None

def get_layout_from_bytes(image_bytes: bytes) -> int:
    """
    ✅ NUOVA FUNZIONE: determina il layout da BYTES in memoria.
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        return _get_layout_logic(img)
    except Exception:
        return None

# ✅ NUOVA FUNZIONE: determina il layout da un'immagine PIL
def get_layout_from_image(pil_image: Image.Image) -> int:
    """Determina il layout da un'immagine PIL."""
    try:
        # Converti PIL Image in array OpenCV (BGR)
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return _get_layout_logic(img)
    except Exception:
        return None