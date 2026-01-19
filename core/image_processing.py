"""image_processing.py - Elaborazione immagini e color matching"""

# Import standard library
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, List, Dict, Optional
import os

from scipy.fft import dct
# Import color_hash (libreria esterna)
from .color_hash import calculate_color_hash, hamming_distance

# Import configurazione
from config import (
    SIMILARITY_THRESHOLD, 
    TEMPLATE_CROP_BOX, 
    SOURCE_ROI_BOXES,
    HASH_SIZE, 
    TEMPLATE_DOWNSCALE_FACTOR, 
    COLOR_TOLERANCE,
    TARGET_GRAY_BGR
)

def is_color_in_range(avg_color, target_color, tolerance):
    """Checks if the average color is within the tolerance of the target color."""
    return all(target - tolerance <= avg <= target + tolerance
               for avg, target in zip(avg_color, target_color))


def calculate_color_hash(image, hash_size=24):
    """
    Calculates a perceptual hash of a PIL image based on DCT of color channels.
    """
    try:
        # 1. Resize image
        img = image.convert('RGB')
        img = img.resize((hash_size, hash_size), Image.Resampling.LANCZOS)

        # 2. Get pixel data and separate channels
        pixels = np.array(img)
        r = pixels[:, :, 0]
        g = pixels[:, :, 1]
        b = pixels[:, :, 2]

        # 3. Compute DCT for each channel
        dct_r = dct(dct(r, axis=0, norm='ortho'), axis=1, norm='ortho')
        dct_g = dct(dct(g, axis=0, norm='ortho'), axis=1, norm='ortho')
        dct_b = dct(dct(b, axis=0, norm='ortho'), axis=1, norm='ortho')

        # 4. Reduce the DCT and calculate averages
        freq_size = hash_size // 4
        dct_r_reduced = dct_r[:freq_size, :freq_size]
        dct_g_reduced = dct_g[:freq_size, :freq_size]
        dct_b_reduced = dct_b[:freq_size, :freq_size]

        avg_r = (np.sum(dct_r_reduced) - dct_r_reduced[0, 0]) / (freq_size * freq_size - 1)
        avg_g = (np.sum(dct_g_reduced) - dct_g_reduced[0, 0]) / (freq_size * freq_size - 1)
        avg_b = (np.sum(dct_b_reduced) - dct_b_reduced[0, 0]) / (freq_size * freq_size - 1)

        # 5. Construct the hash
        hash_string = []
        for y in range(freq_size):
            for x in range(freq_size):
                if x == 0 and y == 0:
                    continue
                hash_string.append('1' if dct_r_reduced[y, x] > avg_r else '0')
                hash_string.append('1' if dct_g_reduced[y, x] > avg_g else '0')
                hash_string.append('1' if dct_b_reduced[y, x] > avg_b else '0')
        
        return "".join(hash_string)

    except Exception as e:
        print(f"Could not calculate color hash: {e}")
        return None


def hamming_distance(hash1, hash2):
    """Calculates the Hamming distance between two hashes."""
    if len(hash1) != len(hash2):
        # This case should ideally not happen if hash_size is consistent
        return len(hash1) # Return max distance

    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))


def calculate_color_hash_from_path(image_path, hash_size=24):
    try:
        img = Image.open(image_path)
        return calculate_color_hash(img, hash_size)
    except Exception as e:
        print(f"Could not open image at {image_path}: {e}")
        return None
