"""
Card Recognizer Library
A reusable library for recognizing Pokémon TCG Pocket cards from screenshots.

Usage:
    from card_recognizer import CardRecognizer

    recognizer = CardRecognizer(db_path='tcg_pocket.db')
    results = recognizer.recognize(image_path='screenshot.png')

    for card in results:
        print(f"Found: {card['card_name']} - Similarity: {card['similarity']}%")
"""

import os
import sqlite3
from PIL import Image
from typing import List, Dict, Optional

from .color_hash import calculate_color_hash, hamming_distance
import io
import numpy as np
from .detect_layout import get_layout, get_layout_from_image # <-- Modificato
# =========================================================================
# 🎯 LAYOUT CONFIGURATION
# =========================================================================

LAYOUT_CROP_BOXES = {
    4: [
        (46, 16, 108, 53),      # Top Left
        (132, 17, 194, 54),     # Top Right
        (46, 131, 108, 168),    # Bottom Left
        (131, 133, 193, 170),   # Bottom Right
    ],
    5: [
        (5, 16, 69, 54),        # Top 1
        (87, 16, 151, 54),      # Top 2
        (170, 16, 234, 54),     # Top 3
        (46, 133, 108, 170),    # Bottom 1
        (130, 132, 193, 170)    # Bottom 2
    ],
    6: [
        (5, 16, 69, 54),        # Top 1
        (87, 16, 151, 54),      # Top 2
        (170, 16, 234, 54),     # Top 3
        (5, 133, 69, 171),      # Bottom 1
        (87, 133, 151, 171),    # Bottom 2
        (170, 133, 234, 171)    # Bottom 3
    ]
}


# =========================================================================
# 🎯 CARD RECOGNIZER CLASS
# =========================================================================

class CardRecognizer:
    """
    Main class for recognizing Pokémon TCG Pocket cards from screenshots.

    Attributes:
        db_path (str): Path to the SQLite database with card templates
        similarity_threshold (float): Minimum similarity % for a match (0-100)
    """

    def __init__(self, db_path: str = 'tcg_pocket.db', similarity_threshold: float = 60.0):
        """
        Initialize the CardRecognizer.

        Args:
            db_path: Path to tcg_pocket.db database
            similarity_threshold: Minimum similarity to consider a match (default: 60%)
        """
        self.db_path = db_path
        self.similarity_threshold = similarity_threshold
        self.template_hashes = None

        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found: {db_path}")

        self._load_templates()

    def _load_templates(self) -> None:
        """Load all template card hashes from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, card_name, card_number, set_code, local_image_path, color_hash
                    FROM cards
                    WHERE color_hash IS NOT NULL
                """)
                results = cursor.fetchall()

                self.template_hashes = []
                for row in results:
                    card_id, card_name, card_number, set_code, local_image_path, hash_str = row
                    if hash_str:
                        self.template_hashes.append({
                            'card_id': card_id,
                            'card_name': card_name,
                            'card_number': card_number,
                            'set_code': set_code,
                            'local_image_path': local_image_path,
                            'hash': hash_str
                        })

                if not self.template_hashes:
                    raise ValueError("No template hashes found in database. Run make_db.py first.")

        except sqlite3.Error as e:
            raise RuntimeError(f"Database error: {e}")



    def recognize_from_image(self, source_img: Image.Image, save_to_db: bool = False,
                             account_name: Optional[str] = None, image_path_for_db: Optional[str] = None) -> List[Dict]:
        """
        ✅ NUOVA FUNZIONE: Riconosce le carte da un oggetto PIL.Image in memoria.
        Questa è ora la funzione principale.
        """
        
        # Step 1: Detect layout (dall'immagine PIL)
        layout = get_layout_from_image(source_img)
        if layout not in LAYOUT_CROP_BOXES:
            raise ValueError(f"Could not determine valid layout. Detected: {layout}")

        crop_boxes = LAYOUT_CROP_BOXES[layout]
        cards_in_image = []

        # Step 3: Process each card position
        for position, crop_box in enumerate(crop_boxes, start=1):
            try:
                card_img = source_img.crop(crop_box)
                card_hash = calculate_color_hash(card_img)

                if not card_hash:
                    continue

                # Step 4: Find best match
                best_match = self._find_best_match(card_hash)

                if best_match and best_match.get('similarity', 0) >= self.similarity_threshold:
                    result = {
                        'position': position,
                        'card_id': best_match.get('card_id') or best_match.get('id'),
                        'card_name': best_match.get('card_name') or best_match.get('card'),
                        'card_number': best_match.get('card_number') or best_match.get('number'),
                        'set_code': best_match.get('set_code'),
                        'set_name': best_match.get('set_name'),
                        'rarity': best_match.get('rarity'),
                        'image_url': best_match.get('image_url'),
                        'local_image_path': best_match.get('local_image_path'),
                        'card_url': best_match.get('card_url'),
                        'set_url': best_match.get('set_url'),
                        'release_date': best_match.get('release_date'),
                        'total_cards': best_match.get('total_cards'),
                        'similarity': best_match.get('similarity'),
                        'layout': layout,
                    }
                    cards_in_image.append(result)

                    # Step 5: Optionally save to database
                    if save_to_db:
                        if not account_name:
                            raise ValueError("account_name required for save_to_db=True")
                        # Usa 'image_path_for_db' se fornito, altrimenti None
                        self._save_to_db(result, account_name, image_path_for_db)

            except Exception as e:
                print(f"Warning: Could not process position {position}: {e}")
                continue

        return cards_in_image


    def recognize(self, image_path: str, save_to_db: bool = False,
                account_name: Optional[str] = None) -> List[Dict]:
        """
        Funzione originale: Riconosce le carte da un PERCORSO file.
        MODIFICATO: Ora legge il file e chiama recognize_from_image.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            # 1. Apri l'immagine dal percorso
            source_img = Image.open(image_path)
        except Exception as e:
            raise RuntimeError(f"Could not open image: {e}")
        
        # 2. Chiama la nuova funzione in-memory
        # Passa image_path solo per il salvataggio nel DB
        return self.recognize_from_image(source_img, save_to_db, account_name, image_path)


    def _find_best_match(self, source_hash: str) -> Optional[Dict]:
        """
        Find the best matching template for a source hash and return full DB details.
        """
        best_match = None
        max_similarity = -1.0

        # 1️⃣ Calcolo della somiglianza
        for template in self.template_hashes:
            template_hash = template['hash']
            distance = hamming_distance(source_hash, template_hash)
            max_distance = len(template_hash)
            similarity = 100.0 - (distance / max_distance) * 100.0

            if similarity > max_similarity:
                max_similarity = similarity
                best_match = {**template, 'similarity': round(similarity, 2)}

        # 2️⃣ Estende con tutti i dettagli da DB
        if best_match:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()

                    # Dati carta
                    cursor.execute("""
                        SELECT 
                            c.id AS card_id,
                            c.set_code,
                            c.card_number,
                            c.card_name,
                            c.rarity,
                            c.image_url,
                            c.local_image_path,
                            c.card_url,
                            s.set_name,
                            s.release_date,
                            s.total_cards,
                            s.cover_image_path,
                            s.url AS set_url
                        FROM cards c
                        LEFT JOIN sets s ON c.set_code = s.set_code
                        WHERE c.id = ?
                    """, (best_match['card_id'],))
                    row = cursor.fetchone()

                    # Se trovato, aggiorna tutto
                    if row:
                        best_match.update(dict(row))
                    else:
                        # Se non c'è match, aggiungi campi vuoti per compatibilità
                        for key in [
                            "set_name", "release_date", "total_cards",
                            "cover_image_path", "set_url"
                        ]:
                            best_match.setdefault(key, None)

            except sqlite3.Error as e:
                print(f"⚠️ Database error while loading card info: {e}")
                for key in [
                    "set_name", "release_date", "total_cards",
                    "cover_image_path", "set_url"
                ]:
                    best_match.setdefault(key, None)

        return best_match




    def _save_to_db(self, result: Dict, account_name: str, image_path: str) -> None:
        """Save recognition result to database."""
        try:
            from datetime import datetime

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                cursor.execute("""
                    INSERT INTO found_cards 
                    (card_id, account_name, source_image_path, match_timestamp, 
                     similarity, position, layout)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    result['card_id'],
                    account_name,
                    image_path,
                    timestamp,
                    result['similarity'],
                    result['position'],
                    result['layout']
                ))
                conn.commit()

        except sqlite3.Error as e:
            print(f"Warning: Could not save to database: {e}")

    def batch_recognize(self, image_paths: List[str], 
                       save_to_db: bool = False,
                       account_name: Optional[str] = None) -> Dict[str, List[Dict]]:
        """
        Recognize cards in multiple screenshot images.

        Args:
            image_paths: List of paths to screenshot images
            save_to_db: If True, save all results to database
            account_name: Account name for database saving

        Returns:
            Dictionary mapping image_path -> list of recognized cards
        """
        all_results = {}

        for image_path in image_paths:
            try:
                results = self.recognize(image_path, save_to_db, account_name)
                all_results[image_path] = results
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                all_results[image_path] = []

        return all_results

    def set_threshold(self, threshold: float) -> None:
        """
        Change the similarity threshold.

        Args:
            threshold: New similarity threshold (0-100)
        """
        if not (0 <= threshold <= 100):
            raise ValueError("Threshold must be between 0 and 100")
        self.similarity_threshold = threshold


# =========================================================================
# 🎯 CONVENIENCE FUNCTIONS
# =========================================================================

def quick_recognize(image_path: str, db_path: str = 'tcg_pocket.db') -> List[Dict]:
    """
    Quick recognition without creating a recognizer instance.

    Args:
        image_path: Path to screenshot
        db_path: Path to database

    Returns:
        List of recognized cards
    """
    recognizer = CardRecognizer(db_path=db_path)
    return recognizer.recognize(image_path)
