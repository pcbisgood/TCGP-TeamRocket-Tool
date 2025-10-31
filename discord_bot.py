# =========================================================================
# IMPORTS 
# =========================================================================

# Standard Library
import sys
import json
import os
import subprocess
import ctypes
from datetime import datetime
import time
import random
import re
import asyncio
import hashlib
import sqlite3
import secrets
from typing import Optional, Tuple, List
from threading import Lock
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from dotenv import load_dotenv

# PyQt5 - UI
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QTextBrowser,
    QGroupBox, QProgressBar, QFileDialog, QMessageBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame, QSystemTrayIcon,
    QMenu, QAction, QStyle, QDialog, QCheckBox, QSpinBox, QScrollArea,
    QGridLayout, QComboBox, QToolButton
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QFont, QPalette, QColor, QPixmap, QIcon, qGray, qRgb, QPainter, QBrush, QPen

# Discord Bot
import discord
from discord.ext import commands

# Web & HTTP
import aiohttp
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Flask
from flask import request, session, render_template
from flask import Flask, send_file
import mimetypes

# Image Processing
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
import imagehash

# Windows Notifications (Optional)
try:
    from windows_toasts import Toast, WindowsToaster, ToastDisplayImage, ToastImagePosition
    WINDOWS_TOAST_AVAILABLE = True
except ImportError:
    WINDOWS_TOAST_AVAILABLE = False
    print("⚠️ windows-toasts not installed")



def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller."""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)

def get_app_data_path(filename):
    """Get path in user's AppData for persistent storage."""
    app_data_dir = os.path.join(os.getenv('APPDATA'), 'TCGPTeamRocketTool')
    os.makedirs(app_data_dir, exist_ok=True)
    return os.path.join(app_data_dir, filename)

# ============================================================================
# PATHS - USE THESE FUNCTIONS
# ============================================================================

# Static resources (bundled with EXE)
ICON_PATH = get_resource_path("gui/icon.ico")
BACKGROUND_PATH = get_resource_path("gui/background.png")
CLOUDFLARED_PATH = get_app_data_path("cloudflared.exe")

# =========================================================================
# 🎯 GLOBAL CONFIGURATION
# =========================================================================

# Discord Bot Configuration
SEARCH_STRING = "Tradeable cards"
LOG_FILENAME = 'trade_log.json'
ACCOUNTS_DIR = get_app_data_path("Accounts")
TCG_IMAGES_DIR = get_app_data_path("tcg_images")
os.makedirs(ACCOUNTS_DIR, exist_ok=True)
os.makedirs(TCG_IMAGES_DIR, exist_ok=True)
ACCOUNT_NAME_REGEX = r'by\s+(.*?)\s+in instance:'
BATCH_SIZE = 500
CHUNK_SIZE = 1024 * 128
MAX_RETRIES = 3
RETRY_DELAY = 1
MAX_CONCURRENT_DOWNLOADS = 20
SAVE_INTERVAL = 100
ACCOUNT_NAME_PATTERN = re.compile(ACCOUNT_NAME_REGEX)
CLOUDFLARE_PASSWORD = os.getenv('CLOUDFLARE_PASSWORD', '')

# Database Configuration
DB_FILENAME = get_app_data_path("tcg_pocket.db")
PROXIES_FILE = get_app_data_path("proxies.txt")
# Card Scanner Configuration (OpenCV Template Matching)
TEMPLATE_DOWNSCALE_FACTOR = 0.20
SIMILARITY_THRESHOLD = 0.65
MAX_WORKERS = os.cpu_count() or 4
TARGET_RARITIES = ('◊◊◊◊', '☆', '☆☆', 'Crown Rare')
TEMPLATE_CROP_BOX = (25, 50, 340, 240)
SOURCE_ROI_BOXES = [(40, 15, 200, 60), (40, 130, 200, 175)]

# Scraper Configuration
PROXY_API_URL = "https://api.proxyscrape.com/v4/free-proxy-list/get?request=get_proxies&skip=0&proxy_format=protocolipport&format=json&limit=1000"

# Layout Detection Configuration (OPTIONAL - if you want to use detect_layout.py)
TARGET_GRAY_BGR = (242, 232, 222)
COLOR_TOLERANCE = 3
TOP_ROW_CHECK_BOX = (119, 27, 123, 40)
BOTTOM_ROW_CHECK_BOX = (119, 134, 123, 147)
load_dotenv()
# =========================================================================
# 🛠️ UTILITY FUNCTIONS
# =========================================================================

def enhance_card_cover(image_path, output_path):
    """Enhance cover quality during scraping"""
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return False
    
    # 1. Denoise (removes noise)
    img = cv2.fastNlMeansDenoisingColored(img, None, h=10, hForForColor=10, templateWindowSize=7, searchWindowSize=21)
    
    # 2. Sharpening (sharpness)
    kernel = np.array([[-1,-1,-1], [-1, 5,-1], [-1,-1,-1]]) / 1
    img = cv2.filter2D(img, -1, kernel)
    
    # 3. Equalize histogram (contrast)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    img = cv2.merge([l, a, b])
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    
    # Save with OpenCV
    cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    return True

def optimize_cover_file(image_path, target_size=(400, 560)):
    """Optimize and compress cover for web"""
    
    # Open with Pillow
    img = Image.open(image_path)
    
    # Resize while maintaining aspect ratio
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    
    # Enhance contrast and sharpness
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)
    
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.5)
    
    # Save optimized (webp is lighter)
    webp_path = image_path.replace('.png', '.webp').replace('.jpg', '.webp')
    img.save(webp_path, 'WEBP', quality=85, method=6)
    
    return webp_path

def get_file_hash_fast(filepath: str) -> Optional[str]:
    """Calculate the MD5 hash of a file."""
    if not os.path.exists(filepath):
        return None
    try:
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except:
        return None



def require_password(f):
    """Decorator to require password for public access"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # ✅ ALWAYS reload .env (so the password is up to date)
        load_dotenv(override=True)
        cloudflare_password = os.getenv('CLOUDFLARE_PASSWORD', '').strip()
        
        # Check if it's a real localhost access (not via tunnel)
        is_localhost = (
            request.remote_addr == '127.0.0.1' and
            not request.headers.get('CF-Connecting-IP')
        )
        
        if is_localhost:
            return f(*args, **kwargs)
        
        if session.get('authenticated'):
            return f(*args, **kwargs)
        
        if request.method == 'POST':
            password = request.form.get('password', '').strip()
            
            if not cloudflare_password:
                return render_template('login.html', error="Password not configured")
            
            if password == cloudflare_password:  # Use the updated local variable
                session['authenticated'] = True
                return f(*args, **kwargs)
            else:
                return render_template('login.html', error="Incorrect password")
        
        return render_template('login.html')
    
    return decorated_function

def extract_trade_data_fast(message):
    """
    Estrae i dati del trade da un messaggio Discord.
    
    Priorità per account_name:
    1. Nome file XML se presente come allegato
    2. Nome file XML dal testo "File name: xxx.xml"
    3. "unknown_account" come fallback
    """
    content = message.content
    
    # 1️⃣ Estrai l'account_name
    account_name = "unknown_account"
    
    # PRIORITÀ 1: Estrai dal file XML allegato
    for att in message.attachments:
        if att.filename.endswith(".xml"):
            account_name = att.filename.replace(".xml", "").strip()
            break
    
    # PRIORITÀ 2: Se non trovato negli allegati, estrai da "File name: xxx.xml" nel testo
    if account_name == "unknown_account":
        file_pattern = r'File name: ([\w\-\(\)]+\.xml)'
        match = re.search(file_pattern, content)
        if match:
            xml_filename = match.group(1)
            account_name = xml_filename.replace(".xml", "").strip()
    
    # 2️⃣ Estrai il nome del file XML dal testo
    xml_filename_text = "N/A"
    file_line_match = re.search(r'File name: ([\w\-\(\)\.]+)', content)
    if file_line_match:
        xml_filename_text = file_line_match.group(1)
    
    # 3️⃣ Estrai le carte trovate (formato "Found: CardName (xN)")
    cards_found = ""
    cards_pattern = r'Found: ([\w\s]+(?:\s*\(x\d+\))?(?:,\s*[\w\s]+\s*\(x\d+\))*)'
    cards_match = re.search(cards_pattern, content)
    if cards_match:
        cards_found = cards_match.group(1).strip()
    
    # 4️⃣ Estrai gli allegati (XML e immagine)
    xml_att = None
    image_att = None
    
    for att in message.attachments:
        if not xml_att and att.filename.endswith('.xml'):
            xml_att = att
        elif not image_att and att.filename.endswith(('.png', '.jpg', '.jpeg', '.webp')):
            image_att = att
        
        if xml_att and image_att:
            break
    
    return {
        "message_id": message.id,
        "account_name": account_name,
        "xml_filename_text": xml_filename_text,
        "cards_found": cards_found,
        "message_link": message.jump_url
    }, xml_att, image_att


    
async def download_attachment_fast(session: aiohttp.ClientSession, attachment,
                                   sub_folder: str, filename: str) -> Tuple[Optional[str], str]:
    """Downloads an attachment from Discord."""
    os.makedirs(sub_folder, exist_ok=True)
    file_path = os.path.join(sub_folder, filename)
    
    if os.path.exists(file_path):
        return file_path, 'skipped'
    
    for attempt in range(MAX_RETRIES):
        try:
            async with session.get(attachment.url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                if resp.status == 200:
                    with open(file_path, 'wb') as f:
                        async for chunk in resp.content.iter_chunked(CHUNK_SIZE):
                            f.write(chunk)
                    return file_path, 'downloaded'
                elif attempt == MAX_RETRIES - 1:
                    return None, 'failed'
        except:
            if attempt == MAX_RETRIES - 1:
                return None, 'failed'
        await asyncio.sleep(RETRY_DELAY)
    
    return None, 'failed'

def load_trade_log_fast():
    """Loads the trade log from the JSON file."""
    if os.path.exists(LOG_FILENAME):
        try:
            with open(LOG_FILENAME, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

def save_trade_log_fast(trade_log):
    """Saves the trade log to the JSON file."""
    try:
        with open(LOG_FILENAME, 'w', encoding='utf-8') as f:
            json.dump(trade_log, f, ensure_ascii=False, separators=(',', ':'))
    except:
        pass

def is_color_in_range(avg_color, target_color, tolerance):
    """Checks if the average color is within the tolerance of the target color."""
    return all(target - tolerance <= avg <= target + tolerance
               for avg, target in zip(avg_color, target_color))

def get_layout(image_path):
    """Analyzes the image to determine the card layout (4, 5, or 6 cards) - OPTIONAL."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
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
    except Exception:
        return None

# =========================================================================
# 🗄️ DATABASE MANAGEMENT CON MIGRAZIONE
# =========================================================================

class DatabaseManager:
    """Gestisce il database e le migrazioni."""
    
    def __init__(self, db_filename=DB_FILENAME, log_callback=None):
        self.db_filename = db_filename
        self.log_callback = log_callback or print
        self.conn = None
        self.db_lock = Lock()
        
    def connect(self):
        """Connette al database."""
        self.conn = sqlite3.connect(self.db_filename, check_same_thread=False)
        self.cursor = self.conn.cursor()
        
    def setup_database(self):
        """Crea il database e le tabelle necessarie."""
        if not self.conn:
            self.connect()
        
        # Tabella sets
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS sets (
                set_code TEXT PRIMARY KEY,
                set_name TEXT,
                cover_image_path TEXT,
                release_date TEXT,
                total_cards INTEGER,
                url TEXT
            )
        """)
        
        # Tabella cards
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS cards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                set_code TEXT,
                card_number TEXT,
                card_name TEXT,
                rarity TEXT,
                image_url TEXT,
                local_image_path TEXT,
                card_url TEXT,
                FOREIGN KEY (set_code) REFERENCES sets(set_code),
                UNIQUE(set_code, card_number)
            )
        """)
        
        # Tabella accounts
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS accounts (
                account_id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_name TEXT UNIQUE NOT NULL
            )
        """)
        
        # Tabella account_inventory
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS account_inventory (
                account_id INTEGER,
                card_id INTEGER,
                quantity INTEGER DEFAULT 0,
                last_updated TEXT,
                FOREIGN KEY (account_id) REFERENCES accounts(account_id),
                FOREIGN KEY (card_id) REFERENCES cards(id),
                UNIQUE(account_id, card_id)
            )
        """)
        
        # Tabella found_cards (per tracking dei match)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS found_cards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                card_id INTEGER,
                account_name TEXT,
                source_image_path TEXT,
                match_timestamp TEXT,
                similarity_score REAL,
                FOREIGN KEY (card_id) REFERENCES cards(id)
            )
        """)
        
        # Tabella wishlist
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS wishlist (
                card_id INTEGER PRIMARY KEY,
                added_date TEXT,
                FOREIGN KEY (card_id) REFERENCES cards(id)
            )
        """)


        self.conn.commit()
        
        # Esegui migrazioni se necessario
        self.migrate_database()


    def update_all_phash(self, log_callback=None):
        """Calcola e aggiorna i pHash per tutte le carte con immagini."""
        log_callback = log_callback or print
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, local_image_path 
            FROM cards 
            WHERE local_image_path IS NOT NULL AND (phash IS NULL OR phash = '')
        """)
        cards_to_update = cursor.fetchall()
        
        log_callback(f"🔄 Updating pHash for {len(cards_to_update)} cards...")
        
        updated = 0
        for card_id, image_path in cards_to_update:
            if os.path.exists(image_path):
                try:
                    img = Image.open(image_path)
                    cropped = img.crop(TEMPLATE_CROP_BOX)
                    new_size = (int(cropped.width * TEMPLATE_DOWNSCALE_FACTOR), 
                               int(cropped.height * TEMPLATE_DOWNSCALE_FACTOR))
                    resized = cropped.resize(new_size, Image.LANCZOS)
                    phash = str(imagehash.phash(resized))
                    
                    cursor.execute("UPDATE cards SET phash = ? WHERE id = ?", (phash, card_id))
                    updated += 1
                    
                    if updated % 50 == 0:
                        log_callback(f"   Progress: {updated}/{len(cards_to_update)}")
                        self.conn.commit()
                except Exception as e:
                    log_callback(f"⚠️ Error for card {card_id}: {e}")
        
        self.conn.commit()
        log_callback(f"✅ Updated {updated} pHash values")
  
    def migrate_database(self):
        """Esegue migrazioni del database se necessario."""
        try:
            # Controlla se la colonna phash esiste
            self.cursor.execute("PRAGMA table_info(cards)")
            columns = [col[1] for col in self.cursor.fetchall()]
            
            if 'phash' not in columns:
                self.log_callback("🔧 Aggiunta colonna 'phash' alla tabella cards...")
                self.cursor.execute("ALTER TABLE cards ADD COLUMN phash TEXT")
                self.conn.commit()
                self.log_callback("✅ Migrazione completata")
        except Exception as e:
            self.log_callback(f"⚠️ Errore migrazione: {e}")
    
    def close(self):
        """Chiude la connessione al database."""
        if self.conn:
            self.conn.close()

# =========================================================================
# 🔍 CARD SCANNER CON OPENCV TEMPLATE MATCHING
# =========================================================================

# Layout Crop Boxes per diversi tipi di pacchetto
LAYOUT_CROP_BOXES = {
    4: [
        (46, 18, 108, 55),    # Top left
        (131, 18, 193, 55),   # Top right
        (46, 133, 108, 170),  # Bottom left
        (130, 132, 193, 170)  # Bottom right
    ],
    5: [
        (5, 16, 69, 54),      # Top 1
        (87, 16, 151, 54),    # Top 2
        (170, 16, 234, 54),   # Top 3
        (46, 133, 108, 170),  # Bottom 1
        (130, 132, 193, 170)  # Bottom 2
    ],
    6: [
        (5, 16, 69, 54),      # Top 1
        (87, 16, 151, 54),    # Top 2
        (170, 16, 234, 54),   # Top 3
        (5, 133, 69, 171),    # Bottom 1
        (87, 133, 151, 171),  # Bottom 2
        (170, 133, 234, 171)  # Bottom 3
    ]
}

# pHash distance threshold
HASH_DISTANCE_THRESHOLD = 15

class CardScanner:
    """Scanner di carte che utilizza Layout Detection + pHash."""
    
    def __init__(self, db_filename=DB_FILENAME, log_callback=None):
        self.db_filename = db_filename
        self.log_callback = log_callback or print
        self.template_hashes = []
        self.db_lock = Lock()
        
        # Carica gli hash dei template dal database
        self.load_template_hashes()
    
    def load_template_hashes(self):
        """Carica tutti gli hash dei template dal database."""
        try:
            with sqlite3.connect(self.db_filename) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, card_name, card_number, set_code, local_image_path, rarity, phash
                    FROM cards
                    WHERE rarity IN (?, ?, ?, ?) AND phash IS NOT NULL
                """, TARGET_RARITIES)
                results = cursor.fetchall()
                
                self.template_hashes = []
                for row in results:
                    card_id, card_name, card_number, set_code, local_image_path, rarity, phash_str = row
                    if phash_str:
                        try:
                            self.template_hashes.append({
                                'card_id': card_id,
                                'card_name': card_name,
                                'card_number': card_number,
                                'set_code': set_code,
                                'local_image_path': local_image_path,
                                'rarity': rarity,
                                'hash': imagehash.hex_to_hash(phash_str)
                            })
                        except:
                            pass
                
                self.log_callback(f"✅ Caricati {len(self.template_hashes)} template hash dal database")
        except Exception as e:
            self.log_callback(f"❌ Errore caricamento template hash: {e}")
    
    def detect_layout(self, image_path):
        """
        Rileva il layout del pacchetto (4, 5 o 6 carte).
        Usa detect_layout.py logic.
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # Check Top Row
            x1, y1, x2, y2 = TOP_ROW_CHECK_BOX
            top_roi = img[y1:y2, x1:x2]
            avg_color_top = np.mean(top_roi, axis=(0, 1))
            top_is_gray = is_color_in_range(avg_color_top, TARGET_GRAY_BGR, COLOR_TOLERANCE)
            cards_in_top_row = 2 if top_is_gray else 3
            
            # Check Bottom Row
            x1, y1, x2, y2 = BOTTOM_ROW_CHECK_BOX
            bottom_roi = img[y1:y2, x1:x2]
            avg_color_bottom = np.mean(bottom_roi, axis=(0, 1))
            bottom_is_gray = is_color_in_range(avg_color_bottom, TARGET_GRAY_BGR, COLOR_TOLERANCE)
            cards_in_bottom_row = 2 if bottom_is_gray else 3
            
            layout = cards_in_top_row + cards_in_bottom_row
            
            if layout in LAYOUT_CROP_BOXES:
                return layout
            else:
                self.log_callback(f"⚠️ Layout rilevato ({layout}) non valido, uso default 4")
                return 4
        except Exception as e:
            self.log_callback(f"⚠️ Errore rilevamento layout: {e}, uso default 4")
            return 4
    
    def hash_card_from_source(self, source_img_pil, crop_box):
        """Calcola l'hash di una carta da un'immagine sorgente."""
        try:
            card_img = source_img_pil.crop(crop_box)
            card_hash = imagehash.phash(card_img)
            return card_hash
        except Exception as e:
            self.log_callback(f"⚠️ Errore calcolo hash: {e}")
            return None
    
    def find_best_match(self, card_hash):
        """Trova il miglior match per un hash dato."""
        if card_hash is None:
            return None
        
        best_match = None
        best_distance = float('inf')
        
        for template_data in self.template_hashes:
            template_hash = template_data['hash']
            distance = card_hash - template_hash  # Hamming distance
            
            if distance < best_distance and distance <= HASH_DISTANCE_THRESHOLD:
                best_distance = distance
                best_match = template_data.copy()
                best_match['distance'] = distance
        
        return best_match
    
    def scan_image(self, image_path, account_name):
        """Scansiona un'immagine e trova le carte corrispondenti usando Layout Detection + pHash."""
        try:
            # 1. Rileva il layout
            layout = self.detect_layout(image_path)
            
            if layout not in LAYOUT_CROP_BOXES:
                self.log_callback(f"⚠️ Layout {layout} non supportato per {image_path}")
                return {}
            
            self.log_callback(f"🎴 Rilevato layout: {layout} carte - {os.path.basename(image_path)}")
            
            # 2. Carica l'immagine con PIL per pHash
            source_img_pil = Image.open(image_path)
            
            # 3. Ottieni i crop boxes per questo layout
            crop_boxes = LAYOUT_CROP_BOXES[layout]
            
            # 4. Processa ogni carta
            results = {}
            
            for idx, crop_box in enumerate(crop_boxes, 1):
                # Calcola hash della carta
                card_hash = self.hash_card_from_source(source_img_pil, crop_box)
                
                if card_hash is None:
                    continue
                
                # Trova il miglior match
                best_match = self.find_best_match(card_hash)
                
                if best_match:
                    card_id = best_match['card_id']
                    
                    # Aggiungi o incrementa il contatore
                    if card_id not in results:
                        results[card_id] = {
                            'card_name': best_match['card_name'],
                            'set_code': best_match['set_code'],
                            'card_number': best_match['card_number'],
                            'local_image_path': best_match['local_image_path'],
                            'rarity': best_match['rarity'],
                            'count': 0,
                            'best_distance': 999
                        }
                    
                    results[card_id]['count'] += 1
                    results[card_id]['best_distance'] = min(
                        results[card_id]['best_distance'], 
                        best_match['distance']
                    )
                    
                    self.log_callback(
                        f"   Posizione {idx}: {best_match['card_name']} "
                        f"({best_match['set_code']}_{best_match['card_number']}) "
                        f"- Distance: {best_match['distance']}"
                    )
                else:
                    self.log_callback(f"   Posizione {idx}: Nessun match trovato")
            
            # 5. Salva nel database
            if results:
                self.update_account_inventory_batch(account_name, results, image_path)
            
            return results
            
        except Exception as e:
            self.log_callback(f"❌ Errore scansione immagine: {e}")
            import traceback
            self.log_callback(traceback.format_exc())
            return {}
    
    def update_account_inventory_batch(self, account_name, cards_dict, source_image_path):
        """Aggiorna l'inventario di un account con un batch di carte."""
        try:
            with sqlite3.connect(self.db_filename, timeout=30.0) as conn:
                cursor = conn.cursor()
                
                # Ottieni o crea l'account
                cursor.execute("""
                    INSERT OR IGNORE INTO accounts (account_name)
                    VALUES (?)
                """, (account_name,))
                
                cursor.execute("""
                    SELECT account_id FROM accounts WHERE account_name = ?
                """, (account_name,))
                account_id = cursor.fetchone()[0]
                
                # Aggiorna l'inventario per ogni carta
                current_time = datetime.now().isoformat()
                for card_id, card_data in cards_dict.items():
                    cursor.execute("""
                        INSERT INTO account_inventory (account_id, card_id, quantity, last_updated)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(account_id, card_id) DO UPDATE SET
                            quantity = quantity + ?,
                            last_updated = ?
                    """, (account_id, card_id, card_data['count'], current_time,
                          card_data['count'], current_time))
                    
                    # Salva anche in found_cards per tracking
                    cursor.execute("""
                        INSERT INTO found_cards (card_id, account_name, source_image_path, match_timestamp, similarity_score)
                        VALUES (?, ?, ?, ?, ?)
                    """, (card_id, account_name, source_image_path, current_time, 
                          float(HASH_DISTANCE_THRESHOLD - card_data.get('best_distance', 0))))
                
                conn.commit()
                return True
        except Exception as e:
            self.log_callback(f"❌ Errore aggiornamento inventario: {e}")
            return False

# =========================================================================
# 🌐 SCRAPER TCG POCKET (VERSIONE OTTIMIZZATA)
# =========================================================================

class TCGPocketScraper:
    """Scraper per TCG Pocket con proxy rotanti e storage SQLite - OTTIMIZZATO."""
    
    def __init__(self, base_url="https://pocket.limitlesstcg.com", proxy_file="proxies.txt", log_callback=None):
        self.base_url = base_url
        self.proxy_file = proxy_file
        self.log_callback = log_callback or print
        self.db_manager = DatabaseManager(log_callback=log_callback)
        self.db_lock = Lock()
        
        # Fetch e carica proxy
        self.fetch_and_save_proxies(PROXY_API_URL)
        self.proxies = self.load_proxies(self.proxy_file)
        self.use_proxy = len(self.proxies) > 1  # Se abbiamo solo [None], non usare proxy
        
        # Setup database
        try:
            self.db_manager.setup_database()
            os.makedirs(TCG_IMAGES_DIR, exist_ok=True)
            proxy_msg = f"✅ Caricati {len(self.proxies)} proxy" if self.use_proxy else "⚠️ Scraping senza proxy"
            self.log_callback(proxy_msg)
        except Exception as e:
            self.log_callback(f"⚠️ Errore setup database: {e}")
    
    def fetch_and_save_proxies(self, api_url):
        """Scarica i proxy dall'API e li salva nel file."""
        self.log_callback(f"📥 Download proxy da API...")
        try:
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            proxy_details_list = data.get('proxies', [])
            ip_port_list = []
            
            for proxy_detail in proxy_details_list:
                if 'ip' in proxy_detail and 'port' in proxy_detail:
                    ip_port = f"{proxy_detail['ip']}:{proxy_detail['port']}"
                    ip_port_list.append(ip_port)
            
            if ip_port_list:
                with open(self.proxy_file, 'w') as f:
                    for ip_port in ip_port_list:
                        f.write(f"{ip_port}\n")
                self.log_callback(f"✅ Salvati {len(ip_port_list)} proxy in {self.proxy_file}")
            else:
                self.log_callback("⚠️ Nessun proxy trovato nell'API")
        except Exception as e:
            self.log_callback(f"⚠️ Errore download proxy: {e}")
    
    def load_proxies(self, proxy_file):
        """Carica i proxy dal file."""
        proxies = []
        try:
            with open(proxy_file, 'r') as f:
                for line in f:
                    proxy = line.strip()
                    if proxy:
                        proxies.append({'http': f'http://{proxy}'})
            return proxies if proxies else [None]
        except:
            return [None]
    
    def get_random_proxy(self):
        """Restituisce un proxy casuale dalla lista."""
        return random.choice(self.proxies) if self.use_proxy else None
    
    def make_request(self, url, max_retries=2, timeout=10):
        """Esegue una richiesta HTTP con proxy casuale e retry VELOCE."""
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        for attempt in range(max_retries):
            try:
                proxy = self.get_random_proxy()
                response = requests.get(url, headers=headers, proxies=proxy, timeout=timeout)
                response.raise_for_status()
                return response
            except Exception as e:
                if attempt == max_retries - 1:
                    # Ultimo tentativo senza proxy
                    try:
                        response = requests.get(url, headers=headers, proxies=None, timeout=timeout)
                        response.raise_for_status()
                        return response
                    except:
                        raise Exception(f"Failed after {max_retries} retries: {e}")
                time.sleep(0.5)
        return None
    
    def get_sets(self):
        """Recupera la lista di tutti i set dalla pagina principale."""
        url = f"{self.base_url}/cards/"
        try:
            response = self.make_request(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            sets = []
            
            rows = soup.find_all('tr')
            for row in rows:
                link = row.find('a', href=re.compile(r'/cards/[A-Za-z0-9-]+$'))
                if link:
                    set_url = link.get('href')
                    set_code = set_url.split('/')[-1]
                    
                    cover_img = link.find('img', class_='set')
                    cover_image_url = cover_img.get('src') if cover_img else None
                    
                    set_name = ""
                    for content in link.children:
                        if isinstance(content, str):
                            text = content.strip()
                            if text:
                                set_name += text + " "
                    
                    set_name = set_name.strip()
                    if not set_name:
                        set_name = link.get_text(strip=True)
                    
                    code_span = link.find('span', class_='code')
                    if code_span:
                        set_name = set_name.replace(code_span.get_text(strip=True), '').strip()
                    
                    cols = row.find_all('td')
                    release_date = cols[1].get_text(strip=True) if len(cols) > 1 else ""
                    total_cards = cols[2].get_text(strip=True) if len(cols) > 2 else "0"
                    
                    sets.append({
                        'code': set_code,
                        'name': set_name,
                        'release_date': release_date,
                        'total_cards': int(total_cards) if total_cards.isdigit() else 0,
                        'url': urljoin(self.base_url, set_url),
                        'cover_image_url': cover_image_url
                    })
            
            return sets
        except Exception as e:
            self.log_callback(f"❌ Errore recupero set: {e}")
            return []
        
    def download_set_cover(self, cover_url, set_folder):
        """Scarica, ingrandisce (2x) e ottimizza la cover di un set - MANTIENE TRASPARENZA."""
        if not cover_url:
            return None
        
        try:
            # Path locale per la cover
            cover_path = os.path.join(set_folder, 'cover.png')
            
            # Se esiste già, ritorna il path
            if os.path.exists(cover_path):
                return cover_path
            
            # Download
            proxy = self.get_random_proxy()
            response = requests.get(cover_url, proxies=proxy, timeout=15)
            response.raise_for_status()
            
            from PIL import Image, ImageEnhance
            import io
            import cv2
            import numpy as np
            
            img = Image.open(io.BytesIO(response.content))
            
            # ========== ENHANCEMENT ==========
            
            # 1. Mantieni trasparenza se presente (NON converti a RGB!)
            has_alpha = img.mode in ('RGBA', 'LA', 'P')
            
            if has_alpha:
                # Separa alpha channel
                if img.mode == 'P':
                    img = img.convert('RGBA')
                
                alpha = img.split()[-1] if img.mode == 'RGBA' else None
                img_rgb = img.convert('RGB') if img.mode == 'RGBA' else img
            else:
                alpha = None
                img_rgb = img
            
            # 2. Upscaling 2x
            original_size = img_rgb.size
            upscaled_size = (original_size[0] * 2, original_size[1] * 2)
            
            img_rgb = img_rgb.resize(upscaled_size, Image.Resampling.LANCZOS)
            
            # Se c'è alpha, scalalo anche lui
            if alpha:
                alpha = alpha.resize(upscaled_size, Image.Resampling.LANCZOS)
            
            self.log_callback(f"📏 Upscaled: {original_size} → {upscaled_size}")
            
            # 3. Enhance con Pillow
            enhancer = ImageEnhance.Contrast(img_rgb)
            img_rgb = enhancer.enhance(1.25)
            
            enhancer = ImageEnhance.Sharpness(img_rgb)
            img_rgb = enhancer.enhance(1.6)
            
            enhancer = ImageEnhance.Brightness(img_rgb)
            img_rgb = enhancer.enhance(1.05)
            
            enhancer = ImageEnhance.Color(img_rgb)
            img_rgb = enhancer.enhance(1.1)
            
            # 4. Converti a OpenCV per denoise
            img_cv = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
            
            # Bilateral filter
            img_cv = cv2.bilateralFilter(img_cv, 9, 75, 75)
            
            # Denoise
            img_cv = cv2.fastNlMeansDenoisingColored(
                img_cv,
                None,
                h=6,
                templateWindowSize=7,
                searchWindowSize=21
            )
            
            # Unsharp mask
            gaussian = cv2.GaussianBlur(img_cv, (0, 0), 2)
            img_cv = cv2.addWeighted(img_cv, 1.3, gaussian, -0.3, 0)
            
            # High-pass sharpening
            kernel_highpass = np.array([[-1, -1, -1],
                                        [-1,  9, -1],
                                        [-1, -1, -1]]) / 1
            highpass = cv2.filter2D(img_cv, -1, kernel_highpass)
            img_cv = cv2.addWeighted(img_cv, 0.9, highpass, 0.1, 0)
            
            # 5. Clip valori
            img_cv = np.clip(img_cv, 0, 255).astype(np.uint8)
            
            # 6. Riconverti a Pillow
            img_final = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            
            # 7. Ri-applica alpha se presente
            if alpha:
                img_final.putalpha(alpha)
                save_mode = 'PNG'
            else:
                save_mode = 'PNG'
            
            # 8. Salva con trasparenza
            img_final.save(cover_path, save_mode, optimize=False)
            
            # Info
            file_size = os.path.getsize(cover_path) / 1024
            mode_info = "RGBA" if has_alpha else "RGB"
            self.log_callback(f"✓ Cover ({mode_info}) scaricata, ingrandita (2x) e ottimizzata ({file_size:.1f}KB)")
            return cover_path
            
        except Exception as e:
            self.log_callback(f"❌ Error downloading/enhancing cover: {e}")
            import traceback
            traceback.print_exc()
            return None



    def get_cards_from_set(self, set_code, set_url):
        """Recupera le carte da un set."""
        try:
            response = self.make_request(set_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            cards = []
            
            card_grid = soup.find('div', class_='card-search-grid')
            if not card_grid:
                return cards
            
            card_links = card_grid.find_all('a', href=re.compile(r'/cards/[A-Za-z0-9-]+/\d+'))
            for link in card_links:
                card_url = urljoin(self.base_url, link.get('href'))
                card_number = card_url.split('/')[-1]
                
                img = link.find('img', class_='card')
                if img:
                    image_url = img.get('src')
                    cards.append({
                        'set_code': set_code,
                        'card_number': card_number,
                        'image_url': image_url,
                        'card_url': card_url
                    })
            
            return cards
        except Exception as e:
            self.log_callback(f"❌ Errore carte set {set_code}: {e}")
            return []
    
    def get_card_details(self, card_url):
        """Recupera i dettagli di una carta."""
        try:
            response = self.make_request(card_url, max_retries=2, timeout=8)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            card_name = ""
            name_elem = soup.select_one('span.card-text-name a')
            if name_elem:
                card_name = name_elem.get_text(strip=True)
            
            rarity = ""
            details_div = soup.find('div', class_='prints-current-details')
            if details_div:
                spans = details_div.find_all('span')
                for span in spans:
                    text = span.get_text(strip=True)
                    if '·' in text:
                        parts = text.split('·')
                        if len(parts) >= 2:
                            rarity = parts[1].strip()
                            break
            
            return card_name, rarity
        except:
            return "", ""
    
    def download_image(self, image_url, save_path):
        """Scarica un'immagine."""
        try:
            proxy = self.get_random_proxy()
            response = requests.get(image_url, proxies=proxy, timeout=10)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
        except:
            return False
    
    def save_card_to_db(self, card_data):
        """Salva una carta nel database con pHash."""
        with self.db_lock:
            # Calcola pHash se l'immagine esiste
            phash_value = None
            if card_data.get('local_image_path') and os.path.exists(card_data['local_image_path']):
                phash_value = self.calculate_phash(card_data['local_image_path'])
            
            self.db_manager.cursor.execute("""
                INSERT OR REPLACE INTO cards
                (set_code, card_number, card_name, rarity, image_url, local_image_path, card_url, phash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (card_data['set_code'], card_data['card_number'], card_data['card_name'],
                  card_data['rarity'], card_data['image_url'], card_data['local_image_path'],
                  card_data['card_url'], phash_value))
            self.db_manager.conn.commit()

    def save_set_to_db(self, set_data):
        """Salva un set nel database."""
        try:
            with self.db_lock:
                conn = sqlite3.connect(DB_FILENAME, timeout=10.0)
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO sets 
                    (set_code, set_name, release_date, total_cards, url, cover_image_path)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    set_data['code'],
                    set_data['name'],
                    set_data['release_date'],
                    set_data['total_cards'],
                    set_data['url'],
                    set_data.get('cover_path')
                ))
                
                conn.commit()
                conn.close()
                
        except Exception as e:
            self.log_callback(f"Error saving set to DB: {e}")



    def calculate_phash(self, image_path):
        """Calcola il pHash di un'immagine di carta."""
        try:
            img = Image.open(image_path)
            # Crop dell'immagine come da configurazione
            cropped = img.crop(TEMPLATE_CROP_BOX)
            # Ridimensiona
            new_size = (int(cropped.width * TEMPLATE_DOWNSCALE_FACTOR), 
                       int(cropped.height * TEMPLATE_DOWNSCALE_FACTOR))
            resized = cropped.resize(new_size, Image.LANCZOS)
            # Calcola pHash
            phash = imagehash.phash(resized)
            return str(phash)
        except Exception as e:
            return None




    def process_single_card(self, card, set_code, set_folder, download_images):
        """Processa una singola carta - per uso in ThreadPoolExecutor."""
        try:
            card_name, rarity = self.get_card_details(card['card_url'])
            card['card_name'] = card_name
            card['rarity'] = rarity
            
            if download_images and card['image_url']:
                # ❌ VECCHIO (zfill aggiunge padding)
                # filename = f"{set_code}_{card['card_number'].zfill(3)}.webp"
                
                # ✅ NUOVO (senza padding)
                filename = f"{set_code}_{card['card_number']}.webp"
                
                save_path = os.path.join(set_folder, filename)
                card['local_image_path'] = save_path
                
                if not os.path.exists(save_path):
                    self.download_image(card['image_url'], save_path)
            else:
                card['local_image_path'] = None
            
            self.save_card_to_db(card)
            return True
        except Exception as e:
            self.log_callback(f"⚠️ Error card {card.get('card_number', '?')}: {e}")
            return False
    
    def process_set(self, set_data, download_images=True):
        """Processa un intero set CON PARALLELIZZAZIONE e salvataggio."""
        set_code = set_data['code']
        set_folder = os.path.join(TCG_IMAGES_DIR, set_code)
        os.makedirs(set_folder, exist_ok=True)
        
        self.log_callback(f"📦 Processing set: {set_code} - {set_data['name']}")
        cover_path = None
        if set_data.get('cover_image_url'):
            self.log_callback(f"{set_code}: Downloading cover...")
            cover_path = self.download_set_cover(set_data['cover_image_url'], set_folder)
            if cover_path:
                self.log_callback(f"{set_code}: Cover saved ✓")
        
        # Aggiorna set_data con il path della cover
        set_data['cover_path'] = cover_path        
        # ⬇️ SALVA IL SET NEL DATABASE ⬇️
        self.save_set_to_db(set_data)
        
        cards = self.get_cards_from_set(set_code, set_data['url'])
        
        if not cards:
            self.log_callback(f"⚠️ No cards found for set {set_code}")
            return 0
        
        self.log_callback(f"   Found {len(cards)} cards in set {set_code}")
        
        success_count = 0
        
        # Processa le carte in parallelo con ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(self.process_single_card, card, set_code, set_folder, download_images): card 
                for card in cards
            }
            
            for idx, future in enumerate(as_completed(futures), 1):
                try:
                    if future.result():
                        success_count += 1
                    
                    # Log progress ogni 10 carte
                    if idx % 10 == 0:
                        self.log_callback(f"   Progress: {idx}/{len(cards)} cards processed...")
                except Exception as e:
                    self.log_callback(f"⚠️ Error processing card: {e}")
        
        return success_count
    
    def close(self):
        """Chiude la connessione al database."""
        if self.db_manager:
            self.db_manager.close()

# =========================================================================
# 🤖 DISCORD BOT CLIENT
# =========================================================================

class TradeMonitorClient(discord.Client):
    """Client Discord per monitorare i trade e scansionare le carte."""
    
    def __init__(self, *, intents: discord.Intents, log_callback, progress_callback, 
                 trade_callback, status_callback, card_found_callback):
        super().__init__(intents=intents)
        self.log_callback = log_callback
        self.progress_callback = progress_callback
        self.trade_callback = trade_callback
        self.status_callback = status_callback
        self.card_found_callback = card_found_callback
        
        self.trade_log = []
        self.processed_message_ids = set()
        self.initial_scan_done = False
        self.session = None
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
        self.pending_trades = []
        
        # Inizializza il card scanner
        self.card_scanner = CardScanner(log_callback=log_callback)
    
    async def setup_hook(self):
        """Setup del client."""
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
        self.session = aiohttp.ClientSession(connector=connector)
    
    async def close(self):
        """Chiude il client e la sessione."""
        if self.session:
            await self.session.close()
        await super().close()
    
    async def on_ready(self):
        """Chiamato quando il bot è pronto."""
        self.log_callback(f'✅ Connected as: {self.user.name}')
        self.status_callback("Connected")
        
        self.trade_log = load_trade_log_fast()
        self.processed_message_ids = set(t['message_id'] for t in self.trade_log)
        self.log_callback(f"📚 Loaded {len(self.trade_log)} trades from log")
        
        if not self.trade_log:
            self.loop.create_task(self.perform_historical_scan_fast())
        else:
            self.loop.create_task(self.perform_incremental_scan_fast())
    
    async def perform_historical_scan_fast(self):
        """Esegue una scansione storica completa del canale."""
        channel = self.get_channel(int(os.getenv('CHANNEL_ID', '0')))
        if not channel:
            self.log_callback(f"❌ Channel not found")
            self.initial_scan_done = True
            return
        
        self.log_callback(f"🔄 Starting historical scan...")
        all_messages = []
        
        try:
            async for message in channel.history(limit=None, oldest_first=False):
                all_messages.append(message)
        except Exception as e:
            self.log_callback(f"❌ Error fetching messages: {e}")
            self.initial_scan_done = True
            return
        
        if all_messages:
            self.log_callback(f"📨 Found {len(all_messages)} total messages")
            batch = []
            
            for i, message in enumerate(all_messages):
                batch.append(message)
                self.progress_callback(i + 1, len(all_messages))
                
                if len(batch) >= BATCH_SIZE:
                    await self.process_message_batch_fast(batch)
                    batch = []
            
            if batch:
                await self.process_message_batch_fast(batch)
            
            save_trade_log_fast(self.trade_log)
            self.log_callback(f"✅ Processed {len(all_messages)} messages | Total trades: {len(self.trade_log)}")
        
        self.initial_scan_done = True
        self.status_callback("Monitoring")
        self.log_callback("✅ Real-time monitoring active")
    
    async def perform_incremental_scan_fast(self):
        """Esegue una scansione incrementale per nuovi messaggi."""
        channel = self.get_channel(int(os.getenv('CHANNEL_ID', '0')))
        if not channel:
            self.log_callback(f"❌ Channel not found")
            self.initial_scan_done = True
            return
        
        if self.trade_log:
            max_msg_id = max(t['message_id'] for t in self.trade_log)
            self.log_callback(f"🔄 Checking for new messages after ID: {max_msg_id}")
            
            new_messages = []
            try:
                async for message in channel.history(limit=None, after=discord.Object(id=max_msg_id), oldest_first=False):
                    new_messages.append(message)
            except Exception as e:
                self.log_callback(f"❌ Error fetching messages: {e}")
                self.initial_scan_done = True
                return
            
            if new_messages:
                self.log_callback(f"📨 Found {len(new_messages)} new messages")
                batch = []
                
                for i, message in enumerate(new_messages):
                    batch.append(message)
                    self.progress_callback(i + 1, len(new_messages))
                    
                    if len(batch) >= BATCH_SIZE:
                        await self.process_message_batch_fast(batch)
                        batch = []
                
                if batch:
                    await self.process_message_batch_fast(batch)
                
                save_trade_log_fast(self.trade_log)
                self.log_callback(f"✅ Processed {len(new_messages)} new messages | Total: {len(self.trade_log)}")
            else:
                self.log_callback("✅ No new messages")
        
        self.initial_scan_done = True
        self.status_callback("Monitoring")
        self.log_callback("✅ Real-time monitoring active")
    
    async def download_with_semaphore(self, trade_data, attachment, folder, filename, file_type):
        """Scarica un allegato con semaforo per limitare i download concorrenti."""
        async with self.semaphore:
            path, status = await download_attachment_fast(self.session, attachment, folder, filename)
            if file_type == 'xml':
                trade_data['xml_path'] = path
            else:
                trade_data['image_path'] = path
    
    async def process_message_batch_fast(self, messages: List) -> int:
        """Processa un batch di messaggi."""
        download_tasks = []
        new_trades = []
        
        for message in messages:
            if SEARCH_STRING in message.content and message.id not in self.processed_message_ids:
                trade_data, xml_att, img_att = extract_trade_data_fast(message)
                account_folder = os.path.join(ACCOUNTS_DIR, trade_data['account_name'])
                
                trade_data['xml_path'] = None
                trade_data['image_path'] = None
                
                if xml_att:
                    xml_target = trade_data['xml_filename_text'] if trade_data['xml_filename_text'] != "N/A" else xml_att.filename
                    download_tasks.append(self.download_with_semaphore(trade_data, xml_att, account_folder, xml_target, 'xml'))
                
                if img_att:
                    image_folder = os.path.join(account_folder, "images")
                    image_ext = os.path.splitext(img_att.filename)[1]
                    image_target = f"{message.id}{image_ext}"
                    download_tasks.append(self.download_with_semaphore(trade_data, img_att, image_folder, image_target, 'image'))
                
                new_trades.append((trade_data, message.id))
        
        if download_tasks:
            await asyncio.gather(*download_tasks, return_exceptions=True)
        
        for trade_data, msg_id in new_trades:
            self.trade_log.append(trade_data)
            self.processed_message_ids.add(msg_id)
            self.pending_trades.append(trade_data)
            self.trade_callback(trade_data)
            
            # Scansiona l'immagine per riconoscere le carte
            if trade_data.get('image_path') and os.path.exists(trade_data['image_path']):
                await self.scan_image_for_cards(trade_data)
        
        if len(self.pending_trades) >= SAVE_INTERVAL:
            save_trade_log_fast(self.trade_log)
            self.pending_trades.clear()
        
        return len(new_trades)
    
    async def scan_image_for_cards(self, trade_data):
        """Scansiona un'immagine per riconoscere le carte."""
        try:
            image_path = trade_data.get('image_path')
            account_name = trade_data.get('account_name')
            
            if not image_path or not account_name:
                return
            
            if not os.path.exists(image_path):
                return
            
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                self.card_scanner.scan_image,
                image_path,
                account_name
            )
            
            if results:
                total_cards = sum(card_data['count'] for card_data in results.values())
                self.log_callback(f"🎴 Trovate {len(results)} carte diverse ({total_cards} copie) - Account: {account_name}")
                
                for card_id, card_data in results.items():
                    self.card_found_callback({
                        'account_name': account_name,
                        'card_name': card_data['card_name'],
                        'set_code': card_data['set_code'],
                        'card_number': card_data['card_number'],
                        'rarity': card_data['rarity'],
                        'count': card_data['count'],
                        'image_path': image_path,  # ⬅️ Screenshot del pacchetto
                        'local_image_path': card_data.get('local_image_path')  # Immagine della carta
                    })
        except Exception as e:
            self.log_callback(f"❌ Errore scansione: {e}")
    
    async def on_message(self, message):
        """Chiamato quando viene ricevuto un nuovo messaggio."""
        if not self.initial_scan_done:
            return
        
        if SEARCH_STRING in message.content and message.id not in self.processed_message_ids:
            self.log_callback(f"📨 New trade detected: {message.id}")
            await self.process_message_batch_fast([message])
            save_trade_log_fast(self.trade_log)

# =========================================================================
# 🧵 THREAD PER LO SCRAPER DEL DATABASE
# =========================================================================

class ScraperThread(QThread):
    """Thread per eseguire lo scraper del database."""
    
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)
    finished_signal = pyqtSignal(bool)
    
    def __init__(self, download_images=True):
        super().__init__()
        self.download_images = download_images
        self.scraper = None
    
    def run(self):
        """Esegue lo scraper."""
        try:
            self.scraper = TCGPocketScraper(log_callback=self.log_signal.emit)
            
            self.log_signal.emit("✅ Recupero lista set...")
            sets = self.scraper.get_sets()
            
            if not sets:
                self.log_signal.emit("⚠️ Nessun set trovato!")
                self.finished_signal.emit(False)
                return
            
            self.log_signal.emit(f"✅ Trovati {len(sets)} set")
            total_sets = len(sets)
            
            for idx, set_data in enumerate(sets):
                self.progress_signal.emit(idx + 1, total_sets)
                cards_count = self.scraper.process_set(set_data, self.download_images)
                self.log_signal.emit(f"✅ Set {set_data['code']}: {cards_count} carte")
            
            self.log_signal.emit("✅ Scraping completato!")
            self.finished_signal.emit(True)
        
        except Exception as e:
            self.log_signal.emit(f"⚠️ Errore: {str(e)}")
            self.finished_signal.emit(False)
        
        finally:
            if self.scraper:
                try:
                    self.scraper.close()
                except Exception as e:
                    self.log_signal.emit(f"⚠️ Errore chiusura: {e}")

# =========================================================================
# 🧵 THREAD PER IL DISCORD BOT
# =========================================================================

class DiscordBotThread(QThread):
    """Thread per eseguire il Discord bot."""
    
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)
    trade_signal = pyqtSignal(dict)
    status_signal = pyqtSignal(str)
    card_found_signal = pyqtSignal(dict)
    
    def __init__(self, token, channel_id):
        super().__init__()
        self.token = token
        self.channel_id = channel_id
        self.client = None
        self.loop = None
    
    def run(self):
        """Avvia il bot Discord."""
        try:
            os.environ['CHANNEL_ID'] = str(self.channel_id)
            
            intents = discord.Intents.default()
            intents.message_content = True
            intents.messages = True
            intents.guilds = True
            
            self.client = TradeMonitorClient(
                intents=intents,
                log_callback=self.log_signal.emit,
                progress_callback=self.progress_signal.emit,
                trade_callback=self.trade_signal.emit,
                status_callback=self.status_signal.emit,
                card_found_callback=self.card_found_signal.emit
            )
            
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.client.start(self.token))
        
        except Exception as e:
            self.log_signal.emit(f"❌ Bot error: {str(e)}")
    
    def stop(self):
        """Ferma il bot Discord."""
        if self.client and self.loop:
            asyncio.run_coroutine_threadsafe(self.client.close(), self.loop)

# =========================================================================
# 🎴 CARD WIDGET - Widget per mostrare una carta nella collezione
# =========================================================================

class CardWidget(QWidget):
    """Widget personalizzato per mostrare una carta con miniatura, conteggio e wishlist."""
    
    clicked = pyqtSignal(str)
    wishlist_changed = pyqtSignal(int, bool)  # card_id, is_wishlisted
    
    def __init__(self, card_data, quantity=0, is_wishlisted=False, parent=None):
        super().__init__(parent)
        self.card_data = card_data
        self.quantity = quantity
        self.card_id = card_data['id']
        self.is_wishlisted = is_wishlisted
        
        # Layout principale
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(2)
        
        # Container per immagine + cuoricino
        image_container = QWidget()
        image_container.setFixedSize(120, 168)
        image_layout = QVBoxLayout(image_container)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(0)
        
        # Label per l'immagine (caricamento lazy)
        self.image_label = QLabel()
        self.image_label.setFixedSize(120, 168)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("QLabel { border: 1px solid #555; background-color: #2a2a2a; }")
        self.image_loaded = False  # Flag per lazy loading
        
        # Placeholder iniziale
        self.image_label.setText("🎴")
        
        image_layout.addWidget(self.image_label)
        
        # Cuoricino wishlist (sopra l'immagine, angolo in alto a sinistra)
        self.wishlist_btn = QPushButton()
        self.wishlist_btn.setFixedSize(28, 28)
        self.wishlist_btn.setCheckable(True)
        self.wishlist_btn.setChecked(is_wishlisted)
        self.update_wishlist_style()
        self.wishlist_btn.clicked.connect(self.toggle_wishlist)
        self.wishlist_btn.setParent(image_container)
        self.wishlist_btn.move(2, 2)  # Angolo in alto a sinistra
        
        main_layout.addWidget(image_container)
        
        # Label per il numero della carta (larghezza fissa uguale all'immagine)
        card_num_label = QLabel(f"#{card_data['card_number']}")
        card_num_label.setFixedWidth(120)
        card_num_label.setAlignment(Qt.AlignCenter)
        card_num_label.setStyleSheet("QLabel { font-size: 9px; color: #888; }")
        main_layout.addWidget(card_num_label)
        
        # Abilita click
        self.setCursor(Qt.PointingHandCursor)
    
    def showEvent(self, event):
        """Carica l'immagine solo quando il widget diventa visibile (LAZY LOADING)."""
        super().showEvent(event)
        if not self.image_loaded:
            self.load_image()
            self.image_loaded = True
    
    def update_wishlist_style(self):
        """Aggiorna lo stile del pulsante wishlist."""
        if self.is_wishlisted:
            self.wishlist_btn.setStyleSheet("""
                QPushButton {
                    background-color: rgba(231, 76, 60, 200);
                    border: 2px solid #c0392b;
                    border-radius: 14px;
                    font-size: 16px;
                }
                QPushButton:hover {
                    background-color: rgba(231, 76, 60, 255);
                }
            """)
            self.wishlist_btn.setText("❤️")
        else:
            self.wishlist_btn.setStyleSheet("""
                QPushButton {
                    background-color: rgba(0, 0, 0, 150);
                    border: 2px solid #555;
                    border-radius: 14px;
                    font-size: 16px;
                }
                QPushButton:hover {
                    background-color: rgba(231, 76, 60, 150);
                }
            """)
            self.wishlist_btn.setText("🤍")
    
    def toggle_wishlist(self):
        """Toggle dello stato wishlist."""
        self.is_wishlisted = not self.is_wishlisted
        self.update_wishlist_style()
        self.wishlist_changed.emit(self.card_id, self.is_wishlisted)
    
    def load_image(self):
        """Carica l'immagine della carta."""
        image_path = self.card_data.get('local_image_path')
        
        if image_path and os.path.exists(image_path):
            try:
                pixmap = QPixmap(image_path)
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(120, 168, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    
                    # Se non posseduta (quantity = 0), applica effetto grigio
                    if self.quantity == 0:
                        image = scaled_pixmap.toImage()
                        for y in range(image.height()):
                            for x in range(image.width()):
                                pixel = image.pixel(x, y)
                                gray = qGray(pixel)
                                gray = int(gray * 0.5)
                                image.setPixel(x, y, qRgb(gray, gray, gray))
                        scaled_pixmap = QPixmap.fromImage(image)
                    
                    self.image_label.setPixmap(scaled_pixmap)
                    
                    # Aggiungi badge con il numero di copie se > 0
                    if self.quantity > 0:
                        self.add_quantity_badge()
                else:
                    self.image_label.setText("❌")
            except Exception as e:
                self.image_label.setText("❌")
                print(f"Error loading image {image_path}: {e}")
        else:
            self.image_label.setText("🎴")
    
    def add_quantity_badge(self):
        """Aggiunge un badge con il numero di copie possedute."""
        current_pixmap = self.image_label.pixmap()
        if current_pixmap and not current_pixmap.isNull():
            pixmap_copy = current_pixmap.copy()
            
            painter = QPainter(pixmap_copy)
            
            # Badge in basso a destra
            badge_size = 25
            x = pixmap_copy.width() - badge_size - 3
            y = pixmap_copy.height() - badge_size - 3
            
            # Sfondo nero semi-trasparente
            painter.setBrush(QBrush(QColor(0, 0, 0, 250)))
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawEllipse(x, y, badge_size, badge_size)
            
            # Testo con il numero
            painter.setFont(QFont("Arial", 10, QFont.Bold))
            painter.setPen(QColor(255, 255, 255))
            text = str(self.quantity) if self.quantity < 100 else "99+"
            painter.drawText(x, y, badge_size, badge_size, Qt.AlignCenter, text)
            
            painter.end()
            self.image_label.setPixmap(pixmap_copy)
    
    def mousePressEvent(self, event):
        """Gestisce il click sulla carta."""
        if event.button() == Qt.LeftButton:
            # Ignora se il click è sul pulsante wishlist
            if self.wishlist_btn.geometry().contains(event.pos()):
                return
            
            # ⬇️ USA IL NUOVO DIALOG INVECE DI ImageViewerDialog
            dialog = CardDetailsDialog(self.card_data, self)
            dialog.exec_()


# =========================================================================
# 📊 CARD DETAILS DIALOG - Mostra dettagli carta e ownership
# =========================================================================

class CardDetailsDialog(QDialog):
    """Dialog per mostrare i dettagli completi di una carta."""
    
    def __init__(self, card_data, parent=None):
        super().__init__(parent)
        self.card_data = card_data
        self.card_id = card_data['id']
        
        self.setWindowTitle(f"{card_data['card_name']} - Card Details")
        self.setModal(True)
        
        # Dimensioni finestra
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(100, 100, min(900, screen.width() - 200), min(700, screen.height() - 100))
        
        layout = QHBoxLayout(self)
        
        # ===== PANNELLO SINISTRO: Immagine =====
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Immagine della carta
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(300, 420)
        self.load_card_image()
        left_layout.addWidget(self.image_label)
        
        # Info carta
        info_text = f"""
        <b>Card Name:</b> {card_data['card_name']}<br>
        <b>Set:</b> {card_data.get('set_code', 'N/A')}<br>
        <b>Number:</b> #{card_data.get('card_number', 'N/A')}<br>
        <b>Rarity:</b> {card_data.get('rarity', 'N/A')}
        """
        info_label = QLabel(info_text)
        info_label.setStyleSheet("QLabel { padding: 10px; background-color: #2a2a2a; border-radius: 5px; }")
        left_layout.addWidget(info_label)
        
        # Pulsante Open Folder
        open_folder_btn = QPushButton("📁 Open Card Image Folder")
        open_folder_btn.clicked.connect(self.open_card_folder)
        open_folder_btn.setStyleSheet("QPushButton { padding: 8px; }")
        left_layout.addWidget(open_folder_btn)
        
        left_layout.addStretch()
        
        layout.addWidget(left_panel)
        
        # ===== PANNELLO DESTRO: Ownership =====
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Titolo
        ownership_title = QLabel("📊 Ownership Information")
        ownership_title.setStyleSheet("QLabel { font-size: 16px; font-weight: bold; padding: 5px; }")
        right_layout.addWidget(ownership_title)
        
        # Tabella degli account
        self.ownership_table = QTableWidget()
        self.ownership_table.setColumnCount(3)
        self.ownership_table.setHorizontalHeaderLabels(["Account", "Copies", "Actions"])
        self.ownership_table.horizontalHeader().setStretchLastSection(False)
        self.ownership_table.setColumnWidth(0, 200)
        self.ownership_table.setColumnWidth(1, 80)
        self.ownership_table.setColumnWidth(2, 150)
        self.ownership_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.ownership_table.setAlternatingRowColors(True)
        
        self.load_ownership_data()
        
        right_layout.addWidget(self.ownership_table)
        
        # Statistiche totali
        total_copies, total_accounts = self.get_total_stats()
        stats_text = f"<b>Total:</b> {total_copies} copies across {total_accounts} account(s)"
        stats_label = QLabel(stats_text)
        stats_label.setStyleSheet("QLabel { padding: 10px; background-color: #2a2a2a; border-radius: 5px; }")
        right_layout.addWidget(stats_label)
        
        # Pulsante Close
        close_btn = QPushButton("✖ Close")
        close_btn.clicked.connect(self.close)
        close_btn.setStyleSheet("QPushButton { padding: 8px; background-color: #e74c3c; color: white; }")
        right_layout.addWidget(close_btn)
        
        layout.addWidget(right_panel)
    
    def load_card_image(self):
        """Carica l'immagine della carta."""
        image_path = self.card_data.get('local_image_path')
        
        if image_path and os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(300, 420, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(scaled_pixmap)
            else:
                self.image_label.setText("❌ Image not found")
        else:
            self.image_label.setText("🎴 No image")
    
    def load_ownership_data(self):
        """Carica i dati di ownership dal database."""
        try:
            with sqlite3.connect(DB_FILENAME) as conn:
                cursor = conn.cursor()
                
                # Recupera tutti gli account che possiedono questa carta
                cursor.execute("""
                    SELECT a.account_name, ai.quantity
                    FROM account_inventory ai
                    JOIN accounts a ON ai.account_id = a.account_id
                    WHERE ai.card_id = ? AND ai.quantity > 0
                    ORDER BY ai.quantity DESC, a.account_name ASC
                """, (self.card_id,))
                
                results = cursor.fetchall()
                
                self.ownership_table.setRowCount(len(results))
                
                for row_idx, (account_name, quantity) in enumerate(results):
                    # Colonna 0: Account name
                    account_item = QTableWidgetItem(account_name)
                    self.ownership_table.setItem(row_idx, 0, account_item)
                    
                    # Colonna 1: Quantity
                    quantity_item = QTableWidgetItem(str(quantity))
                    quantity_item.setTextAlignment(Qt.AlignCenter)
                    self.ownership_table.setItem(row_idx, 1, quantity_item)
                    
                    # Colonna 2: Action button
                    action_widget = QWidget()
                    action_layout = QHBoxLayout(action_widget)
                    action_layout.setContentsMargins(5, 2, 5, 2)
                    
                    xml_btn = QPushButton("📄 Open XML Folder")
                    xml_btn.setStyleSheet("QPushButton { padding: 4px 8px; font-size: 10px; }")
                    xml_btn.clicked.connect(lambda checked, acc=account_name: self.open_xml_folder(acc))
                    action_layout.addWidget(xml_btn)
                    
                    self.ownership_table.setCellWidget(row_idx, 2, action_widget)
        
        except Exception as e:
            print(f"Error loading ownership data: {e}")
            QMessageBox.warning(self, "Error", f"Failed to load ownership data: {str(e)}")
    
    def get_total_stats(self):
        """Calcola le statistiche totali."""
        try:
            with sqlite3.connect(DB_FILENAME) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT SUM(quantity), COUNT(DISTINCT account_id)
                    FROM account_inventory
                    WHERE card_id = ? AND quantity > 0
                """, (self.card_id,))
                
                result = cursor.fetchone()
                total_copies = result[0] if result[0] else 0
                total_accounts = result[1] if result[1] else 0
                
                return total_copies, total_accounts
        except Exception as e:
            print(f"Error getting stats: {e}")
            return 0, 0
    
    def open_card_folder(self):
        """Apre la cartella contenente l'immagine della carta."""
        image_path = self.card_data.get('local_image_path')
        if image_path and os.path.exists(image_path):
            folder = os.path.dirname(os.path.abspath(image_path))
            self.open_folder(folder)
        else:
            QMessageBox.information(self, "Info", "Card image not found")
    
    def open_xml_folder(self, account_name):
        """Apre la cartella XML di un account."""
        try:
            xml_folder = os.path.join(ACCOUNTS_DIR, account_name, "xml")
            if os.path.exists(xml_folder):
                self.open_folder(xml_folder)
            else:
                QMessageBox.information(self, "Info", f"XML folder not found for {account_name}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Cannot open folder: {str(e)}")
    
    def open_folder(self, folder_path):
        """Apre una cartella nel file explorer."""
        try:
            if sys.platform == 'win32':
                os.startfile(folder_path)
            elif sys.platform == 'darwin':
                subprocess.run(['open', folder_path])
            else:
                subprocess.run(['xdg-open', folder_path])
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Cannot open folder: {str(e)}")


# =========================================================================
# 🖼️ IMAGE VIEWER DIALOG (VERSIONE FUNZIONANTE)
# =========================================================================

class ImageViewerDialog(QDialog):
    """Dialog per visualizzare immagini a schermo intero."""
    
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(os.path.basename(image_path))
        self.setModal(True)
        
        # Dimensioni finestra
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(100, 100, min(1200, screen.width() - 200), min(800, screen.height() - 100))
        
        layout = QVBoxLayout(self)
        
        # Label per l'immagine
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        
        # Carica e mostra l'immagine
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            # Scala l'immagine mantenendo l'aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.width() - 40, 
                self.height() - 100, 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
        else:
            self.image_label.setText("❌ Impossibile caricare l'immagine")
        
        layout.addWidget(self.image_label)
        
        # Info label
        info_label = QLabel(f"📁 {os.path.dirname(image_path)}\n📄 {os.path.basename(image_path)}")
        info_label.setStyleSheet("QLabel { color: #888; font-size: 10px; padding: 5px; }")
        layout.addWidget(info_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        open_folder_btn = QPushButton("📁 Open Folder")
        open_folder_btn.clicked.connect(lambda: self.open_folder(image_path))
        open_folder_btn.setStyleSheet("QPushButton { padding: 5px 15px; }")
        button_layout.addWidget(open_folder_btn)
        
        button_layout.addStretch()
        
        close_btn = QPushButton("✖ Close")
        close_btn.clicked.connect(self.close)
        close_btn.setStyleSheet("QPushButton { padding: 5px 15px; background-color: #e74c3c; color: white; }")
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def open_folder(self, file_path):
        """Apre la cartella contenente il file."""
        folder = os.path.dirname(os.path.abspath(file_path))
        try:
            if sys.platform == 'win32':
                os.startfile(folder)
            elif sys.platform == 'darwin':
                subprocess.run(['open', folder])
            else:
                subprocess.run(['xdg-open', folder])
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Cannot open folder: {str(e)}")


# =========================================================================
# 🧵 THREAD PER IL CARICAMENTO DELLA COLLEZIONE
# =========================================================================

class CollectionLoaderThread(QThread):
    """Thread per caricare la collezione in background."""
    
    progress_signal = pyqtSignal(str)  # Messaggi di stato
    set_ready_signal = pyqtSignal(object, str, str, int, str, str, dict)  # (frame, set_code, set_name, total, cover, account, inventory)
    finished_signal = pyqtSignal(int, int)  # (total_owned, total_cards)
    error_signal = pyqtSignal(str)
    
    def __init__(self, db_filename, selected_account):
        super().__init__()
        self.db_filename = db_filename
        self.selected_account = selected_account
        self.sets_data = []
    
    def run(self):
        """Carica i dati della collezione dal database."""
        try:
            with sqlite3.connect(self.db_filename) as conn:
                cursor = conn.cursor()
                
                # Recupera tutti i set
                cursor.execute("""
                    SELECT set_code, set_name, total_cards, cover_image_path
                    FROM sets
                    ORDER BY release_date DESC
                """)
                sets = cursor.fetchall()
                
                if not sets:
                    self.error_signal.emit("No sets found in database")
                    return
                
                total_owned = 0
                total_cards = 0
                
                for set_code, set_name, set_total_cards, cover_path in sets:
                    self.progress_signal.emit(f"Loading {set_code}...")
                    
                    # Calcola statistiche
                    if self.selected_account == "All Accounts":
                        cursor.execute("""
                            SELECT COUNT(DISTINCT c.id)
                            FROM cards c
                            JOIN account_inventory ai ON c.id = ai.card_id
                            WHERE c.set_code = ? AND ai.quantity > 0
                        """, (set_code,))
                    else:
                        cursor.execute("""
                            SELECT COUNT(DISTINCT c.id)
                            FROM cards c
                            JOIN account_inventory ai ON c.id = ai.card_id
                            JOIN accounts a ON ai.account_id = a.account_id
                            WHERE c.set_code = ? AND ai.quantity > 0 AND a.account_name = ?
                        """, (set_code, self.selected_account))
                    
                    result = cursor.fetchone()
                    owned = result[0] if result and result[0] else 0
                    total_owned += owned
                    total_cards += set_total_cards if set_total_cards else 0
                    
                    # Recupera inventario
                    if self.selected_account == "All Accounts":
                        cursor.execute("""
                            SELECT c.id, SUM(ai.quantity)
                            FROM cards c
                            JOIN account_inventory ai ON c.id = ai.card_id
                            WHERE c.set_code = ?
                            GROUP BY c.id
                        """, (set_code,))
                    else:
                        cursor.execute("""
                            SELECT c.id, SUM(ai.quantity)
                            FROM cards c
                            JOIN account_inventory ai ON c.id = ai.card_id
                            JOIN accounts a ON ai.account_id = a.account_id
                            WHERE c.set_code = ? AND a.account_name = ?
                            GROUP BY c.id
                        """, (set_code, self.selected_account))
                    
                    inventory = {row[0]: row[1] for row in cursor.fetchall()}
                    
                    # Invia i dati del set (senza creare i widget qui)
                    self.set_ready_signal.emit(
                        None,  # Frame creato nel thread principale
                        set_code,
                        set_name,
                        set_total_cards if set_total_cards else 0,
                        cover_path,
                        self.selected_account,
                        inventory
                    )
                
                self.finished_signal.emit(total_owned, total_cards)
        
        except Exception as e:
            import traceback
            self.error_signal.emit(f"Error: {str(e)}\n{traceback.format_exc()}")



# =========================================================================
# 🌐 CLOUDFLARE TUNNEL THREAD
# =========================================================================

class CloudflareTunnelThread(QThread):
    """Thread per eseguire Cloudflare Tunnel e ottenere URL pubblico."""
    
    log_signal = pyqtSignal(str)
    url_ready_signal = pyqtSignal(str)  # Quando l'URL è pronto
    stopped_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    
    def __init__(self, local_port=5000):
        super().__init__()
        self.local_port = local_port
        self.process = None
        self.public_url = None
        
    def run(self):
        """Avvia cloudflared tunnel."""
        try:
            # Path di cloudflared.exe
            if getattr(sys, 'frozen', False):
                # Se è in EXE
                cloudflared_path = os.path.join(sys._MEIPASS, 'cloudflared.exe')
            else:
                # Se è in sviluppo
                cloudflared_path = os.path.join(os.getcwd(), 'cloudflared.exe')
            
            if not os.path.exists(cloudflared_path):
                self.error_signal.emit(
                    f"cloudflared.exe not found at: {cloudflared_path}\n"
                    "Download from: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/"
                )
                return
            
            self.log_signal.emit(f"🌐 Starting Cloudflare Tunnel on port {self.local_port}...")
            
            # Esegui cloudflared
            self.process = subprocess.Popen(
                [cloudflared_path, 'tunnel', '--url', f'http://localhost:{self.local_port}'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Leggi output per catturare URL
            for line in self.process.stdout:
                self.log_signal.emit(f"Cloudflare: {line.strip()}")
                
                # Cerca l'URL pubblico (formato: https://xxxxx.trycloudflare.com)
                if 'trycloudflare.com' in line:
                    import re
                    match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
                    if match:
                        self.public_url = match.group(0)
                        self.url_ready_signal.emit(self.public_url)
                        self.log_signal.emit(f"✅ Public URL: {self.public_url}")
            
        except Exception as e:
            import traceback
            self.error_signal.emit(f"Cloudflare Tunnel error: {str(e)}\n{traceback.format_exc()}")
        finally:
            self.stopped_signal.emit()
    
    def stop_tunnel(self):
        """Ferma il tunnel Cloudflare."""
        if self.process:
            self.log_signal.emit("🛑 Stopping Cloudflare Tunnel...")
            self.process.terminate()
            self.process.wait(timeout=5)
            self.process = None
        self.quit()



# =========================================================================
# 🌐 FLASK WEB SERVER THREAD
# =========================================================================

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QLabel, QTextEdit, 
                             QPushButton, QMessageBox)
from PyQt5.QtCore import Qt
from pathlib import Path
import os

class CloudflarePasswordDialog(QDialog):
    """Dialog per configurare password Cloudflare"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.password = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("🔐 Cloudflare Password Setup")
        self.setGeometry(100, 100, 500, 350)
        self.setStyleSheet("background-color: #2b2b2b; color: white;")
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Configura Password per Cloudflare")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Info
        info = QLabel("This password will protect your access when sharing via Cloudflare.\n"
                    "No password is needed locally.")
        info.setStyleSheet("color: #aaa; margin-bottom: 15px;")
        layout.addWidget(info)
        
        # Textarea
        self.password_input = QTextEdit()
        self.password_input.setPlaceholderText("Enter a secure password (minimum 6 characters)...")
        self.password_input.setStyleSheet("""
            QTextEdit {
                background-color: #3a3a3a;
                color: white;
                border: 1px solid #555;
                border-radius: 5px;
                padding: 10px;
                font-size: 12px;
            }
        """)
        self.password_input.setMinimumHeight(120)
        layout.addWidget(QLabel("Password:"))
        layout.addWidget(self.password_input)
        
        # Status (se esiste già)
        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: #888; font-size: 11px; margin-top: 10px;")
        self.check_existing_password()
        layout.addWidget(self.status_label)
        
        # Buttons
        button_layout = QVBoxLayout()
        
        save_btn = QPushButton("💾 Save Password")
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #0d6efd;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #0b5ed7;
            }
        """)
        save_btn.clicked.connect(self.save_password)
        button_layout.addWidget(save_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
        """)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def check_existing_password(self):
        """Controlla se esiste una password nel .env"""
        env_file = Path('.env')
        if env_file.exists():
            try:
                with open(env_file, 'r') as f:
                    for line in f:
                        if line.startswith('CLOUDFLARE_PASSWORD='):
                            self.status_label.setText("✅ A password is already set (you can change it)")
                            return
            except:
                pass
        self.status_label.setText("ℹ️ No password set yet")
    
    def save_password(self):
        """Salva la password nel .env"""
        password = self.password_input.toPlainText().strip()
        
        # Validazione
        if not password:
            QMessageBox.warning(self, "Errore", "Password cannot be empty!")
            return
        
        if len(password) < 6:
            QMessageBox.warning(self, "Errore", "Password must be at least 6 characters long!")
            return
        
        try:
            # Scrivi nel .env
            env_file = Path('.env')
            with open(env_file, 'w') as f:
                f.write(f'CLOUDFLARE_PASSWORD={password}\n')
            
            # Aggiorna la variabile d'ambiente
            os.environ['CLOUDFLARE_PASSWORD'] = password
            
            QMessageBox.information(self, "✅ Success", 
                                   "Password saved successfully!\n\n"
                                   "When you log in through Cloudflare,\n"
                                   "you will be asked for this password.")
            
            self.password = password
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "Errore", f"Error saving: {str(e)}")


class FlaskServerThread(QThread):
    """Thread per eseguire il server Flask in background senza bloccare la GUI."""
    
    log_signal = pyqtSignal(str)
    started_signal = pyqtSignal()
    stopped_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.flask_app = None
        self.server = None
        self.should_stop = False
        
    def run(self):
        """Avvia il server Flask in un thread separato."""
        try:
            from flask import Flask, render_template, jsonify, send_from_directory
            from werkzeug.serving import make_server
            
            # Crea Flask app
            app = Flask(__name__, 
                       template_folder='templates',
                       static_folder='static')

            app.static_folder = '.'
            app.static_url_path = '/static'
            # ✅ IMPORTANTE: Configura la chiave segreta per le sessioni
            app.config['SECRET_KEY'] = secrets.token_hex(32)

            # Configurazioni aggiuntive
            app.config['SESSION_COOKIE_SECURE'] = False  # True se usi HTTPS
            app.config['SESSION_COOKIE_HTTPONLY'] = True
            app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
            app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 ora            
            self.flask_app = app
            
            # ===== ROUTES  ====
            @app.route('/tcg_images/<path:image_path>')
            def serve_tcg_images(image_path):
                """Serve immagini TCG - funziona con EXE."""
                try:
                    import urllib.parse
                    
                    # Decodifica il path
                    decoded_path = urllib.parse.unquote(image_path)
                    
                    # Costruisci path completo
                    full_path = os.path.join(TCG_IMAGES_DIR, decoded_path)
                    
                    # Normalizza il path (rimuovi .. e simili)
                    full_path = os.path.abspath(full_path)
                    base_dir = os.path.abspath(TCG_IMAGES_DIR)
                    
                    # Verifica che il file sia dentro TCG_IMAGES_DIR (security check)
                    if not full_path.startswith(base_dir):
                        return "Access denied", 403
                    
                    print(f"📁 Requested: {decoded_path}")
                    print(f"📁 Full path: {full_path}")
                    print(f"✅ Exists: {os.path.exists(full_path)}")
                    
                    if not os.path.exists(full_path):
                        return "Not found", 404
                    
                    # Determina il tipo MIME
                    mimetype, _ = mimetypes.guess_type(full_path)
                    if mimetype is None:
                        mimetype = 'image/webp'
                    
                    print(f"✅ Sending: {mimetype}")
                    return send_file(full_path, mimetype=mimetype)
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                    return f"Error: {str(e)}", 500
                
            @app.route('/debug/images')
            def debug_images():
                """Debug: mostra il percorso delle immagini."""
                import os
                
                html = f"""
                <h1>Debug Images</h1>
                <p><strong>TCG_IMAGES_DIR:</strong> {TCG_IMAGES_DIR}</p>
                <p><strong>Exists:</strong> {os.path.exists(TCG_IMAGES_DIR)}</p>
                <hr>
                <h2>Available images:</h2>
                <ul>
                """
                
                if os.path.exists(TCG_IMAGES_DIR):
                    for root, dirs, files in os.walk(TCG_IMAGES_DIR):
                        for file in files:
                            if file.endswith(('.webp', '.png', '.jpg')):
                                full_path = os.path.join(root, file)
                                rel_path = os.path.relpath(full_path, TCG_IMAGES_DIR)
                                
                                # Converti backslash a forward slash
                                url_path = rel_path.replace('\\', '/')
                                
                                html += f"""
                                <li>
                                    <strong>{file}</strong><br>
                                    Full: {full_path}<br>
                                    Rel: {rel_path}<br>
                                    URL: <a href="/tcg_images/{url_path}">Test</a>
                                </li>
                                """
                
                html += "</ul>"
                return html

            @app.route('/set/<set_code>')
            def view_set(set_code):
                """Mostra le carte di un set."""
                try:
                    # Recupera il set dal database
                    conn = sqlite3.connect(DB_FILENAME)
                    cursor = conn.cursor()
                    
                    cursor.execute("SELECT set_code, set_name, release_date, total_cards FROM sets WHERE set_code = ?", (set_code,))
                    set_row = cursor.fetchone()
                    
                    if not set_row:
                        return render_template('error.html', error="Set not found"), 404
                    
                    set_code, set_name, release_date, total_cards = set_row
                    
                    # Recupera le carte del set
                    cursor.execute("""
                        SELECT set_code, card_number, card_name, rarity, local_image_path
                        FROM cards 
                        WHERE set_code = ?
                        ORDER BY CAST(card_number AS INTEGER)
                    """, (set_code,))
                    
                    cards = []
                    for row in cursor.fetchall():
                        cards.append({
                            'set_code': row[0],
                            'card_number': row[1],
                            'card_name': row[2],
                            'rarity': row[3],
                            'local_image_path': row[4]
                        })
                    
                    # Recupera il path della cover
                    cursor.execute("SELECT cover_image_path FROM sets WHERE set_code = ?", (set_code,))
                    cover_result = cursor.fetchone()
                    cover_path = cover_result[0] if cover_result else None
                    
                    return render_template('set_view.html',
                        set_code=set_code,
                        set_name=set_name,
                        release_date=release_date,
                        total_cards=total_cards,
                        cover_path=cover_path,
                        cards=cards
                    )
                
                except Exception as e:
                    print(f"❌ Error: {e}")
                    return render_template('error.html', error=str(e)), 500



            @app.route('/account/<account_name>', methods=['GET', 'POST'])
            @require_password
            def account_collection(account_name):
                """Visualizza la collezione completa di un account specifico."""
                try:
                    conn = sqlite3.connect(DB_FILENAME)
                    cursor = conn.cursor()
                    
                    # Verifica che l'account esista
                    cursor.execute("SELECT account_id FROM accounts WHERE account_name = ?", (account_name,))
                    account = cursor.fetchone()
                    
                    if not account:
                        conn.close()
                        return f"<h1>Account '{account_name}' not found</h1><a href='/'>Back to home</a>", 404
                    
                    account_id = account[0]
                    
                    # Get inventory dell'account
                    cursor.execute("""
                        SELECT card_id, quantity 
                        FROM account_inventory 
                        WHERE account_id = ? AND quantity > 0
                    """, (account_id,))
                    
                    inventory = {row[0]: row[1] for row in cursor.fetchall()}
                    
                    # Get statistiche account
                    cursor.execute("""
                        SELECT 
                            COUNT(DISTINCT c.id) as unique_cards,
                            SUM(ai.quantity) as total_copies,
                            COUNT(DISTINCT c.set_code) as sets_owned
                        FROM cards c
                        JOIN account_inventory ai ON c.id = ai.card_id
                        WHERE ai.account_id = ? AND ai.quantity > 0
                    """, (account_id,))
                    
                    stats = cursor.fetchone()
                    
                    # Get collezione per set
                    cursor.execute("""
                        SELECT DISTINCT c.set_code, s.set_name
                        FROM cards c
                        JOIN sets s ON c.set_code = s.set_code
                        JOIN account_inventory ai ON c.id = ai.card_id
                        WHERE ai.account_id = ? AND ai.quantity > 0
                        ORDER BY s.set_code
                    """, (account_id,))
                    
                    sets = cursor.fetchall()
                    
                    # Get carte per ogni set
                    collection_by_set = {}
                    for set_code, set_name in sets:
                        cursor.execute("""
                            SELECT c.id, c.card_number, c.card_name, c.rarity, ai.quantity
                            FROM cards c
                            JOIN account_inventory ai ON c.id = ai.card_id
                            WHERE c.set_code = ? AND ai.account_id = ? AND ai.quantity > 0
                            ORDER BY CAST(c.card_number AS INTEGER)
                        """, (set_code, account_id))
                        
                        cards = cursor.fetchall()
                        collection_by_set[set_code] = {
                            'set_name': set_name,
                            'cards': cards
                        }
                    
                    conn.close()
                    
                    return render_template('account_collection.html',
                                        account_name=account_name,
                                        stats={
                                            'unique_cards': stats[0] or 0,
                                            'total_copies': stats[1] or 0,
                                            'sets_owned': stats[2] or 0
                                        },
                                        collection_by_set=collection_by_set)
                
                except Exception as e:
                    import traceback
                    error_msg = traceback.format_exc()
                    return f"<h1>Error loading collection</h1><pre>{error_msg}</pre><a href='/'>Back to home</a>", 500



            @app.route('/tcg_images/<path:filename>', methods=['GET', 'POST'])
            @require_password
            def serve_card_image(filename):
                return send_from_directory('tcg_images', filename)
            
            @app.route('/', methods=['GET', 'POST'])
            @require_password
            def index():
                """Pagina principale con lista di tutti i set."""
                try:
                    conn = sqlite3.connect(DB_FILENAME)
                    cursor = conn.cursor()
                    
                    # Target rarities
                    TARGET_RARITIES = ('◊◊◊◊', '☆', '☆☆', 'Crown Rare')
                    placeholders = ','.join('?' * len(TARGET_RARITIES))
                    
                    # Get tutti i set con stats CORRETTI
                    cursor.execute(f"""
                        SELECT 
                            s.set_code,
                            s.set_name,
                            s.release_date,
                            COUNT(DISTINCT CASE WHEN c.rarity IN ({placeholders}) THEN c.id END) as target_total,
                            COUNT(DISTINCT CASE WHEN c.rarity IN ({placeholders}) AND ai.quantity > 0 THEN c.id END) as owned_cards,
                            COALESCE(SUM(CASE WHEN c.rarity IN ({placeholders}) THEN ai.quantity END), 0) as total_copies
                        FROM sets s
                        LEFT JOIN cards c ON s.set_code = c.set_code
                        LEFT JOIN account_inventory ai ON c.id = ai.card_id
                        GROUP BY s.set_code
                        ORDER BY s.set_code DESC
                    """, TARGET_RARITIES + TARGET_RARITIES + TARGET_RARITIES)
                    
                    sets_data = cursor.fetchall()
                    
                    # Formatta i dati per il template
                    sets = []
                    for row in sets_data:
                        set_code, set_name, release_date, target_total, owned_cards, total_copies = row
                        
                        # Calcola completion correttamente
                        if target_total > 0:
                            completion = int((owned_cards / target_total) * 100)
                        else:
                            completion = 0
                        
                        sets.append({
                            'code': set_code,
                            'name': set_name,
                            'release_date': release_date or 'N/A',
                            'total_cards': target_total,
                            'owned': owned_cards,
                            'completion': completion,
                            'copies': total_copies or 0
                        })
                    
                    conn.close()
                    
                    return render_template('index.html', sets=sets)
                
                except Exception as e:
                    import traceback
                    print(f"Error in index: {traceback.format_exc()}")
                    return f"<h1>Error</h1><pre>{traceback.format_exc()}</pre>", 500



            
            @app.route('/set/<set_code>', methods=['GET', 'POST'])
            @require_password
            def set_view(set_code):
                """Visualizza tutte le carte di un set specifico con copie."""
                try:
                    conn = sqlite3.connect(DB_FILENAME)
                    cursor = conn.cursor()
                    
                    # Get info del set
                    cursor.execute("""
                        SELECT set_name, release_date, total_cards 
                        FROM sets 
                        WHERE set_code = ?
                    """, (set_code,))
                    
                    set_info = cursor.fetchone()
                    
                    if not set_info:
                        conn.close()
                        return f"<h1>Set '{set_code}' not found</h1><a href='/'>Back</a>", 404
                    
                    set_name, release_date, total_cards = set_info
                    
                    # Target rarities
                    TARGET_RARITIES = ('◊◊◊◊', '☆', '☆☆', 'Crown Rare')
                    
                    # Get tutte le carte del set con copie
                    cursor.execute("""
                        SELECT 
                            c.id, 
                            c.card_number, 
                            c.card_name, 
                            c.rarity,
                            COALESCE(SUM(ai.quantity), 0) as total_copies
                        FROM cards c
                        LEFT JOIN account_inventory ai ON c.id = ai.card_id
                        WHERE c.set_code = ? AND c.rarity IN (?, ?, ?, ?)
                        GROUP BY c.id
                        ORDER BY CAST(c.card_number AS INTEGER)
                    """, (set_code,) + TARGET_RARITIES)
                    
                    cards = cursor.fetchall()
                    
                    conn.close()
                    
                    return render_template('set_view.html',
                                        set_code=set_code,
                                        set_name=set_name,
                                        release_date=release_date or 'N/A',
                                        total_cards=len(cards),  # Conta solo target rarities
                                        cards=cards)
                
                except Exception as e:
                    import traceback
                    return f"<h1>Error</h1><pre>{traceback.format_exc()}</pre><a href='/'>Back</a>", 500
            
            @app.route('/card/<int:card_id>', methods=['GET', 'POST'])
            @require_password
            def card_details(card_id):
                conn = sqlite3.connect(DB_FILENAME)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT c.*, s.set_name
                    FROM cards c
                    JOIN sets s ON c.set_code = s.set_code
                    WHERE c.id = ?
                """, (card_id,))
                card = cursor.fetchone()
                if not card:
                    return "Card not found", 404
                cursor.execute("""
                    SELECT a.account_name, ai.quantity
                    FROM account_inventory ai
                    JOIN accounts a ON ai.account_id = a.account_id
                    WHERE ai.card_id = ? AND ai.quantity > 0
                    ORDER BY ai.quantity DESC
                """, (card_id,))
                owners = cursor.fetchall()
                cursor.execute("""
                    SELECT COALESCE(SUM(quantity), 0)
                    FROM account_inventory WHERE card_id = ?
                """, (card_id,))
                total_copies = cursor.fetchone()[0]
                conn.close()
                return render_template('card_details.html', 
                                      card=card, owners=owners, total_copies=total_copies)
            
            @app.route('/stats', methods=['GET', 'POST'])
            @require_password
            def stats():
                conn = sqlite3.connect(DB_FILENAME)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT a.account_name, 
                           COUNT(DISTINCT ai.card_id) as unique_cards,
                           SUM(ai.quantity) as total_copies
                    FROM accounts a
                    JOIN account_inventory ai ON a.account_id = ai.account_id
                    WHERE ai.quantity > 0
                    GROUP BY a.account_id
                    ORDER BY unique_cards DESC LIMIT 5
                """)
                top_accounts = cursor.fetchall()
                cursor.execute("""
                    SELECT c.card_name, c.set_code, c.rarity,
                           SUM(ai.quantity) as total_copies
                    FROM cards c
                    JOIN account_inventory ai ON c.id = ai.card_id
                    WHERE ai.quantity > 0
                    GROUP BY c.id
                    ORDER BY total_copies DESC LIMIT 5
                """)
                top_cards = cursor.fetchall()
                conn.close()
                return render_template('stats.html',
                                      top_accounts=top_accounts,
                                      top_cards=top_cards)
            
            # ⬇️ USA make_server per poterlo fermare correttamente ⬇️
            self.server = make_server('0.0.0.0', 5000, app, threaded=True)
            self.log_signal.emit("🌐 Flask server started on http://localhost:5000")
            self.started_signal.emit()
            
            # Esegui il server (blocca fino a shutdown)
            self.server.serve_forever()
            
        except OSError as e:
            if "Address already in use" in str(e):
                self.error_signal.emit("Port 5000 already in use. Stop the other server first.")
            else:
                self.error_signal.emit(f"Server error: {str(e)}")
        except Exception as e:
            import traceback
            self.error_signal.emit(f"Flask error: {str(e)}\n{traceback.format_exc()}")
        finally:
            self.log_signal.emit("🌐 Flask server stopped")
            self.stopped_signal.emit()
    
    def stop_server(self):
        """Ferma il server Flask in modo sicuro."""
        if self.server:
            self.log_signal.emit("🌐 Stopping Flask server...")
            self.server.shutdown()  # ⬅️ Shutdown thread-safe
            self.server = None
        self.quit()
        self.wait(3000)  # Aspetta max 3 secondi

# =========================================================================
# 🖥️ GUI APPLICATION - MAIN WINDOW
# =========================================================================



class MainWindow(QMainWindow):
    """Finestra principale dell'applicazione."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TCGP Team Rocket Tool")
        self.setGeometry(100, 100, 1400, 900)
        
        # Imposta l'icona se esiste
        # ⬇️ AGGIUNGI QUESTA PARTE ⬇️
        # Imposta icona della finestra (taskbar + titlebar)
        if os.path.exists(ICON_PATH):
            app_icon = QIcon(ICON_PATH)
            self.setWindowIcon(app_icon)
            
            # Imposta anche l'icona dell'applicazione (per Windows taskbar)
            if sys.platform == 'win32':
                import ctypes
                myappid = 'pcbisgood.tcgpockettracker.teamrocket.1'  # ID arbitrario univoco
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
                QApplication.setWindowIcon(app_icon)
        
        # Variabili
        self.bot_thread = None
        self.scraper_thread = None
        self.found_cards = []
        self.collection_loaded = False
        
        # Setup UI
        self.setup_ui()
        self.set_background_image('gui/background.png')
        # Load settings
        self.load_settings()
        
        # Setup system tray
        self.setup_system_tray()
    
    def setup_ui(self):
        """Configura l'interfaccia utente."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # Tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Tab 1: Discord Bot
        self.setup_bot_tab()
        
        # Tab 2: Cards Found
        self.setup_cards_found_tab()
        
        self.setup_collection_tab()

        # Tab 3: Database Setup
        self.setup_database_tab()
        
        # Tab 4: Statistics
        self.setup_stats_tab()
        
        # Tab 5: Settings
        self.setup_settings_tab()
        self.tabs.currentChanged.connect(self.on_tab_changed)


    def open_cloudflare_dialog(self):
        """Apre il dialog per configurare Cloudflare"""
        dialog = CloudflarePasswordDialog(self)
        if dialog.exec_() == QDialog.Accepted:  # PyQt5 usa exec_()
            password = dialog.password
            if password:
                # Mostra info su come usare Cloudflare
                info_msg = (
                    "✅ Password configured!\n\n"
                    "The public URL will be password protected."
                )
                QMessageBox.information(self, "ℹ️ Cloudflare Setup", info_msg)


    def open_url(self, url):
        """Apre un URL nel browser predefinito."""
        import webbrowser
        try:
            webbrowser.open(url)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open URL: {str(e)}")


    # =========================================================================
    # FLASK WEB SERVER
    # =========================================================================
    
    def toggle_web_server(self):
        """Avvia o ferma il server web Flask."""
        if not hasattr(self, 'flask_thread') or not self.flask_thread.isRunning():
            self.start_web_server()
        else:
            self.stop_web_server()
    
    def start_web_server(self):
        """Avvia il server web Flask in un thread separato."""
        try:
            if not os.path.exists(DB_FILENAME):
                QMessageBox.warning(self, "Warning", 
                                   "Database not found. Please run the scraper first.")
                return
            
            # Crea e avvia thread Flask
            self.flask_thread = FlaskServerThread()
            self.flask_thread.log_signal.connect(self.on_flask_log)
            self.flask_thread.started_signal.connect(self.on_flask_started)
            self.flask_thread.stopped_signal.connect(self.on_flask_stopped)
            self.flask_thread.error_signal.connect(self.on_flask_error)
            
            # Disabilita pulsante durante l'avvio
            self.web_viewer_btn.setEnabled(False)
            self.web_viewer_btn.setText("🌐 Starting Server...")
            
            # Avvia thread (NON blocca la GUI)
            self.flask_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start web server: {str(e)}")
            self.web_viewer_btn.setEnabled(True)
    
    def stop_web_server(self):
        """Ferma il server web Flask."""
        try:
            if hasattr(self, 'flask_thread') and self.flask_thread.isRunning():
                self.web_viewer_btn.setEnabled(False)
                self.web_viewer_btn.setText("🌐 Stopping Server...")
                
                # Ferma il thread (NON blocca la GUI)
                self.flask_thread.stop_server()
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Error stopping server: {str(e)}")
            self.on_flask_stopped()
    
    def on_flask_log(self, message):
        """Gestisce i log del server Flask."""
        self.append_bot_log(message)
        print(message)
    
    def on_flask_started(self):
        """Chiamato quando Flask è avviato."""
        self.web_viewer_btn.setEnabled(True)
        self.web_viewer_btn.setText("🛑 Stop Web Viewer")
        self.web_viewer_btn.setStyleSheet(
            "QPushButton { background-color: #e74c3c; color: white; padding: 5px 15px; font-weight: bold; }")
        
        # Apri browser dopo 1 secondo
        QTimer.singleShot(1000, self.open_web_browser)
        
        QMessageBox.information(self, "Web Server Started", 
                               "Flask web server is running!\n\n"
                               "URL: http://localhost:5000\n\n"
                               "Your browser will open automatically.")
    
    def on_flask_stopped(self):
        """Chiamato quando Flask è fermato."""
        self.web_viewer_btn.setEnabled(True)
        self.web_viewer_btn.setText("🖥️ Open Web Viewer")
        self.web_viewer_btn.setStyleSheet(
            "QPushButton { background-color: #27ae60; color: white; padding: 5px 15px; }")
    
    def on_flask_error(self, error_message):
        """Chiamato in caso di errore Flask."""
        QMessageBox.critical(self, "Flask Server Error", error_message)
        self.on_flask_stopped()
    
    def open_web_browser(self):
        """Apre il browser con l'URL del server Flask."""
        import webbrowser
        try:
            webbrowser.open('http://localhost:5000')
        except Exception as e:
            print(f"Could not open browser: {e}")


    def on_wishlist_changed(self, card_id, is_wishlisted):
        """Gestisce il cambio di stato della wishlist."""
        try:
            with sqlite3.connect(DB_FILENAME) as conn:
                cursor = conn.cursor()
                
                if is_wishlisted:
                    # Aggiungi alla wishlist
                    cursor.execute("""
                        INSERT OR IGNORE INTO wishlist (card_id, added_date)
                        VALUES (?, ?)
                    """, (card_id, datetime.now().isoformat()))
                else:
                    # Rimuovi dalla wishlist
                    cursor.execute("DELETE FROM wishlist WHERE card_id = ?", (card_id,))
                
                conn.commit()
        except Exception as e:
            print(f"Error updating wishlist: {e}")


    def check_database_content(self):
        """Verifica il contenuto del database (debug)."""
        try:
            with sqlite3.connect(DB_FILENAME) as conn:
                cursor = conn.cursor()
                
                # Conta set
                cursor.execute("SELECT COUNT(*) FROM sets")
                sets_count = cursor.fetchone()[0]
                
                # Conta carte
                cursor.execute("SELECT COUNT(*) FROM cards")
                cards_count = cursor.fetchone()[0]
                
                # Mostra i set
                cursor.execute("SELECT set_code, set_name, total_cards FROM sets")
                sets_list = cursor.fetchall()
                
                msg = f"📊 Database Content:\n\n"
                msg += f"Sets: {sets_count}\n"
                msg += f"Cards: {cards_count}\n\n"
                
                if sets_list:
                    msg += "Sets in database:\n"
                    for set_code, set_name, total in sets_list:
                        msg += f"  • {set_code}: {set_name} ({total} cards)\n"
                else:
                    msg += "⚠️ No sets found in database!\n"
                
                QMessageBox.information(self, "Database Check", msg)
                self.append_scraper_log(msg)
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to check database: {str(e)}")


    def setup_bot_tab(self):
        """Configura il tab del Discord bot."""
        bot_widget = QWidget()
        bot_layout = QVBoxLayout(bot_widget)
        
        # Configuration Group
        config_group = QGroupBox("🔧 Configuration")
        config_layout = QVBoxLayout(config_group)
        
        # Token
        token_layout = QHBoxLayout()
        token_layout.addWidget(QLabel("Bot Token:"))
        self.token_input = QLineEdit()
        self.token_input.setEchoMode(QLineEdit.Password)
        self.token_input.setPlaceholderText("Enter your Discord bot token")
        token_layout.addWidget(self.token_input)
        config_layout.addLayout(token_layout)
        
        # Channel ID
        channel_layout = QHBoxLayout()
        channel_layout.addWidget(QLabel("Channel ID:"))
        self.channel_input = QLineEdit()
        self.channel_input.setPlaceholderText("Enter the Discord channel ID")
        channel_layout.addWidget(self.channel_input)
        config_layout.addLayout(channel_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.start_bot_btn = QPushButton("▶ Start Bot")
        self.start_bot_btn.clicked.connect(self.start_bot)
        self.start_bot_btn.setStyleSheet("QPushButton { background-color: #2ecc71; color: white; padding: 8px; font-weight: bold; }")
        
        self.stop_bot_btn = QPushButton("⏹ Stop Bot")
        self.stop_bot_btn.clicked.connect(self.stop_bot)
        self.stop_bot_btn.setEnabled(False)
        self.stop_bot_btn.setStyleSheet("QPushButton { background-color: #e74c3c; color: white; padding: 8px; font-weight: bold; }")
        
        button_layout.addWidget(self.start_bot_btn)
        button_layout.addWidget(self.stop_bot_btn)
        config_layout.addLayout(button_layout)
        
        bot_layout.addWidget(config_group)
        
        # Status Group
        status_group = QGroupBox("📊 Status")
        status_layout = QVBoxLayout(status_group)
        self.bot_status_label = QLabel("Status: ⚫ Stopped")
        self.bot_status_label.setStyleSheet("QLabel { font-size: 14px; font-weight: bold; }")
        status_layout.addWidget(self.bot_status_label)
        bot_layout.addWidget(status_group)
        
        # Progress
        progress_group = QGroupBox("⏳ Progress")
        progress_layout = QVBoxLayout(progress_group)
        self.bot_progress_bar = QProgressBar()
        self.bot_progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #3498db; }")
        progress_layout.addWidget(self.bot_progress_bar)
        bot_layout.addWidget(progress_group)
        
        # Log
        log_group = QGroupBox("📝 Log")
        log_layout = QVBoxLayout(log_group)
        self.bot_log_text = QTextEdit()
        self.bot_log_text.setReadOnly(True)
        self.bot_log_text.setStyleSheet("""
            QTextEdit { 
                font-family: 'Consolas', 'Segoe UI Mono', monospace; 
                font-size: 12px; 
            }
        """)
        log_layout.addWidget(self.bot_log_text)
        bot_layout.addWidget(log_group)
        
        # Trades Table CON MINIATURE
        trades_group = QGroupBox("📦 Recent Trades")
        trades_layout = QVBoxLayout(trades_group)
        self.trades_table = QTableWidget()
        self.trades_table.setColumnCount(5)
        self.trades_table.setHorizontalHeaderLabels(["Preview", "Account", "Cards Found", "XML", "Image"])
        self.trades_table.horizontalHeader().setStretchLastSection(True)
        self.trades_table.setAlternatingRowColors(True)
        
        # Imposta dimensioni colonne
        self.trades_table.setColumnWidth(0, 80)  # Preview column
        self.trades_table.verticalHeader().setDefaultSectionSize(70)  # Altezza righe
        
        # Abilita interazioni
        self.trades_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.trades_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.trades_table.cellClicked.connect(self.on_trade_table_clicked)
        self.trades_table.cellDoubleClicked.connect(self.on_trade_table_double_clicked)
        
        trades_layout.addWidget(self.trades_table)
        bot_layout.addWidget(trades_group)
        
        self.tabs.addTab(bot_widget, "🤖 Discord Bot")
    
    def setup_cards_found_tab(self):
        """Configura il tab delle carte trovate."""
        cards_widget = QWidget()
        cards_layout = QVBoxLayout(cards_widget)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.clear_cards_btn = QPushButton("🗑️ Clear List")
        self.clear_cards_btn.clicked.connect(self.clear_cards_list)
        controls_layout.addWidget(self.clear_cards_btn)
        
        self.export_cards_btn = QPushButton("💾 Export to CSV")
        self.export_cards_btn.clicked.connect(self.export_cards_to_csv)
        controls_layout.addWidget(self.export_cards_btn)
        
        controls_layout.addStretch()
        
        self.cards_count_label = QLabel("Total Cards Found: 0")
        self.cards_count_label.setStyleSheet("QLabel { font-size: 12px; font-weight: bold; }")
        controls_layout.addWidget(self.cards_count_label)
        
        cards_layout.addLayout(controls_layout)
        
        # Cards Table CON MINIATURE (CARTA + PACCHETTO)
        self.cards_table = QTableWidget()
        self.cards_table.setColumnCount(7)  # ⬅️ Aumentato da 6 a 7
        self.cards_table.setHorizontalHeaderLabels([
            "Card", "Pack", "Account", "Set", "Card #", "Card Name", "Rarity"  # ⬅️ Aggiunto "Pack"
        ])
        self.cards_table.horizontalHeader().setStretchLastSection(True)
        self.cards_table.setAlternatingRowColors(True)
        self.cards_table.setSortingEnabled(True)
        
        # Imposta dimensioni colonne
        self.cards_table.setColumnWidth(0, 80)  # Card preview
        self.cards_table.setColumnWidth(1, 80)  # Pack preview ⬅️ NUOVA COLONNA
        self.cards_table.verticalHeader().setDefaultSectionSize(70)  # Altezza righe
        
        # Abilita interazioni
        self.cards_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.cards_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.cards_table.cellClicked.connect(self.on_card_table_clicked)
        self.cards_table.cellDoubleClicked.connect(self.on_card_table_double_clicked)
        
        cards_layout.addWidget(self.cards_table)
        
        self.tabs.addTab(cards_widget, "🎴 Cards Found")

    def download_cloudflared(self):
        """Scarica cloudflared.exe automaticamente e riavvia l'app."""
        try:
            from urllib.request import urlretrieve
            import shutil
            
            # URL di download per Windows 64-bit
            cloudflared_url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
            
            # Path di destinazione finale (nella cartella dell'app)
            if getattr(sys, 'frozen', False):
                # Se è EXE, metti nella stessa cartella dell'EXE
                app_dir = os.path.dirname(sys.executable)
            else:
                # Se è sviluppo, metti nella cartella corrente
                app_dir = os.getcwd()
            
            final_path = os.path.join(app_dir, 'cloudflared.exe')
            
            # Path temporaneo per il download
            temp_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'cloudflared-windows-amd64.exe')
            
            # Mostra progress dialog
            progress = QProgressBar()
            progress_dialog = QMessageBox(self)
            progress_dialog.setWindowTitle("Downloading Cloudflared")
            progress_dialog.setText("Downloading cloudflared.exe...\n\nThis may take a few minutes (~50 MB).")
            progress_dialog.setStandardButtons(QMessageBox.NoButton)
            progress_dialog.layout().addWidget(progress, 1, 1)
            progress_dialog.show()
            QApplication.processEvents()
            
            # Funzione per aggiornare progress
            def report_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = int((downloaded / total_size) * 100)
                    progress.setValue(min(percent, 100))
                QApplication.processEvents()
            
            # Download nel path temporaneo
            self.append_bot_log(f"📥 Downloading cloudflared from GitHub...")
            urlretrieve(cloudflared_url, temp_path, reporthook=report_progress)
            
            # Copia nella cartella dell'app e rinomina
            self.append_bot_log(f"📂 Installing to: {final_path}")
            shutil.copy2(temp_path, final_path)
            
            # Rimuovi file temporaneo
            try:
                os.remove(temp_path)
            except:
                pass
            
            progress_dialog.close()
            
            # ⬇️ CHIEDI SE RIAVVIARE L'APP ⬇️
            reply = QMessageBox.question(
                self,
                "Download Complete",
                "Cloudflared installed successfully!\n\n"
                "The application needs to restart to use Cloudflare Tunnel.\n\n"
                "Restart now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                self.restart_application()
            else:
                QMessageBox.information(
                    self,
                    "Restart Required",
                    "Please restart the application manually to use Cloudflare Tunnel."
                )
            
        except Exception as e:
            import traceback
            error_msg = f"Failed to download cloudflared: {str(e)}\n\n{traceback.format_exc()}"
            self.append_bot_log(f"❌ {error_msg}")
            
            QMessageBox.critical(
                self,
                "Download Failed",
                f"Failed to download cloudflared: {str(e)}\n\n"
                f"Please download manually from:\n"
                f"https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/\n\n"
                f"And place 'cloudflared.exe' in:\n{app_dir}"
            )

    def restart_application(self):
        """Riavvia l'applicazione."""
        try:
            self.append_bot_log("🔄 Restarting application...")
            
            # Salva le impostazioni prima di riavviare
            self.save_settings()
            
            # Ferma tutti i thread attivi
            if hasattr(self, 'bot_thread') and self.bot_thread and self.bot_thread.isRunning():
                self.bot_thread.stop_bot()
                self.bot_thread.wait(3000)
            
            if hasattr(self, 'scraper_thread') and self.scraper_thread and self.scraper_thread.isRunning():
                self.scraper_thread.quit()
                self.scraper_thread.wait(3000)
            
            if hasattr(self, 'flask_thread') and self.flask_thread and self.flask_thread.isRunning():
                self.flask_thread.stop_server()
                self.flask_thread.wait(3000)
            
            # Ottieni path dell'eseguibile
            if getattr(sys, 'frozen', False):
                # Se è EXE
                executable = sys.executable
            else:
                # Se è sviluppo (Python script)
                executable = sys.executable
                script = os.path.abspath(sys.argv[0])
            
            # Chiudi l'app corrente
            QApplication.quit()
            
            # Riavvia in un nuovo processo
            if getattr(sys, 'frozen', False):
                # EXE: riavvia direttamente
                subprocess.Popen([executable])
            else:
                # Script Python: riavvia con Python
                subprocess.Popen([executable, script])
            
            # Termina il processo corrente
            sys.exit(0)
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Restart Failed",
                f"Failed to restart application: {str(e)}\n\n"
                f"Please restart manually."
            )


    def setup_collection_tab(self):
        """Configura il tab della collezione con sezioni a scomparsa per set."""
        collection_widget = QWidget()
        collection_layout = QVBoxLayout(collection_widget)
        
        # Header con controlli
        header_layout = QHBoxLayout()
        
        # Account selector
        header_layout.addWidget(QLabel("Account:"))
        self.collection_account_combo = QComboBox()
        self.collection_account_combo.addItem("All Accounts")
        self.collection_account_combo.currentTextChanged.connect(self.refresh_collection)
        header_layout.addWidget(self.collection_account_combo)
        
        header_layout.addStretch()
        cloudflare_btn = QPushButton("☁️ Configura Cloudflare")
        cloudflare_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff9500;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ff8c00;
            }
        """)
        cloudflare_btn.clicked.connect(self.open_cloudflare_dialog)
        header_layout.addWidget(cloudflare_btn)

        self.tunnel_btn = QPushButton("📱 Expose Publicly (Cloudflare)")
        self.tunnel_btn.clicked.connect(self.toggle_cloudflare_tunnel)
        self.tunnel_btn.setStyleSheet("QPushButton { background-color: #2c3e50; color: white; padding: 5px 15px; }")
        header_layout.addWidget(self.tunnel_btn)
        self.web_viewer_btn = QPushButton("🖥️ Open Web Viewer")
        self.web_viewer_btn.clicked.connect(self.toggle_web_server)
        self.web_viewer_btn.setStyleSheet("QPushButton { background-color: #27ae60; color: white; padding: 5px 15px; }")
        header_layout.addWidget(self.web_viewer_btn)        
        # Refresh button
        refresh_collection_btn = QPushButton("🔄 Refresh Collection")
        refresh_collection_btn.clicked.connect(self.refresh_collection)
        refresh_collection_btn.setStyleSheet("QPushButton { background-color: #3498db; color: white; padding: 5px 15px; }")
        header_layout.addWidget(refresh_collection_btn)
        
        collection_layout.addLayout(header_layout)
        
        # Stats bar
        self.collection_stats_label = QLabel("Loading collection...")
        self.collection_stats_label.setStyleSheet("QLabel { font-size: 11px; color: #888; padding: 5px; }")
        collection_layout.addWidget(self.collection_stats_label)
        
        # Scroll area per le sezioni dei set
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Widget container per i set
        self.collection_container = QWidget()
        self.collection_container_layout = QVBoxLayout(self.collection_container)
        self.collection_container_layout.setSpacing(5)
        self.collection_container_layout.addStretch()
        
        scroll_area.setWidget(self.collection_container)
        collection_layout.addWidget(scroll_area)
        
        self.tabs.addTab(collection_widget, "📚 Collection")

    # =========================================================================
    # CLOUDFLARE TUNNEL
    # =========================================================================
    
    def toggle_cloudflare_tunnel(self):
        """Avvia o ferma il tunnel Cloudflare."""
        if not hasattr(self, 'tunnel_thread') or not self.tunnel_thread.isRunning():
            self.start_cloudflare_tunnel()
        else:
            self.stop_cloudflare_tunnel()
    
    def start_cloudflare_tunnel(self):
        """Avvia il tunnel Cloudflare per esporre Flask pubblicamente."""
        try:
            # ⬇️ STEP 1: Verifica che cloudflared.exe esista ⬇️
            if getattr(sys, 'frozen', False):
                cloudflared_path = os.path.join(sys._MEIPASS, 'cloudflared.exe')
            else:
                cloudflared_path = os.path.join(os.getcwd(), 'cloudflared.exe')
            
            if not os.path.exists(cloudflared_path):
                # Cloudflared non trovato - chiedi download
                reply = QMessageBox.question(
                    self, 
                    'Cloudflared Not Found',
                    'Cloudflared is required to expose your app publicly.\n\n'
                    'Would you like to download it now?\n\n'
                    '(Download size: ~50 MB)',
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                
                if reply == QMessageBox.Yes:
                    self.download_cloudflared()
                else:
                    self.append_bot_log("⚠️ Cloudflare Tunnel cancelled - cloudflared not installed")
                return
            
            # ⬇️ STEP 2: Verifica che Flask sia in esecuzione, altrimenti avvialo ⬇️
            if not hasattr(self, 'flask_thread') or not self.flask_thread.isRunning():
                self.append_bot_log("🌐 Flask not running - starting automatically...")
                
                # Avvia Flask
                self.start_web_server()
                
                # Aspetta 3 secondi per Flask
                QTimer.singleShot(3000, self._start_tunnel_after_flask)
                return
            
            # ⬇️ STEP 3: Avvia il tunnel ⬇️
            self._start_tunnel_now()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start Cloudflare Tunnel: {str(e)}")
            self.tunnel_btn.setEnabled(True)
    
    def _start_tunnel_after_flask(self):
        """Avvia il tunnel dopo che Flask è partito."""
        # Verifica che Flask sia effettivamente partito
        if hasattr(self, 'flask_thread') and self.flask_thread.isRunning():
            self._start_tunnel_now()
        else:
            QMessageBox.warning(self, "Error", "Failed to start Flask web server. Cannot start tunnel.")
            self.tunnel_btn.setEnabled(True)
    
    def _start_tunnel_now(self):
        """Avvia effettivamente il tunnel (helper method)."""
        try:
            # Crea e avvia tunnel thread
            self.tunnel_thread = CloudflareTunnelThread(local_port=5000)
            self.tunnel_thread.log_signal.connect(self.on_tunnel_log)
            self.tunnel_thread.url_ready_signal.connect(self.on_tunnel_url_ready)
            self.tunnel_thread.stopped_signal.connect(self.on_tunnel_stopped)
            self.tunnel_thread.error_signal.connect(self.on_tunnel_error)
            
            # Aggiorna UI
            self.tunnel_btn.setEnabled(False)
            self.tunnel_btn.setText("📱 Starting Tunnel...")
            
            # Avvia thread
            self.tunnel_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start tunnel: {str(e)}")
            self.tunnel_btn.setEnabled(True)

    
    def stop_cloudflare_tunnel(self):
        """Ferma il tunnel Cloudflare."""
        try:
            if hasattr(self, 'tunnel_thread') and self.tunnel_thread.isRunning():
                self.tunnel_btn.setEnabled(False)
                self.tunnel_btn.setText("🛑 Stopping Tunnel...")
                
                self.tunnel_thread.stop_tunnel()
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Error stopping tunnel: {str(e)}")
            self.on_tunnel_stopped()
    
    def on_tunnel_log(self, message):
        """Gestisce i log del tunnel."""
        self.append_bot_log(message)
    
    def on_tunnel_url_ready(self, public_url):
        """Chiamato quando l'URL pubblico è pronto."""
        self.tunnel_btn.setEnabled(True)
        self.tunnel_btn.setText("🛑 Stop Public Exposure")
        self.tunnel_btn.setStyleSheet(
            "QPushButton { background-color: #e74c3c; color: white; padding: 5px 15px; font-weight: bold; }")
        
        # Mostra dialog con URL
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Public URL Ready")
        msg.setText(f"Your app is now accessible publicly!\n\nURL: {public_url}")
        msg.setDetailedText(
            "Share this URL with anyone to access your collection viewer.\n"
            "The tunnel will stay active until you stop it."
        )
        
        # Pulsante per copiare URL
        copy_btn = msg.addButton("Copy URL", QMessageBox.ActionRole)
        open_btn = msg.addButton("Open in Browser", QMessageBox.ActionRole)
        msg.addButton(QMessageBox.Ok)
        
        msg.exec_()
        
        clicked = msg.clickedButton()
        if clicked == copy_btn:
            QApplication.clipboard().setText(public_url)
            self.append_bot_log("📋 Public URL copied to clipboard")
        elif clicked == open_btn:
            import webbrowser
            webbrowser.open(public_url)
    
    def on_tunnel_stopped(self):
        """Chiamato quando il tunnel è fermato."""
        self.tunnel_btn.setEnabled(True)
        self.tunnel_btn.setText("📱 Expose Publicly (Cloudflare)")
        self.tunnel_btn.setStyleSheet(
            "QPushButton { background-color: #2c3e50; color: white; padding: 5px 15px; }")
    
    def on_tunnel_error(self, error_message):
        """Chiamato in caso di errore tunnel."""
        QMessageBox.critical(self, "Cloudflare Tunnel Error", error_message)
        self.on_tunnel_stopped()


    def setup_database_tab(self):
        """Configura il tab del database."""
        db_widget = QWidget()
        db_layout = QVBoxLayout(db_widget)
        
        # Configuration Group
        config_group = QGroupBox("⚙️ Scraper Configuration")
        config_layout = QVBoxLayout(config_group)
        
        # Download Images Checkbox
        self.download_images_cb = QCheckBox("Download Card Images")
        self.download_images_cb.setChecked(True)
        config_layout.addWidget(self.download_images_cb)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.start_scraper_btn = QPushButton("▶ Start Database Setup")
        self.start_scraper_btn.clicked.connect(self.start_scraper)
        self.start_scraper_btn.setStyleSheet("QPushButton { background-color: #2ecc71; color: white; padding: 8px; font-weight: bold; }")
        
        self.stop_scraper_btn = QPushButton("⏹ Stop Scraper")
        self.stop_scraper_btn.clicked.connect(self.stop_scraper)
        self.stop_scraper_btn.setEnabled(False)
        self.stop_scraper_btn.setStyleSheet("QPushButton { background-color: #e74c3c; color: white; padding: 8px; font-weight: bold; }")
        # Nel metodo setup_database_tab(), dopo i pulsanti Start/Stop, aggiungi:

        self.update_phash_btn = QPushButton("🔄 Update pHash for Existing Images")
        self.update_phash_btn.clicked.connect(self.update_phash_values)
        button_layout.addWidget(self.update_phash_btn)


        self.check_db_btn = QPushButton("🔍 Check Database")
        self.check_db_btn.clicked.connect(self.check_database_content)
        button_layout.addWidget(self.check_db_btn)

        button_layout.addWidget(self.start_scraper_btn)
        button_layout.addWidget(self.stop_scraper_btn)
        config_layout.addLayout(button_layout)
        
        db_layout.addWidget(config_group)
        
        # Progress Group
        progress_group = QGroupBox("⏳ Progress")
        progress_layout = QVBoxLayout(progress_group)
        self.scraper_progress_bar = QProgressBar()
        self.scraper_progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #9b59b6; }")
        progress_layout.addWidget(self.scraper_progress_bar)
        db_layout.addWidget(progress_group)
        
        # Log
        log_group = QGroupBox("📝 Scraper Log")
        log_layout = QVBoxLayout(log_group)
        self.scraper_log_text = QTextEdit()
        self.scraper_log_text.setReadOnly(True)
        self.scraper_log_text.setStyleSheet("QTextEdit { font-family: 'Courier New'; font-size: 12px; }")  # ⬅️ 10px → 12px
        log_layout.addWidget(self.scraper_log_text)
        db_layout.addWidget(log_group)
        
        self.tabs.addTab(db_widget, "💾 Database Setup")
    
    def setup_stats_tab(self):
        """Configura il tab delle statistiche."""
        stats_widget = QWidget()
        stats_layout = QVBoxLayout(stats_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.refresh_stats_btn = QPushButton("🔄 Refresh Statistics")
        self.refresh_stats_btn.clicked.connect(self.refresh_stats)
        self.refresh_stats_btn.setStyleSheet("QPushButton { background-color: #3498db; color: white; padding: 8px; font-weight: bold; }")
        button_layout.addWidget(self.refresh_stats_btn)
        button_layout.addStretch()
        stats_layout.addLayout(button_layout)
        
        # Stats Text
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setStyleSheet("QTextEdit { font-family: 'Courier New'; font-size: 11px; }")
        stats_layout.addWidget(self.stats_text)
        
        self.tabs.addTab(stats_widget, "📊 Statistics")
    
    def setup_settings_tab(self):
        """Configura il tab delle impostazioni."""
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)
        info_group = QGroupBox("ℹ️ Extra & Info")
        info_layout = QVBoxLayout(info_group)
        
        # Testo informativo con formatting
        info_text = QTextBrowser() 
        info_text.setReadOnly(True)
        info_text.setMaximumHeight(300)
        info_text.setStyleSheet("""
            QTextEdit {
                background-color: #2a2a2a;
                border: 1px solid #555;
                border-radius: 5px;
                padding: 15px;
                font-size: 11px;
                line-height: 1.6;
            }
        """)
        
        # Contenuto HTML con link cliccabili
        info_html = """
        <div style="color: #e0e0e0; font-family: 'Segoe UI', Arial, sans-serif;">
            <p style="font-size: 13px; font-weight: bold; color: #3498db; margin-bottom: 10px;">
                🎴 Pokemon TeamRocket Tool
            </p>
            
            <p style="margin-bottom: 12px;">
                Thank you for downloading <b>Pokemon TeamRocket Tool</b>!<br>
                Created by <b style="color: #e74c3c;">pcb.is.good</b>, 
                designed to work alongside 
                <a href="https://github.com/Arturo-1212/PTCGPB" 
                   style="color: #3498db; text-decoration: none;">
                    Arturo-1212/PTCGPB
                </a>.
            </p>
            
            <p style="font-size: 12px; font-weight: bold; color: #f39c12; margin-top: 15px; margin-bottom: 8px;">
                💝 Special Thanks:
            </p>
            <ul style="margin-left: 20px; margin-top: 5px;">
                <li style="margin-bottom: 5px;">
                    <b>Arturo</b> (Bot Creator) - 
                    <a href="https://github.com/Arturo-1212" 
                       style="color: #3498db; text-decoration: none;">
                        Arturo-1212
                    </a>
                </li>
                <li style="margin-bottom: 5px;">
                    <b>GummyBaer</b> (Feedback + Card and Pack Matching Algorithm)
                </li>
            </ul>
            
            <p style="margin-top: 15px; margin-bottom: 8px;">
                For any questions, refer to the official bot Discord:
            </p>
            <p style="margin-left: 20px;">
                🔗 <a href="https://discord.gg/Msa5vNjUUf" 
                       style="color: #7289da; text-decoration: none; font-weight: bold;">
                    https://discord.gg/Msa5vNjUUf
                </a>
            </p>
            
            <hr style="border: none; border-top: 1px solid #555; margin: 15px 0;">
            
            <p style="font-size: 10px; color: #888; text-align: center;">
                Version 1.0 | Built with ❤️ for the TCG Pocket community
            </p>
        </div>
        """
        
        info_text.setHtml(info_html)
        
        # Abilita apertura link esterni
        info_text.setOpenExternalLinks(True)
        
        info_layout.addWidget(info_text)
        
        # Pulsanti social/link veloci
        links_layout = QHBoxLayout()
        
        # Pulsante Discord
        discord_btn = QPushButton("💬 Join Discord")
        discord_btn.clicked.connect(lambda: self.open_url("https://discord.gg/Msa5vNjUUf"))
        discord_btn.setStyleSheet("""
            QPushButton {
                background-color: #7289da;
                color: white;
                padding: 8px 15px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #5b6eae;
            }
        """)
        links_layout.addWidget(discord_btn)
        
        # Pulsante GitHub Bot
        github_bot_btn = QPushButton("📦 PTCGPB GitHub")
        github_bot_btn.clicked.connect(lambda: self.open_url("https://github.com/Arturo-1212/PTCGPB"))
        github_bot_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                padding: 8px 15px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
        """)
        links_layout.addWidget(github_bot_btn)
        
        # Pulsante GitHub Creator
        github_creator_btn = QPushButton("👤 pcb.is.good")
        github_creator_btn.clicked.connect(lambda: self.open_url("https://github.com/pcbisgood"))
        github_creator_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 8px 15px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        links_layout.addWidget(github_creator_btn)
        
        links_layout.addStretch()
        
        info_layout.addLayout(links_layout)
        
        settings_layout.addWidget(info_group)
        
        # ⬆️ FINE DEL NUOVO BLOCCO ⬆️
        
        settings_layout.addStretch()
        
        self.tabs.addTab(settings_widget, "⚙️ Settings")       
        settings_group = QGroupBox("⚙️ Application Settings")
        settings_group_layout = QVBoxLayout(settings_group)
        
        # Theme selection (placeholder)
        theme_layout = QHBoxLayout()
        theme_layout.addWidget(QLabel("Theme:"))
        self.theme_combo = QCheckBox("Dark Theme (Default)")
        self.theme_combo.setChecked(True)
        theme_layout.addWidget(self.theme_combo)
        theme_layout.addStretch()
        settings_group_layout.addLayout(theme_layout)
        
        # Auto-start bot
        self.autostart_cb = QCheckBox("Auto-start bot on launch")
        settings_group_layout.addWidget(self.autostart_cb)
        
        # Minimize to tray
        self.minimize_tray_cb = QCheckBox("Minimize to system tray")
        self.minimize_tray_cb.setChecked(True)
        settings_group_layout.addWidget(self.minimize_tray_cb)
        
        settings_layout.addWidget(settings_group)
        settings_layout.addStretch()
        
        self.tabs.addTab(settings_widget, "⚙️ Settings")

    def show_from_tray(self):
        """Mostra la finestra dal system tray."""
        self.show()
        self.raise_()
        self.activateWindow()
    
    def quit_application(self):
        """✅ CORRETTO - Chiude l'applicazione in modo sicuro."""
        try:
            print("🛑 Shutting down application...")
            
            # Ferma il bot Discord se è in esecuzione
            if hasattr(self, 'bot_thread') and self.bot_thread:
                try:
                    if hasattr(self.bot_thread, 'stop_bot'):
                        self.bot_thread.stop_bot()
                    self.bot_thread.join(timeout=2)
                    print("✅ Discord bot stopped")
                except Exception as e:
                    print(f"⚠️ Error stopping bot: {e}")
            
            # Ferma Flask se è in esecuzione
            if hasattr(self, 'flask_thread') and self.flask_thread:
                try:
                    if hasattr(self.flask_thread, 'stop_server'):
                        self.flask_thread.stop_server()
                    self.flask_thread.join(timeout=2)
                    print("✅ Flask server stopped")
                except Exception as e:
                    print(f"⚠️ Error stopping Flask: {e}")
            
            # Chiudi il database
            if hasattr(self, 'db_manager') and self.db_manager:
                try:
                    self.db_manager.close()
                    print("✅ Database closed")
                except Exception as e:
                    print(f"⚠️ Error closing database: {e}")
            
            # Chiudi Cloudflare tunnel se attivo
            if hasattr(self, 'cloudflare_process') and self.cloudflare_process:
                try:
                    self.cloudflare_process.terminate()
                    self.cloudflare_process.wait(timeout=2)
                    print("✅ Cloudflare tunnel closed")
                except Exception as e:
                    print(f"⚠️ Error closing Cloudflare: {e}")
            
            # Chiudi l'applicazione PyQt
            QApplication.quit()
            print("✅ Application closed")
            
        except Exception as e:
            print(f"❌ Error during shutdown: {e}")
            import traceback
            traceback.print_exc()
            # Force quit comunque
            QApplication.quit()


    def setup_system_tray(self):
        """Configura l'icona nel system tray."""
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
        
        self.tray_icon = QSystemTrayIcon(self)
        
        # Imposta icona
        if os.path.exists(ICON_PATH):
            self.tray_icon.setIcon(QIcon(ICON_PATH))
        else:
            self.tray_icon.setIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        
        # Menu del tray
        tray_menu = QMenu()
        
        show_action = tray_menu.addAction("Show")
        show_action.triggered.connect(self.show_from_tray)  # ⬅️ Cambia qui
        
        quit_action = tray_menu.addAction("Quit")
        quit_action.triggered.connect(self.quit_application)  # ⬅️ Cambia qui
        
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self.tray_icon_activated)
        self.tray_icon.show()
    
    def tray_icon_activated(self, reason):
        """Gestisce il click sull'icona del system tray."""
        if reason == QSystemTrayIcon.DoubleClick:
            self.show()

    # =========================================================================
    # BOT CONTROL FUNCTIONS
    # =========================================================================
    
    def start_bot(self):
        """Avvia il Discord bot."""
        token = self.token_input.text().strip()
        channel_id = self.channel_input.text().strip()
        
        if not token or not channel_id:
            QMessageBox.warning(self, "Error", "Please provide both Token and Channel ID")
            return
        
        try:
            channel_id = int(channel_id)
        except:
            QMessageBox.warning(self, "Error", "Channel ID must be a number")
            return
        
        self.bot_thread = DiscordBotThread(token, channel_id)
        self.bot_thread.log_signal.connect(self.append_bot_log)
        self.bot_thread.progress_signal.connect(self.update_bot_progress)
        self.bot_thread.trade_signal.connect(self.add_trade_to_table)
        self.bot_thread.status_signal.connect(self.update_bot_status)
        self.bot_thread.card_found_signal.connect(self.on_card_found)
        
        self.bot_thread.start()
        
        self.start_bot_btn.setEnabled(False)
        self.stop_bot_btn.setEnabled(True)
        self.bot_status_label.setText("Status: 🟡 Starting...")
        
        self.save_settings()
    
    def stop_bot(self):
        """Ferma il Discord bot."""
        if self.bot_thread:
            self.bot_thread.stop()
            self.bot_thread = None
        
        self.start_bot_btn.setEnabled(True)
        self.stop_bot_btn.setEnabled(False)
        self.bot_status_label.setText("Status: ⚫ Stopped")
        self.append_bot_log("Bot stopped")
    
    def start_scraper(self):
        """Avvia lo scraper del database."""
        download_images = self.download_images_cb.isChecked()
        
        reply = QMessageBox.question(self, 'Confirm', 
                                     'This will download all card data from the website. Continue?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.No:
            return
        
        self.scraper_thread = ScraperThread(download_images=download_images)
        self.scraper_thread.log_signal.connect(self.append_scraper_log)
        self.scraper_thread.progress_signal.connect(self.update_scraper_progress)
        self.scraper_thread.finished_signal.connect(self.on_scraper_finished)
        
        self.scraper_thread.start()
        
        self.start_scraper_btn.setEnabled(False)
        self.stop_scraper_btn.setEnabled(True)
    
    def stop_scraper(self):
        """Ferma lo scraper del database."""
        if self.scraper_thread:
            self.scraper_thread.terminate()
            self.scraper_thread = None
        
        self.start_scraper_btn.setEnabled(True)
        self.stop_scraper_btn.setEnabled(False)
        self.append_scraper_log("Scraper stopped")
    
    # ⬇️ INSERISCI QUI ⬇️
    def update_phash_values(self):
        """Aggiorna i valori pHash per le carte esistenti."""
        reply = QMessageBox.question(self, 'Confirm', 
                                     'This will calculate pHash for all cards with images. Continue?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            try:
                db_manager = DatabaseManager(log_callback=self.append_scraper_log)
                db_manager.connect()
                db_manager.update_all_phash(log_callback=self.append_scraper_log)
                db_manager.close()
                QMessageBox.information(self, "Success", "pHash values updated!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to update pHash: {str(e)}")
    
    # =========================================================================
    # LOG AND PROGRESS FUNCTIONS
    # =========================================================================
    
    def append_bot_log(self, message):
        """Aggiunge un messaggio al log del bot."""
        # ... resto del codice

    
    # =========================================================================
    # LOG AND PROGRESS FUNCTIONS
    # =========================================================================
    
    def append_bot_log(self, message):
        """Aggiunge un messaggio al log del bot."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.bot_log_text.append(f"[{timestamp}] {message}")
        self.bot_log_text.verticalScrollBar().setValue(
            self.bot_log_text.verticalScrollBar().maximum()
        )
    
    def append_scraper_log(self, message):
        """Aggiunge un messaggio al log dello scraper."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.scraper_log_text.append(f"[{timestamp}] {message}")
        self.scraper_log_text.verticalScrollBar().setValue(
            self.scraper_log_text.verticalScrollBar().maximum()
        )
    
    def update_bot_progress(self, current, total):
        """Aggiorna la barra di progresso del bot."""
        self.bot_progress_bar.setMaximum(total)
        self.bot_progress_bar.setValue(current)
    
    def update_scraper_progress(self, current, total):
        """Aggiorna la barra di progresso dello scraper."""
        self.scraper_progress_bar.setMaximum(total)
        self.scraper_progress_bar.setValue(current)
    
    def update_bot_status(self, status):
        """Aggiorna lo status del bot."""
        status_icons = {
            "Connected": "🟢",
            "Monitoring": "🟢",
            "Scanning": "🟡",
            "Error": "🔴"
        }
        icon = status_icons.get(status, "⚫")
        self.bot_status_label.setText(f"Status: {icon} {status}")
    
    # =========================================================================
    # TRADE AND CARD MANAGEMENT
    # =========================================================================
    
    def add_trade_to_table(self, trade_data):
        """Aggiunge un trade alla tabella CON MINIATURA."""
        row = self.trades_table.rowCount()
        self.trades_table.insertRow(row)
        
        # Colonna 0: Miniatura dell'immagine
        image_path = trade_data.get('image_path')
        preview_label = QLabel()
        preview_label.setAlignment(Qt.AlignCenter)
        
        if image_path and os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(60, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                preview_label.setPixmap(scaled_pixmap)
            else:
                preview_label.setText("❌")
        else:
            preview_label.setText("🖼️")
        
        self.trades_table.setCellWidget(row, 0, preview_label)
        
        # Altre colonne
        account_item = QTableWidgetItem(trade_data.get('account_name', ''))
        account_item.setData(Qt.UserRole, image_path)  # Salva il path per doppio click
        
        cards_item = QTableWidgetItem(trade_data.get('cards_found', ''))
        xml_item = QTableWidgetItem('✓' if trade_data.get('xml_path') else '✗')
        image_item = QTableWidgetItem('✓' if image_path else '✗')
        
        self.trades_table.setItem(row, 1, account_item)
        self.trades_table.setItem(row, 2, cards_item)
        self.trades_table.setItem(row, 3, xml_item)
        self.trades_table.setItem(row, 4, image_item)
        
        # Keep only last 1000 rows
        if self.trades_table.rowCount() > 1000:
            self.trades_table.removeRow(0)
        
        self.trades_table.scrollToBottom()
    
    def on_card_found(self, card_data):
        """Chiamato quando viene trovata una carta."""
        # Add to log
        self.append_bot_log(
            f"   → {card_data['set_code']}_{card_data['card_number']} - "
            f"{card_data['card_name']} ({card_data['rarity']}) x{card_data['count']}"
        )
        
        # Controlla se la carta è nella wishlist
        try:
            with sqlite3.connect(DB_FILENAME) as conn:
                cursor = conn.cursor()
                
                # Ottieni il card_id dalla combinazione set_code + card_number
                cursor.execute("""
                    SELECT id FROM cards
                    WHERE set_code = ? AND card_number = ?
                """, (card_data['set_code'], card_data['card_number']))
                result = cursor.fetchone()
                
                if result:
                    card_id = result[0]
                    
                    # Controlla se è nella wishlist
                    cursor.execute("SELECT 1 FROM wishlist WHERE card_id = ?", (card_id,))
                    is_wishlisted = cursor.fetchone() is not None
                    
                    if is_wishlisted:
                        # Invia notifica Windows
                        self.send_wishlist_notification(card_data)
        except Exception as e:
            print(f"Error checking wishlist: {e}")
        
        # Add to cards table CON DUE MINIATURE (CARTA + PACCHETTO)
        for _ in range(card_data['count']):
            row = self.cards_table.rowCount()
            self.cards_table.insertRow(row)
            
            # Colonna 0: Miniatura della carta
            card_preview_label = QLabel()
            card_preview_label.setAlignment(Qt.AlignCenter)
            
            card_image_path = card_data.get('local_image_path')
            if card_image_path and os.path.exists(card_image_path):
                pixmap = QPixmap(card_image_path)
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(60, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    card_preview_label.setPixmap(scaled_pixmap)
                else:
                    card_preview_label.setText("❌")
            else:
                card_preview_label.setText("🎴")
            
            self.cards_table.setCellWidget(row, 0, card_preview_label)
            
            # ⬇️ COLONNA 1: MINIATURA DEL PACCHETTO (SCREENSHOT) ⬇️
            pack_preview_label = QLabel()
            pack_preview_label.setAlignment(Qt.AlignCenter)
            pack_preview_label.setCursor(Qt.PointingHandCursor)
            
            pack_image_path = card_data.get('image_path')  # Path dello screenshot
            if pack_image_path and os.path.exists(pack_image_path):
                pixmap = QPixmap(pack_image_path)
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(60, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    pack_preview_label.setPixmap(scaled_pixmap)
                    # Salva il path per il doppio click
                    pack_preview_label.setProperty("pack_path", pack_image_path)
                else:
                    pack_preview_label.setText("❌")
            else:
                pack_preview_label.setText("📦")
            
            # Abilita click sulla miniatura del pacchetto
            pack_preview_label.mousePressEvent = lambda event, path=pack_image_path: self.open_pack_image(event, path)
            
            self.cards_table.setCellWidget(row, 1, pack_preview_label)
            
            # Altre colonne (shiftate di 1)
            self.cards_table.setItem(row, 2, QTableWidgetItem(card_data.get('account_name', '')))
            self.cards_table.setItem(row, 3, QTableWidgetItem(card_data.get('set_code', '')))
            self.cards_table.setItem(row, 4, QTableWidgetItem(card_data.get('card_number', '')))
            self.cards_table.setItem(row, 5, QTableWidgetItem(card_data.get('card_name', '')))
            self.cards_table.setItem(row, 6, QTableWidgetItem(card_data.get('rarity', '')))
            
            self.found_cards.append(card_data)
        
        # Update count
        self.cards_count_label.setText(f"Total Cards Found: {self.cards_table.rowCount()}")
        self.cards_table.scrollToBottom()

    def open_pack_image(self, event, pack_path):
        """Apre l'immagine del pacchetto quando si clicca sulla miniatura."""
        if event.button() == Qt.LeftButton:
            if pack_path and os.path.exists(pack_path):
                dialog = ImageViewerDialog(pack_path, self)
                dialog.exec_()
            else:
                QMessageBox.information(self, "Info", "Pack screenshot not found")

    def send_wishlist_notification(self, card_data):
        """Invia una notifica quando una carta nella wishlist viene trovata.
        
        Layout: [Immagine Carta PICCOLA] + Testo + Icona App nel titolo
        """
        if not WINDOWS_TOAST_AVAILABLE:
            return
        
        try:
            card_name = card_data.get('card_name', 'Unknown')
            set_code = card_data.get('set_code', '')
            rarity = card_data.get('rarity', '')
            card_image = card_data.get('local_image_path')
            
            toaster = WindowsToaster('🎴 TCGP Team Rocket Tool')
            toast = Toast()
            
            # ✅ Testo notifica
            toast.text_fields = [
                '❤️ Wishlist Card Found!',
                f'{card_name}',
                f'{set_code} #{card_data.get("card_number", "?")} • {rarity}'
            ]
            
            # =====================================================
            # ✅ IMMAGINE CARTA: Piccola a sinistra (Logo position)
            # =====================================================
            if card_image and os.path.exists(card_image):
                try:
                    image_path = card_image
                    
                    # Se è WebP, converti a PNG
                    if card_image.endswith('.webp'):
                        from PIL import Image
                        import tempfile
                        
                        temp_png = os.path.join(tempfile.gettempdir(), 'card_notification.png')
                        
                        img = Image.open(card_image)
                        
                        # Ridimensiona per notifica piccola (80x120 - dimensione toast)
                        img.thumbnail((80, 120), Image.Resampling.LANCZOS)
                        
                        img.save(temp_png, 'PNG')
                        image_path = temp_png
                    
                    # ✅ Aggiungi come immagine PRINCIPALE (sinistra)
                    # imagePosition=0 = Immagine piccola a sinistra
                    toast.AddImage(
                        ToastDisplayImage.fromPath(image_path)
                    )
                    
                except Exception as e:
                    print(f"⚠️ Error loading card image: {e}")
            
            # =====================================================
            # MOSTRA NOTIFICA
            # =====================================================
            toaster.show_toast(toast)
            
            print(f"✅ Wishlist notification: {card_name} ({set_code})")
            
        except Exception as e:
            print(f"❌ Error sending notification: {e}")
            import traceback
            traceback.print_exc()

    
    def send_windows_toast(self, card_name, set_code, card_number, card_image_path):
        """Invia notifica Windows toast con windows-toasts (Microsoft ufficiale)."""
        try:
            # Crea toaster
            toaster = WindowsToaster('TCGP Team Rocket Tool')
            
            # Crea toast
            new_toast = Toast()
            new_toast.text_fields = [
                '💝 Wishlist Card Found!',
                card_name,
                f'{set_code} #{card_number}'
            ]
            
            # Aggiungi immagine grande della carta
            if card_image_path and os.path.exists(card_image_path):
                new_toast.AddImage(
                    ToastDisplayImage.fromPath(
                        os.path.abspath(card_image_path),
                        position=ToastImagePosition.Hero  # Immagine grande
                    )
                )
            
            # Aggiungi icona app
            if os.path.exists(ICON_PATH):
                new_toast.AddImage(
                    ToastDisplayImage.fromPath(
                        os.path.abspath(ICON_PATH),
                        position=ToastImagePosition.AppLogo,  # Icona piccola
                        circleCrop=True  # Circolare
                    )
                )
            
            # Mostra notifica
            toaster.show_toast(new_toast)
            
            print(f"✅ Windows toast notification sent for {card_name}")
        
        except Exception as e:
            import traceback
            print(f"Error sending toast: {e}\n{traceback.format_exc()}")
            self.send_tray_notification(card_name, set_code, card_number)
    
    def send_tray_notification(self, card_name, set_code, card_number):
        """Invia notifica tramite system tray (fallback senza immagine)."""
        try:
            if hasattr(self, 'tray_icon') and self.tray_icon.isVisible():
                self.tray_icon.showMessage(
                    "💝 Wishlist Card Found!",
                    f"{card_name}\n({set_code} #{card_number})",
                    QSystemTrayIcon.Information,
                    10000
                )
        except Exception as e:
            print(f"Tray notification error: {e}")
    # =========================================================================
    # IMAGE PREVIEW AND INTERACTION
    # =========================================================================
    
    def on_trade_table_clicked(self, row, column):
        """Gestisce il click sulla tabella dei trade."""
        # Mostra anteprima dell'immagine nella status bar (se disponibile)
        pass
    
    def on_trade_table_double_clicked(self, row, column):
        """Gestisce il doppio click sulla tabella dei trade - Apre l'immagine."""
        try:
            # Recupera l'account dalla riga
            account_item = self.trades_table.item(row, 1)
            if not account_item:
                return
            
            account_name = account_item.text()
            
            # Cerca l'immagine più recente per questo account
            image_folder = os.path.join(ACCOUNTS_DIR, account_name, "images")
            
            if os.path.exists(image_folder):
                # Trova l'immagine più recente
                images = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.webp'))]
                if images:
                    images.sort(key=lambda x: os.path.getmtime(os.path.join(image_folder, x)), reverse=True)
                    latest_image = os.path.join(image_folder, images[0])
                    
                    # Apri il dialog
                    dialog = ImageViewerDialog(latest_image, self)
                    dialog.exec_()
                else:
                    QMessageBox.information(self, "Info", "No images found for this account")
            else:
                QMessageBox.information(self, "Info", "Image folder not found")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to open image: {str(e)}")
    
    def on_card_table_clicked(self, row, column):
        """Gestisce il click sulla tabella delle carte."""
        # Evidenzia la riga
        pass
    
    def on_card_table_double_clicked(self, row, column):
        """Gestisce il doppio click sulla tabella delle carte."""
        try:
            # Colonna 0: Apre l'immagine della carta dal database
            if column == 0:
                set_item = self.cards_table.item(row, 3)
                card_num_item = self.cards_table.item(row, 4)
                
                if not set_item or not card_num_item:
                    return
                
                set_code = set_item.text()
                card_number = card_num_item.text()
                
                with sqlite3.connect(DB_FILENAME) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT local_image_path 
                        FROM cards 
                        WHERE set_code = ? AND card_number = ?
                    """, (set_code, card_number))
                    result = cursor.fetchone()
                    
                    if result and result[0]:
                        image_path = result[0]
                        if os.path.exists(image_path):
                            dialog = ImageViewerDialog(image_path, self)
                            dialog.exec_()
                        else:
                            QMessageBox.information(self, "Info", "Card image file not found")
                    else:
                        QMessageBox.information(self, "Info", "Card image not available in database")
            
            # Colonna 1: Apre lo screenshot del pacchetto
            elif column == 1:
                pack_widget = self.cards_table.cellWidget(row, 1)
                if pack_widget and hasattr(pack_widget, 'property'):
                    pack_path = pack_widget.property("pack_path")
                    if pack_path and os.path.exists(pack_path):
                        dialog = ImageViewerDialog(pack_path, self)
                        dialog.exec_()
                    else:
                        QMessageBox.information(self, "Info", "Pack screenshot not found")
        
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to open image: {str(e)}")
    
    def show_image_preview(self, image_path):
        """Mostra un'anteprima piccola dell'immagine (tooltip o status bar)."""
        # Implementazione opzionale per tooltip hover
        pass



    def clear_cards_list(self):
        """Pulisce la lista delle carte trovate."""
        reply = QMessageBox.question(self, 'Confirm', 
                                     'Are you sure you want to clear the cards list?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.cards_table.setRowCount(0)
            self.found_cards.clear()
            self.cards_count_label.setText("Total Cards Found: 0")
    
    def export_cards_to_csv(self):
        """Esporta le carte trovate in CSV."""
        if self.cards_table.rowCount() == 0:
            QMessageBox.information(self, "Info", "No cards to export")
            return
        
        filename, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
        
        if filename:
            try:
                import csv
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    # Header
                    headers = []
                    for col in range(self.cards_table.columnCount()):
                        headers.append(self.cards_table.horizontalHeaderItem(col).text())
                    writer.writerow(headers)
                    
                    # Data
                    for row in range(self.cards_table.rowCount()):
                        row_data = []
                        for col in range(self.cards_table.columnCount()):
                            item = self.cards_table.item(row, col)
                            row_data.append(item.text() if item else '')
                        writer.writerow(row_data)
                
                QMessageBox.information(self, "Success", f"Exported {self.cards_table.rowCount()} cards to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export: {str(e)}")
    
    def on_scraper_finished(self, success):
        """Chiamato quando lo scraper finisce."""
        self.start_scraper_btn.setEnabled(True)
        self.stop_scraper_btn.setEnabled(False)
        
        if success:
            self.append_scraper_log("\n✅ Scraping completato con successo!")
            QMessageBox.information(self, "Success", "Database setup completed successfully!")
            self.refresh_stats()
        else:
            self.append_scraper_log("\n⚠️ Scraping terminato con errori")
            QMessageBox.warning(self, "Warning", "Database setup completed with errors")
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    def refresh_stats(self):
        """Aggiorna le statistiche."""
        try:
            with sqlite3.connect(DB_FILENAME) as conn:
                cursor = conn.cursor()
                
                # Sets
                cursor.execute("SELECT COUNT(*) FROM sets")
                sets_count = cursor.fetchone()[0]
                
                # Cards
                cursor.execute("SELECT COUNT(*) FROM cards")
                cards_count = cursor.fetchone()[0]
                
                # Cards by rarity
                cursor.execute("""
                    SELECT rarity, COUNT(*) 
                    FROM cards 
                    WHERE rarity IN (?, ?, ?, ?)
                    GROUP BY rarity
                """, TARGET_RARITIES)
                rarity_counts = cursor.fetchall()
                
                # Accounts
                cursor.execute("SELECT COUNT(*) FROM accounts")
                accounts_count = cursor.fetchone()[0]
                
                # Total inventory
                cursor.execute("SELECT SUM(quantity) FROM account_inventory")
                total_inventory = cursor.fetchone()[0] or 0
                
                # Found cards
                cursor.execute("SELECT COUNT(*) FROM found_cards")
                found_count = cursor.fetchone()[0]
                
                # Top accounts
                cursor.execute("""
                    SELECT a.account_name, SUM(ai.quantity) as total
                    FROM accounts a
                    JOIN account_inventory ai ON a.account_id = ai.account_id
                    GROUP BY a.account_id
                    ORDER BY total DESC
                    LIMIT 10
                """)
                top_accounts = cursor.fetchall()
                
                # Top cards
                cursor.execute("""
                    SELECT c.card_name, c.set_code, c.rarity, COUNT(*) as times_found
                    FROM found_cards fc
                    JOIN cards c ON fc.card_id = c.id
                    GROUP BY fc.card_id
                    ORDER BY times_found DESC
                    LIMIT 10
                """)
                top_cards = cursor.fetchall()
                
                stats_text = f"""
╔═══════════════════════════════════════════════════════════════════╗
║                    DATABASE STATISTICS                            ║
╚═══════════════════════════════════════════════════════════════════╝

📦 SETS
   Total Sets: {sets_count}

🎴 CARDS
   Total Cards: {cards_count}
   Cards by Rarity:
"""
                for rarity, count in rarity_counts:
                    stats_text += f"      • {rarity}: {count}\n"
                
                stats_text += f"""
👥 ACCOUNTS
   Total Accounts: {accounts_count}
   Total Cards in Inventory: {total_inventory}

🔍 SCANNING ACTIVITY
   Total Cards Found: {found_count}

📊 TOP 10 ACCOUNTS (by card count):
"""
                for idx, (account_name, total) in enumerate(top_accounts, 1):
                    stats_text += f"   {idx:2d}. {account_name}: {total} cards\n"
                
                stats_text += f"""
🌟 TOP 10 MOST FOUND CARDS:
"""
                for idx, (card_name, set_code, rarity, times) in enumerate(top_cards, 1):
                    stats_text += f"   {idx:2d}. {card_name} ({set_code}) - {rarity} - Found {times}x\n"
                
                self.stats_text.setPlainText(stats_text)
        
        except Exception as e:
            self.stats_text.setPlainText(f"Error loading statistics:\n{str(e)}")
    
    # =========================================================================
    # SETTINGS
    # =========================================================================

    def refresh_collection(self):
        """Avvia il caricamento della collezione in un thread separato."""
        # Verifica che il database esista
        if not os.path.exists(DB_FILENAME):
            self.collection_stats_label.setText("⚠️ Database not found. Please run the scraper first.")
            return
        
        # Disabilita i controlli durante il caricamento
        self.collection_account_combo.setEnabled(False)
        self.collection_stats_label.setText("⏳ Loading collection...")
        
        # Pulisci il container
        while self.collection_container_layout.count() > 1:
            item = self.collection_container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Ottieni account selezionato
        if self.collection_account_combo.count() == 0:
            self.collection_account_combo.addItem("All Accounts")
        
        selected_account = self.collection_account_combo.currentText()
        if not selected_account:
            selected_account = "All Accounts"
        
        # Aggiorna lista account
        try:
            with sqlite3.connect(DB_FILENAME) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT account_name FROM accounts ORDER BY account_name")
                accounts = [row[0] for row in cursor.fetchall()]
                
                current_accounts = [self.collection_account_combo.itemText(i) 
                                   for i in range(self.collection_account_combo.count())]
                
                if set(["All Accounts"] + accounts) != set(current_accounts):
                    self.collection_account_combo.blockSignals(True)
                    self.collection_account_combo.clear()
                    self.collection_account_combo.addItem("All Accounts")
                    self.collection_account_combo.addItems(accounts)
                    if selected_account in ["All Accounts"] + accounts:
                        self.collection_account_combo.setCurrentText(selected_account)
                    self.collection_account_combo.blockSignals(False)
        except Exception as e:
            print(f"Error updating accounts: {e}")
        
        # Avvia il thread di caricamento
        self.collection_loader_thread = CollectionLoaderThread(DB_FILENAME, selected_account)
        self.collection_loader_thread.progress_signal.connect(self.on_collection_progress)
        self.collection_loader_thread.set_ready_signal.connect(self.on_set_ready)
        self.collection_loader_thread.finished_signal.connect(self.on_collection_finished)
        self.collection_loader_thread.error_signal.connect(self.on_collection_error)
        self.collection_loader_thread.start()
    
    def on_collection_progress(self, message):
        """Aggiorna il messaggio di progresso."""
        self.collection_stats_label.setText(f"⏳ {message}")
    
    def on_set_ready(self, frame, set_code, set_name, total_cards, cover_path, account_name, inventory):
        """Chiamato quando un set è pronto per essere visualizzato."""
        try:
            # Crea la sezione del set nel thread principale (per i widget Qt)
            with sqlite3.connect(DB_FILENAME) as conn:
                cursor = conn.cursor()
                set_section = self.create_set_section_fast(
                    set_code, set_name, total_cards, cover_path, account_name, inventory, cursor
                )
                if set_section:
                    self.collection_container_layout.insertWidget(
                        self.collection_container_layout.count() - 1,
                        set_section
                    )
        except Exception as e:
            print(f"Error creating set section: {e}")
    
    def create_set_section_fast(self, set_code, set_name, total_cards, cover_path, account_name, inventory, cursor):
        """Crea una sezione collapsible per un set con cover image."""
        try:
            # Frame principale
            frame = QFrame()
            frame.setFrameShape(QFrame.StyledPanel)
            frame.setStyleSheet("QFrame { border: 1px solid #555; border-radius: 5px; margin: 2px; }")
            
            main_layout = QVBoxLayout(frame)
            main_layout.setContentsMargins(0, 0, 0, 0)
            main_layout.setSpacing(0)
            
            # =========================================================================
            # HEADER CLICCABILE CON COVER
            # =========================================================================
            
            # Container per il header
            header_widget = QWidget()
            header_layout = QHBoxLayout(header_widget)
            header_layout.setContentsMargins(5, 5, 5, 5)
            header_layout.setSpacing(10)
            
            # Arrow button (per expand/collapse)
            arrow_btn = QToolButton()
            arrow_btn.setCheckable(True)
            arrow_btn.setChecked(False)
            arrow_btn.setArrowType(Qt.RightArrow)
            arrow_btn.setFixedSize(20, 20)
            arrow_btn.setStyleSheet("""
                QToolButton {
                    border: none;
                    background: transparent;
                }
            """)
            header_layout.addWidget(arrow_btn)
            
            # Cover image (se esiste)
            if cover_path and os.path.exists(cover_path):
                try:
                    cover_label = QLabel()
                    cover_pixmap = QPixmap(cover_path)
                    if not cover_pixmap.isNull():
                        scaled_cover = cover_pixmap.scaled(60, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        cover_label.setPixmap(scaled_cover)
                        cover_label.setFixedSize(60, 40)
                        cover_label.setStyleSheet("""
                            QLabel {
                                border: 1px solid #555;
                                border-radius: 3px;
                                background-color: #2a2a2a;
                            }
                        """)
                        header_layout.addWidget(cover_label)
                except Exception as e:
                    print(f"Error loading cover image for {set_code}: {e}")
            
            # Calcola statistiche del set
            owned_count = 0
            total_copies = 0
            
            try:
                if account_name == "All Accounts":
                    cursor.execute("""
                        SELECT COUNT(DISTINCT c.id), SUM(ai.quantity)
                        FROM cards c
                        LEFT JOIN account_inventory ai ON c.id = ai.card_id
                        WHERE c.set_code = ? AND ai.quantity > 0
                    """, (set_code,))
                else:
                    cursor.execute("""
                        SELECT COUNT(DISTINCT c.id), SUM(ai.quantity)
                        FROM cards c
                        LEFT JOIN account_inventory ai ON c.id = ai.card_id
                        LEFT JOIN accounts a ON ai.account_id = a.account_id
                        WHERE c.set_code = ? AND ai.quantity > 0 AND a.account_name = ?
                    """, (set_code, account_name))
                
                result = cursor.fetchone()
                owned_count = result[0] if result and result[0] else 0
                total_copies = result[1] if result and result[1] else 0
            except Exception as e:
                print(f"Error calculating stats for {set_code}: {e}")
                owned_count = 0
                total_copies = 0
            
            completion = (owned_count / total_cards * 100) if total_cards and total_cards > 0 else 0
            
            # Text label con nome e stats
            text_label = QLabel(
                f"<b>{set_name}</b> ({set_code}) - {owned_count}/{total_cards} "
                f"({completion:.0f}%) • {total_copies} copies"
            )
            text_label.setStyleSheet("""
                QLabel {
                    color: white;
                    font-size: 14px;
                    padding: 5px;
                }
            """)
            header_layout.addWidget(text_label)
            header_layout.addStretch()
            
            # Rendi tutto il widget cliccabile
            header_widget.setStyleSheet("""
                QWidget {
                    background-color: #353535;
                    border: 1px solid #555;
                    border-radius: 5px;
                }
                QWidget:hover {
                    background-color: #454545;
                }
            """)
            
            main_layout.addWidget(header_widget)
            
            # =========================================================================
            # CONTENT WIDGET (nascosto inizialmente)
            # =========================================================================
            
            content_widget = QWidget()
            content_widget.setVisible(False)
            content_widget.cards_loaded = False  # Flag per lazy loading
            content_layout = QVBoxLayout(content_widget)
            content_layout.setContentsMargins(10, 10, 10, 10)
            
            main_layout.addWidget(content_widget)
            
            # =========================================================================
            # TOGGLE FUNCTION CON LAZY LOADING (✅ CORRETTA)
            # =========================================================================
            
            def toggle_content():
                """Toggle espansione/collasso con lazy loading."""
                try:
                    is_expanded = arrow_btn.isChecked()
                    content_widget.setVisible(is_expanded)
                    arrow_btn.setArrowType(Qt.DownArrow if is_expanded else Qt.RightArrow)
                    
                    # Lazy load delle carte solo la prima volta
                    if is_expanded and not content_widget.cards_loaded:
                        try:
                            self.load_set_cards(content_widget, set_code, inventory, cursor)
                            content_widget.cards_loaded = True
                        except Exception as e:
                            print(f"❌ Error loading cards for {set_code}: {e}")
                            import traceback
                            traceback.print_exc()
                            
                            # Mostra messaggio di errore
                            error_label = QLabel(f"❌ Error loading cards: {str(e)}")
                            error_label.setStyleSheet("QLabel { color: #e74c3c; padding: 10px; }")
                            content_widget.layout().addWidget(error_label)
                            
                except Exception as e:
                    print(f"❌ Error in toggle_content: {e}")
                    import traceback
                    traceback.print_exc()
            
            # ✅ CORRETTO - Collega il signal toggled direttamente
            arrow_btn.toggled.connect(toggle_content)
            
            # ✅ CORRETTO - Gestisci il click sul header widget in modo sicuro
            def header_clicked(event):
                """Gestisce il click sul header per espandere/collassare."""
                try:
                    if event.button() == Qt.LeftButton:
                        arrow_btn.setChecked(not arrow_btn.isChecked())
                        # toggle_content() verrà chiamato automaticamente dal signal toggled
                except Exception as e:
                    print(f"❌ Error in header_clicked: {e}")
                    import traceback
                    traceback.print_exc()
            
            header_widget.mousePressEvent = header_clicked
            
            return frame
            
        except Exception as e:
            print(f"❌ Error creating set section for {set_code}: {e}")
            import traceback
            traceback.print_exc()
            
            # Ritorna frame di errore
            error_frame = QFrame()
            error_layout = QVBoxLayout(error_frame)
            error_label = QLabel(f"❌ Error loading set {set_code}: {str(e)}")
            error_label.setStyleSheet("QLabel { color: #e74c3c; padding: 10px; font-size: 12px; }")
            error_layout.addWidget(error_label)
            return error_frame

    def load_set_cards(self, content_widget, set_code, inventory, cursor):
        """Carica le carte di un set (chiamato solo quando necessario - lazy loading)."""
        try:
            # Query per ottenere info set (inclusa cover)
            cursor.execute("""
                SELECT set_name, total_cards, cover_image_path 
                FROM sets 
                WHERE set_code = ?
            """, (set_code,))
            
            set_info = cursor.fetchone()
            set_name = set_info[0] if set_info else set_code
            total_cards = set_info[1] if set_info and set_info[1] else 0
            cover_path = set_info[2] if set_info and len(set_info) > 2 and set_info[2] else None
            
            # =========================================================================
            # HEADER DEL SET CON COVER IMAGE
            # =========================================================================
            
            set_header = QWidget()
            header_layout = QHBoxLayout(set_header)
            header_layout.setContentsMargins(10, 10, 10, 10)
            header_layout.setSpacing(15)
            
            # Cover image del set
            if cover_path and os.path.exists(cover_path):
                try:
                    cover_label = QLabel()
                    cover_pixmap = QPixmap(cover_path)
                    if not cover_pixmap.isNull():
                        # Scala la cover mantenendo proporzioni
                        scaled_cover = cover_pixmap.scaled(80, 112, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        cover_label.setPixmap(scaled_cover)
                        cover_label.setFixedSize(80, 112)
                        cover_label.setStyleSheet("""
                            QLabel {
                                border: 2px solid #555;
                                border-radius: 5px;
                                background-color: #2a2a2a;
                                padding: 2px;
                            }
                        """)
                        header_layout.addWidget(cover_label)
                except Exception as e:
                    print(f"Error loading cover image: {e}")
            
            # Info widget (nome e stats)
            info_widget = QWidget()
            info_layout = QVBoxLayout(info_widget)
            info_layout.setSpacing(5)
            info_layout.setContentsMargins(0, 0, 0, 0)
            
            # Nome set
            set_label = QLabel(f"📦 {set_name}")
            set_label.setStyleSheet("QLabel { font-size: 16px; font-weight: bold; color: #e0e0e0; }")
            info_layout.addWidget(set_label)
            
            # Set code
            code_label = QLabel(f"Set Code: {set_code}")
            code_label.setStyleSheet("QLabel { color: #888; font-size: 11px; }")
            info_layout.addWidget(code_label)
            
            # ✅ CORRETTO - Calcola stats in modo sicuro
            owned_count = 0
            total_owned_copies = 0
            
            try:
                # Recupera tutte le carte di questo set
                cursor.execute("""
                    SELECT id
                    FROM cards
                    WHERE set_code = ?
                """, (set_code,))
                
                set_card_ids = set(row[0] for row in cursor.fetchall())
                
                # Conta carte possedute in questo set
                owned_count = sum(1 for card_id, qty in inventory.items() 
                                if qty > 0 and card_id in set_card_ids)
                
                # Totale copie possedute
                total_owned_copies = sum(qty for card_id, qty in inventory.items() 
                                    if card_id in set_card_ids)
                
            except Exception as e:
                print(f"Error calculating owned cards for {set_code}: {e}")
                owned_count = 0
                total_owned_copies = 0
            
            # Completion percentage
            completion = (owned_count / total_cards * 100) if total_cards > 0 else 0
            
            # Stats label
            stats_label = QLabel(
                f"Owned: {owned_count}/{total_cards} cards ({completion:.1f}%)\n"
                f"Total Copies: {total_owned_copies}"
            )
            stats_label.setStyleSheet("QLabel { color: #3498db; font-size: 12px; }")
            info_layout.addWidget(stats_label)
            
            info_layout.addStretch()
            header_layout.addWidget(info_widget)
            header_layout.addStretch()
            
            content_widget.layout().addWidget(set_header)
            
            # Separator line
            separator = QFrame()
            separator.setFrameShape(QFrame.HLine)
            separator.setFrameShadow(QFrame.Sunken)
            separator.setStyleSheet("QFrame { background-color: #555; max-height: 1px; }")
            content_widget.layout().addWidget(separator)
            
            # =========================================================================
            # GRID DELLE CARTE
            # =========================================================================
            
            try:
                # Query per ottenere le carte del set
                cursor.execute("""
                    SELECT id, card_number, card_name, rarity, local_image_path
                    FROM cards
                    WHERE set_code = ?
                    ORDER BY CAST(card_number AS INTEGER)
                """, (set_code,))
                
                cards = cursor.fetchall()
                
                # Recupera wishlist
                cursor.execute("SELECT card_id FROM wishlist")
                wishlist_ids = set(row[0] for row in cursor.fetchall())
                
                # Container per le carte
                cards_widget = QWidget()
                cards_grid = QGridLayout(cards_widget)
                cards_grid.setSpacing(10)
                cards_grid.setContentsMargins(10, 10, 10, 10)
                
                # Crea widget per ogni carta
                row, col = 0, 0
                max_cols = 6
                
                for card_id, card_number, card_name, rarity, image_path in cards:
                    try:
                        quantity = inventory.get(card_id, 0)
                        is_wishlisted = card_id in wishlist_ids
                        
                        card_data = {
                            'id': card_id,
                            'card_number': card_number,
                            'card_name': card_name,
                            'rarity': rarity,
                            'local_image_path': image_path,
                            'set_code': set_code
                        }
                        
                        card_widget = CardWidget(card_data, quantity, is_wishlisted)
                        card_widget.wishlist_changed.connect(self.on_wishlist_changed)
                        
                        cards_grid.addWidget(card_widget, row, col)
                        
                        col += 1
                        if col >= max_cols:
                            col = 0
                            row += 1
                            
                    except Exception as e:
                        print(f"Error creating widget for card {card_id}: {e}")
                        continue
                
                content_widget.layout().addWidget(cards_widget)
                
            except Exception as e:
                print(f"❌ Error fetching cards for {set_code}: {e}")
                import traceback
                traceback.print_exc()
                
                error_label = QLabel(f"❌ Error loading cards: {str(e)}")
                error_label.setStyleSheet("QLabel { color: #e74c3c; padding: 10px; }")
                content_widget.layout().addWidget(error_label)
            
        except Exception as e:
            print(f"❌ Error loading cards for set {set_code}: {e}")
            import traceback
            traceback.print_exc()
            
            error_label = QLabel(f"❌ Error loading set data: {str(e)}")
            error_label.setStyleSheet("QLabel { color: #e74c3c; padding: 10px; }")
            content_widget.layout().addWidget(error_label)


    def on_collection_finished(self, total_owned, total_cards):
        """Chiamato quando il caricamento è completo."""
        completion = (total_owned / total_cards * 100) if total_cards > 0 else 0
        selected_account = self.collection_account_combo.currentText()
        self.collection_stats_label.setText(
            f"📊 Total: {total_owned}/{total_cards} cards ({completion:.1f}% complete) | "
            f"Account: {selected_account}"
        )
        self.collection_account_combo.setEnabled(True)
    
    def on_collection_error(self, error_msg):
        """Chiamato in caso di errore."""
        self.collection_stats_label.setText(f"❌ {error_msg}")
        self.collection_account_combo.setEnabled(True)

    def on_tab_changed(self, index):
        """Chiamato quando l'utente cambia tab."""
        # Tab Collection è il 3° (index 2)
        if index == 2 and not self.collection_loaded:
            # Carica la collezione solo la prima volta
            if os.path.exists(DB_FILENAME):
                self.collection_loaded = True
                self.refresh_collection()
            else:
                self.collection_stats_label.setText("⚠️ Database not found. Please run the scraper first.")


    def save_settings(self):
        """Salva le impostazioni."""
        settings = {
            'token': self.token_input.text(),
            'channel_id': self.channel_input.text(),
            'autostart': self.autostart_cb.isChecked(),
            'minimize_to_tray': self.minimize_tray_cb.isChecked(),
            'dark_theme': self.theme_combo.isChecked()
        }
        try:
            with open('settings.json', 'w') as f:
                json.dump(settings, f)
        except:
            pass
    
    def load_settings(self):
        """Carica le impostazioni."""
        try:
            if os.path.exists('settings.json'):
                with open('settings.json', 'r') as f:
                    settings = json.load(f)
                self.token_input.setText(settings.get('token', ''))
                self.channel_input.setText(settings.get('channel_id', ''))
                self.autostart_cb.setChecked(settings.get('autostart', False))
                self.minimize_tray_cb.setChecked(settings.get('minimize_to_tray', True))
                self.theme_combo.setChecked(settings.get('dark_theme', True))
                
                # Auto-start bot if enabled
                if settings.get('autostart', False):
                    QTimer.singleShot(1000, self.start_bot)
        except:
            pass
    
    # =========================================================================
    # WINDOW EVENTS
    # =========================================================================
    
    def closeEvent(self, event):
        """Gestisce la chiusura della finestra."""
        # Se "Minimize to tray" è abilitato, nascondi invece di chiudere
        if hasattr(self, 'minimize_tray_cb') and self.minimize_tray_cb.isChecked():
            event.ignore()
            self.hide()
            if hasattr(self, 'tray_icon'):
                self.tray_icon.showMessage(
                    "TCGP Team Rocket Tool",
                    "Application minimized to tray. Right-click the icon to quit.",
                    QSystemTrayIcon.Information,
                    2000
                )
        else:
            # Altrimenti, chiudi completamente
            self.quit_application()
    # =========================================================================
    # BACKGROUND IMAGE
    # =========================================================================
    
    def set_background_image(self, image_path):
        """Imposta un'immagine di sfondo con overlay applicato direttamente."""
        try:
            if not os.path.exists(image_path):
                print(f"⚠️ Background image not found: {image_path}")
                return
            
            from PIL import Image, ImageDraw, ImageEnhance
            
            # Carica l'immagine con PIL
            img = Image.open(image_path)
            
            # Ridimensiona se troppo grande (per performance)
            max_size = (1920, 1080)
            img.thumbnail(max_size, Image.LANCZOS)
            
            # Converti in RGBA per l'overlay
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # Crea overlay grigio scuro semitrasparente
            overlay = Image.new('RGBA', img.size, (30, 30, 30, 180))  # RGB + Alpha (0-255)
            
            # Combina immagine e overlay
            img_with_overlay = Image.alpha_composite(img, overlay)
            
            # Salva temporaneamente
            temp_bg_path = os.path.join('gui/background_with_overlay.png')
            img_with_overlay.save(temp_bg_path, 'PNG')
            
            # Usa come sfondo
            self.background_label = QLabel(self)
            self.background_label.setScaledContents(False)
            self.background_label.lower()
            
            self.original_background = QPixmap(temp_bg_path)
            self.update_background_size()
            
            print(f"✅ Background image with overlay set: {image_path}")
        except Exception as e:
            print(f"❌ Error setting background: {e}")
            import traceback
            traceback.print_exc()
    
    def update_background_size(self):
        """Aggiorna le dimensioni dell'immagine di sfondo."""
        if hasattr(self, 'background_label') and hasattr(self, 'original_background'):
            window_size = self.size()
            
            # Scala in modalità cover
            scaled_pixmap = self.original_background.scaled(
                window_size,
                Qt.KeepAspectRatioByExpanding,
                Qt.SmoothTransformation
            )
            
            self.background_label.setPixmap(scaled_pixmap)
            
            # Centra l'immagine
            x_offset = (scaled_pixmap.width() - window_size.width()) // 2
            y_offset = (scaled_pixmap.height() - window_size.height()) // 2
            self.background_label.setGeometry(
                -x_offset, -y_offset,
                scaled_pixmap.width(), scaled_pixmap.height()
            )


    
    def resizeEvent(self, event):
        """Chiamato quando la finestra viene ridimensionata."""
        super().resizeEvent(event)
        self.update_background_size()

# =========================================================================
# 🎨 STYLING AND MAIN FUNCTION
# =========================================================================

def apply_dark_theme(app):
    """Applica il tema scuro all'applicazione."""
    app.setStyle("Fusion")
    
    darkpalette = QPalette()
    darkpalette.setColor(QPalette.Window, QColor(53, 53, 53))
    darkpalette.setColor(QPalette.WindowText, Qt.white)
    darkpalette.setColor(QPalette.Base, QColor(35, 35, 35))
    darkpalette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    darkpalette.setColor(QPalette.ToolTipBase, QColor(25, 25, 25))
    darkpalette.setColor(QPalette.ToolTipText, Qt.white)
    darkpalette.setColor(QPalette.Text, Qt.white)
    darkpalette.setColor(QPalette.Button, QColor(53, 53, 53))
    darkpalette.setColor(QPalette.ButtonText, Qt.white)
    darkpalette.setColor(QPalette.BrightText, Qt.red)
    
    # ⬇️ CAMBIA QUESTI DA BLU A GIALLO ⬇️
    darkpalette.setColor(QPalette.Link, QColor(243, 156, 18))  # #f39c12 - Giallo oro
    darkpalette.setColor(QPalette.Highlight, QColor(243, 156, 18))  # #f39c12 - Giallo oro
    darkpalette.setColor(QPalette.HighlightedText, Qt.black)
    
    darkpalette.setColor(QPalette.Disabled, QPalette.Text, Qt.darkGray)
    darkpalette.setColor(QPalette.Disabled, QPalette.ButtonText, Qt.darkGray)
    
    app.setPalette(darkpalette)
    
    # Additional stylesheet
    app.setStyleSheet("""
        QToolTip { 
            color: #ffffff; 
            background-color: #2a82da; 
            border: 1px solid white; 
        }
        QGroupBox {
            border: 2px solid #555;
            border-radius: 5px;
            margin-top: 1ex;
            padding: 15px;  /* ⬅️ Padding interno del box */
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 5px 10px;  /* ⬅️ Padding del titolo (verticale, orizzontale) */
            color: #f39c12;
        }

        QTabWidget::pane {
            border: 1px solid #444;
            border-radius: 3px;
        }
        QTabBar::tab {
            background: #353535;
            border: 1px solid #444;
            padding: 8px 16px;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background: #2a82da;
            color: white;
        }
        QTabBar::tab:hover {
            background: #454545;
        }
        QPushButton {
            border: 1px solid #555;
            border-radius: 3px;
            padding: 5px 15px;
            background-color: #454545;
        }
        QPushButton:hover {
            background-color: #555;
        }
        QPushButton:pressed {
            background-color: #353535;
        }
        QPushButton:disabled {
            background-color: #2a2a2a;
            color: #666;
        }
        QLineEdit {
            border: 1px solid #555;
            border-radius: 3px;
            padding: 5px;
            background-color: #2a2a2a;
        }
        QLineEdit:focus {
            border: 1px solid #2a82da;
        }
        QTextEdit {
            border: 1px solid #555;
            border-radius: 3px;
            background-color: #2a2a2a;
        }
        QTableWidget {
            border: 1px solid #555;
            gridline-color: #555;
            background-color: #2a2a2a;
        }
        QTableWidget::item:selected {
            background-color: #2a82da;
        }
        QHeaderView::section {
            background-color: #454545;
            padding: 5px;
            border: 1px solid #555;
            font-weight: bold;
        }
        QProgressBar {
            border: 1px solid #555;
            border-radius: 3px;
            text-align: center;
            background-color: #2a2a2a;
        }
        QProgressBar::chunk {
            background-color: #f39c12;
            border-radius: 2px;
        }
        QCheckBox {
            spacing: 5px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
        }
        QCheckBox::indicator:unchecked {
            border: 1px solid #555;
            background-color: #2a2a2a;
            border-radius: 3px;
        }
        QCheckBox::indicator:checked {
            border: 1px solid #2a82da;
            background-color: #2a82da;
            border-radius: 3px;
        }
        QSpinBox {
            border: 1px solid #555;
            border-radius: 3px;
            padding: 3px;
            background-color: #2a2a2a;
        }
        QScrollBar:vertical {
            border: none;
            background: #2a2a2a;
            width: 14px;
            margin: 0;
        }
        QScrollBar::handle:vertical {
            background: #555;
            min-height: 20px;
            border-radius: 7px;
        }
        QScrollBar::handle:vertical:hover {
            background: #666;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        QScrollBar:horizontal {
            border: none;
            background: #2a2a2a;
            height: 14px;
            margin: 0;
        }
        QScrollBar::handle:horizontal {
            background: #555;
            min-width: 20px;
            border-radius: 7px;
        }
        QScrollBar::handle:horizontal:hover {
            background: #666;
        }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            width: 0px;
        }
    """)

def main():
    """Funzione principale."""
    # Enable High DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    
    # Set application info
    app.setApplicationName("TCGP - Team Rocket Tool")
    app.setApplicationVersion("2.0.0")
    app.setOrganizationName("TCG Tools")
    
    # Set style
    app.setStyle('Fusion')
    
    # Apply dark theme
    apply_dark_theme(app)
    
    # Set font
    font = QFont("Segoe UI", 9)
    app.setFont(font)
    
    # Create main window
    window = MainWindow()
    window.show()
    
    # Show welcome message
    welcome_msg = """
    ╔══════════════════════════════════════════════════════╗
    ║      TCGP Team Rocket Tool v1.27                ║
    ║                                                      ║
    ║  Features:                                           ║
    ║  • Discord bot for trade monitoring                  ║
    ║  • Automatic card recognition (OpenCV)               ║
    ║  • Database setup and management                     ║
    ║  • Statistics and export tools                       ║
    ║                                                      ║
    ║  Quick Start:                                        ║
    ║  1. Go to 'Database Setup' and run the scraper      ║
    ║  2. Configure your bot token and channel ID          ║
    ║  3. Start the Discord bot                            ║
    ║                                                      ║
    ║  Enjoy! 🎴                                           ║
    ╚══════════════════════════════════════════════════════╝
    """
    
    # Check if database exists
    if not os.path.exists(DB_FILENAME):
        window.append_bot_log("⚠️ Database not found!")
        window.append_bot_log("Please go to 'Database Setup' tab and run the scraper first.")
        QMessageBox.information(
            window, 
            "First Run", 
            "Welcome to TCGP Team Rocket Tool!\n\n"
            "Please go to the 'Database Setup' tab and run the scraper to download card data.\n\n"
            "This is required before using the Discord bot."
        )
        # Switch to database tab
        window.tabs.setCurrentIndex(3)
    else:
        # Initialize database manager to check for migrations
        db_manager = DatabaseManager(log_callback=window.append_bot_log)
        db_manager.setup_database()
        db_manager.close()
        
        window.append_bot_log("✅ Database loaded successfully")
        window.append_bot_log("Ready to start monitoring!")
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
