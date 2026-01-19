# -*- coding: utf-8 -*-
r"""
config.py - Configurazione globale e costanti dell'applicazione
Pokemon TCG Pocket - TeamRocket Tool

✅ TUTTI I DATI SALVATI IN: C:\Users\{USER}\AppData\Roaming\TCGPTeamRocketTool\
"""


import os
import sys
import re
import json
from dotenv import load_dotenv


from core.translations import t

# Carica variabili d'ambiente
load_dotenv()

# ============================================================================
# PERCORSI E RISORSE - DEFINIZIONE APPDATA
# ============================================================================
    
def create_default_settings():
    """Crea il file settings.json di default se non esiste."""
    
    if not os.path.exists(SETTINGS_FILE):
        print(t("config.settings_not_found", file=SETTINGS_FILE))
        
        default_settings = {
            "token": "",
            "selected_channel_ids": [], # <-- NUOVA CHIAVE: lista di ID interi
            "channel_configs": [], # <-- Mantenuta per retrocompatibilità in load_settings
            "autostart": False,
            "minimize_to_tray": True,
            "dark_theme": True,
            "language": "en",
            "selected_rarities": [],
            "last_updated": "",
            "bot_folder": "",        
            "discord_notifications": False,
            "notification_channel": ""
        }
        
        try:
            # Assicurati che la cartella esista
            os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
            
            # Scrivi il file JSON
            with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(default_settings, f, indent=4, ensure_ascii=False)
                
        except Exception as e:
            print(t("config.settings_creation_error", e=e))
            # L'app potrebbe non avviarsi correttamente, ma è meglio che crashare qui.



#def get_resource_path(relative_path):
#    """
#    Ottiene il percorso assoluto di una risorsa (per i file inclusi con PyInstaller).
#    """
#    if getattr(sys, 'frozen', False):
#        base_path = sys._MEIPASS
#    else:
#        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#    return os.path.join(base_path, relative_path)

def get_resource_path(relative_path):
    """
    Ottiene il percorso assoluto di una risorsa (per i file inclusi con PyInstaller).
    """
    if getattr(sys, 'frozen', False):
        # Se l'app è compilata con PyInstaller
        base_path = sys._MEIPASS
    else:
        # Durante lo sviluppo: cartella che contiene config.py
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    return os.path.join(base_path, relative_path)


# ✅ DIRECTORY PRINCIPALE APPDATA
APP_DATA_DIR = os.path.join(
    os.path.expanduser("~"),
    "AppData",
    "Roaming",
    "TCGPTeamRocketTool"
)

def get_app_data_path(filename=""):
    """
    Ottiene il percorso nella directory AppData dell'utente.
    """
    os.makedirs(APP_DATA_DIR, exist_ok=True)
    if filename:
        return os.path.join(APP_DATA_DIR, filename)
    return APP_DATA_DIR



# ============================================================================
# PERCORSI PRINCIPALI - TUTTI IN APPDATA
# ============================================================================


# ==========================================================================
# ⚠️ MODIFICA: Ora anche le risorse grafiche puntano ad APPDATA
# ==========================================================================
ICON_PATH = get_resource_path("gui/icon.ico")
BACKGROUND_PATH = get_resource_path("gui/background.png")
RARITY_ICONS_DIR = get_resource_path("gui/")
# ==========================================================================


# ✅ PATH DATI - TUTTO IN APPDATA
SETTINGS_FILE = get_app_data_path("settings.json")
ACCOUNTS_DIR = get_app_data_path("Accounts")
TCG_IMAGES_DIR = get_app_data_path("tcg_images")
DB_FILENAME = get_app_data_path("tcg_pocket.db")
PROXIES_FILE = get_app_data_path("proxies.txt")
LOG_DIR = get_app_data_path("logs")
CACHE_DIR = get_app_data_path("cache")
CLOUDFLARED_PATH = get_app_data_path("cloudflared.exe")
LOG_FILENAME = get_app_data_path("trade_log.json")

# Crea le directory necessarie (inclusa la 'gui' in AppData)
os.makedirs(ACCOUNTS_DIR, exist_ok=True)
#os.makedirs(TCG_IMAGES_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(get_app_data_path("gui"), exist_ok=True) # <-- Crea la cartella gui in AppData


# ============================================================================
# CONFIGURAZIONI DISCORD E SCANSIONE
# ============================================================================


SEARCH_STRING = "Tradeable cards"
ACCOUNT_NAME_REGEX = r'by\s+(.*?)\s+in instance:'
ACCOUNT_NAME_PATTERN = re.compile(ACCOUNT_NAME_REGEX)


# ============================================================================
# PARAMETRI DI PERFORMANCE E DOWNLOAD
# ============================================================================


BATCH_SIZE = 500
CHUNK_SIZE = 1024 * 128
MAX_RETRIES = 3
RETRY_DELAY = 1
MAX_CONCURRENT_DOWNLOADS = 100
SAVE_INTERVAL = 1
MAX_WORKERS = os.cpu_count() or 8
CARD_RECOGNITION_WORKERS = 8


# ============================================================================
# CONFIGURAZIONI WEB SCRAPING
# ============================================================================


PROXY_API_URL = "https://api.proxyscrape.com/v4/free-proxy-list/get?request=get_proxies&skip=0&proxy_format=protocolipport&format=json&limit=1000"


# ============================================================================
# CONFIGURAZIONI IMAGE PROCESSING E CARD RECOGNITION
# ============================================================================


TEMPLATE_DOWNSCALE_FACTOR = 0.20
SIMILARITY_THRESHOLD = 0.75
TEMPLATE_CROP_BOX = (25, 50, 340, 240)
SOURCE_ROI_BOXES = [(40, 15, 200, 60), (40, 130, 200, 175)]
HASH_SIZE = 24


# Color detection per layout
TARGET_GRAY_BGR = (242, 232, 222)
COLOR_TOLERANCE = 3
TOP_ROW_CHECK_BOX = (119, 27, 123, 40)
BOTTOM_ROW_CHECK_BOX = (119, 134, 123, 147)


# Layout crop boxes per diverse orientazioni
LAYOUT_CROP_BOXES = {
    'standard': (25, 50, 340, 240),
    'alternate': (40, 15, 200, 60),
    'bottom': (40, 130, 200, 175)
}


# Threshold di matching per diversi tipi di card
MATCHING_THRESHOLDS = {
    'exact': 0.95,
    'high': 0.85,
    'medium': 0.65,
    'low': 0.50
}


# ============================================================================
# RARITÀ E DATI CARD
# ============================================================================

# ✅ Usa il percorso nel RARITY_DATA
# ============================================================================
# RARITÀ E DATI CARD
# ============================================================================

RARITY_DATA = {
    'Common': os.path.join("gui", "diamond_1.png"),
    'Uncommon': os.path.join("gui", "diamond_2.png"),
    'Rare': os.path.join("gui", "diamond_3.png"),
    'Double Rare': os.path.join("gui", "diamond_4.png"),
    'Art Rare': os.path.join("gui", "star_1.png"),
    'Super Rare': os.path.join("gui", "star_2.png"),
    'Special Art Rare': os.path.join("gui", "rainbow_star.png"),
    'Immersive Rare': os.path.join("gui", "star_3.png"),
    'Shiny': os.path.join("gui", "shiny_1.png"),
    'Shiny Super Rare': os.path.join("gui", "shiny_2.png"),
    'Crown Rare': os.path.join("gui", "crown.png"),
}


SELECTED_RARITIES = list(RARITY_DATA.keys())


# ============================================================================
# SICUREZZA E AUTENTICAZIONE
# ============================================================================


CLOUDFLARE_PASSWORD = os.getenv('CLOUDFLARE_PASSWORD', '')


# ============================================================================
# COLLECTION LOADING PERFORMANCE
# ============================================================================


COLLECTION_BATCH_SIZE = 50
CARD_PREVIEW_SIZE = (150, 200)
COLLECTION_CACHE_SIZE = 500
LAZY_LOAD_ENABLED = True


# ============================================================================
# RILEVAMENTO SISTEMA OPERATIVO
# ============================================================================


try:
    import ctypes
    from windows_toasts import WindowsToastNotifier
    WINDOWS_TOAST_AVAILABLE = True
except ImportError:
    WINDOWS_TOAST_AVAILABLE = False


# ============================================================================
# SCHEMA DATABASE
# ============================================================================


TABLES_SCHEMA = {
    'sets': {
        'id': 'INTEGER PRIMARY KEY',
        'code': 'TEXT UNIQUE NOT NULL',
        'name': 'TEXT NOT NULL',
        'release_date': 'TEXT',
        'cover_image_url': 'TEXT',
        'cover_image_path': 'TEXT'
    },
    'cards': {
        'id': 'INTEGER PRIMARY KEY',
        'set_code': 'TEXT NOT NULL',
        'card_id': 'TEXT NOT NULL',
        'name': 'TEXT',
        'rarity': 'TEXT',
        'image_url': 'TEXT',
        'image_path': 'TEXT',
        'hash': 'TEXT'
    },
    'ownership': {
        'id': 'INTEGER PRIMARY KEY',
        'card_id': 'INTEGER NOT NULL',
        'account_name': 'TEXT NOT NULL',
        'quantity': 'INTEGER DEFAULT 0',
        'timestamp': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
    },
    'wishlist': {
        'id': 'INTEGER PRIMARY KEY',
        'card_id': 'INTEGER NOT NULL',
        'account_name': 'TEXT NOT NULL',
        'added_date': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
    }
}


# ============================================================================
# LINGUE E LOCALIZZAZIONE
# ============================================================================


SUPPORTED_LANGUAGES = {
    'it': 'Italiano',
    'en': 'English',
    'es': 'Español',
    'fr': 'Français',
    'de': 'Deutsch'
}


DEFAULT_LANGUAGE = 'en'


# ============================================================================
# CONFIGURAZIONI FLASK
# ============================================================================


FLASK_HOST = '127.0.0.1'
FLASK_PORT = 5000
FLASK_DEBUG = False


# ============================================================================
# CONFIGURAZIONI CLOUDFLARE
# ============================================================================


CLOUDFLARE_TUNNEL_NAME = 'tcg-pocket-tool'
CLOUDFLARE_METRICS_PORT = 8000


# ============================================================================
# TIMEOUTS E LIMITI
# ============================================================================


REQUEST_TIMEOUT = 30
DISCORD_SCAN_TIMEOUT = 300
DOWNLOAD_TIMEOUT = 60
DATABASE_TIMEOUT = 10


# ============================================================================
# LOGGING
# ============================================================================


LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


# ============================================================================
# FUNZIONI HELPER PER SETTINGS
# ============================================================================


def load_settings():
    """
    Carica i settings dal JSON principale in AppData.
    Ritorna un dizionario con tutte le impostazioni.
    """
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(t("config.settings_loading_warning", e=e))
    
    # Valori di default
    return {
        'token': '',
        'channel_id': '',
        'autostart': False,
        'minimize_to_tray': True,
        'dark_theme': True,
        'language': DEFAULT_LANGUAGE,
        'selected_rarities': SELECTED_RARITIES,
    }


def save_settings(settings):
    """
    Salva i settings nel JSON principale in AppData.
    Ritorna True se salvato con successo.
    """
    try:
        os.makedirs(APP_DATA_DIR, exist_ok=True)
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=4)
        return True
    except Exception as e:
        print(t("config.settings_saving_error", e=e))
        return False


# ============================================================================
# PRINT INFO PERCORSI (PER DEBUG)
# ============================================================================


def print_paths_info():
    """Stampa i percorsi per debug."""
    print("\n" + "="*70)
    print(t("config.paths_info_title"))
    print("="*70)
    print(f"{t('config.app_data_dir')}      {APP_DATA_DIR}")
    print(f"{t('config.settings_file')}     {SETTINGS_FILE}")
    print(f"{t('config.database')}          {DB_FILENAME}")
    print(f"{t('config.accounts_dir')}      {ACCOUNTS_DIR}")
#    print(f"TCG Images Dir:    {TCG_IMAGES_DIR}")
    print(f"{t('config.cache_dir')}         {CACHE_DIR}")
    print(f"{t('config.logs_dir')}          {LOG_DIR}")
    print("="*70 + "\n")