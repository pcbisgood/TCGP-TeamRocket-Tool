"""ui_main_window.py - Finestra principale PyQt5"""

# =========================================================================
# 1. IMPORT LIBRERIA STANDARD (Built-in Python)
# =========================================================================
import os
import io
import sys
import json
import time
import queue
import base64
import sqlite3
import asyncio
import subprocess
import webbrowser
import urllib.request
from datetime import datetime
from threading import Lock
from typing import Optional, Dict, List, Callable

# =========================================================================
# 2. IMPORT LIBRERIE DI TERZE PARTI (Pip install)
# =========================================================================
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

# PyQt5
from PyQt5.QtCore import (
    Qt,
    pyqtSignal,
    QTimer,
    QSize,
    QThread,
    QThreadPool,
    QRunnable,
    QObject,
    pyqtSlot,
    QUrl,
)
from PyQt5.QtGui import (
    QPixmap,
    QIcon,
    QFont,
    QColor,
    QPalette,
    QBrush,
    qGray,
    qRgb,
    QPainter,
    QLinearGradient,
)
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QScrollArea,
    QFrame,
    QComboBox,
    QCheckBox,
    QFileDialog,
    QMessageBox,
    QApplication,
    QTableWidget,
    QTableWidgetItem,
    QDialog,
    QGroupBox,
    QLineEdit,
    QProgressBar,
    QTextEdit,
    QTextBrowser,
    QSystemTrayIcon,
    QStyle,
    QMenu,
    QToolButton,
    QSizePolicy,
    QGraphicsOpacityEffect,
    QDialogButtonBox,
)

# Optional: Windows Toasts
from .notification_manager import send_toast_notification
from .accounts_tab import AccountsTab

# =========================================================================
# 3. IMPORT MODULI LOCALI (Applicazione)
# =========================================================================

# Import configurazione
from config import (
    ICON_PATH,
    BACKGROUND_PATH,
    DEFAULT_LANGUAGE,
    get_app_data_path,
    get_resource_path,
    ACCOUNTS_DIR,
    DB_FILENAME,
    RARITY_DATA,
    SELECTED_RARITIES,
    TOP_ROW_CHECK_BOX,
    BOTTOM_ROW_CHECK_BOX,
    TARGET_GRAY_BGR,
    COLOR_TOLERANCE,
)

# Import traduzioni
from .translations import t, set_language, get_language

# Import moduli Core (Database, Cache, Processing)
from .database import DatabaseManager
from .image_cache import LRUImageCache as ImageCache

# Import componenti UI (Tabs)
from .collection_tab import CollectionTab
from .scraper_tab import ScraperTab
from .cards_found_tab import CardsFoundTab

# Import UI widgets
from .ui_widgets import CardWidget, CardDetailsDialog, ImageViewerDialog

# Import manager e helper (Logica)
from .wishlist_manager import WishlistManager
from .cloudflare import CloudflarePasswordDialog, CloudflareTunnelThread
from .flask_server import FlaskServerThread

# Import threads
from .threads import (
    DiscordBotThread,
    ScraperThread,
    CollectionLoaderThread,
    DiscordChannelLoaderThread,
)

# =========================================================================
# 🖥️ GUI APPLICATION - MAIN WINDOW
# =========================================================================

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QClipboard
from PyQt5.QtCore import QTimer, pyqtSlot
import weakref


def send_discord_bot_message(card_data: dict) -> bool:
    """
    Invia notifica Discord con layout migliorato ed estetica accattivante.
    
    Layout Migliorato:
    - Colore embed dinamico basato sulla rarità
    - Carta grande come immagine principale
    - Set cover come thumbnail
    - Icona rarità nell'author dell'embed
    - Layout compatto e leggibile
    """
    try:
        import json
        import threading
        import os
        import io
        from datetime import datetime
        
        # Carica settings e config
        from config import get_app_data_path, get_resource_path, RARITY_DATA
        
        settings_path = get_app_data_path("settings.json")
        
        if not os.path.exists(settings_path):
            #print("⚠️ File settings.json non trovato")
            return False
        
        with open(settings_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)
        
        # Leggi bot token e channel ID
        bot_token = settings.get('token', '')
        channel_id = settings.get('notification_channel', '')
        
        if not bot_token or not channel_id:
            #print("⚠️ Discord bot token o channel ID non configurati")
            return False
        
        try:
            channel_id = int(channel_id)
        except (ValueError, TypeError):
            #print("❌ Discord channel ID non è un numero valido")
            return False
        
        # ============================================================================
        # ESTRAI DATI DALLA CARTA
        # ============================================================================
        
        card_name = card_data.get('card_name', 'Unknown')
        card_number = card_data.get('card_number', '?')
        set_code = card_data.get('set_code', '?')
        rarity = card_data.get('rarity', '?')
        account_name = card_data.get('account_name', 'Unknown Account')
        
        # Blob dal database
        thumbnail_blob = card_data.get('thumbnail_blob', None)
        
        # ✅ RECUPERA IL BLOB DEL SET DAL DATABASE
        set_cover_blob = None
        try:
            from database import DatabaseManager
            db = DatabaseManager()
            db.connect()
            
            # Query per ottenere il blob del set cover
            db.cursor.execute(
                "SELECT cover_image_blob FROM sets WHERE set_code = ?",
                (set_code,)
            )
            result = db.cursor.fetchone()
            
            if result and result[0]:
                set_cover_blob = result[0]
                #print(f"   ✅ Set cover blob recuperato per {set_code}")
            #else:
            #    print(f"   ⚠️ Set cover blob non trovato per {set_code}")
            
            db.close()
            
        except Exception as e:
            print(f"   ⚠️ Errore recupero set cover blob: {e}")
        
        
        # ============================================================================
        # MAPPA COLORI PER RARITÀ
        # ============================================================================
    
        
        # Scegli il colore basato sulla rarità
        embed_color =  0xFFD700  # Default: oro
    
        
        # Crea il task async in un thread separato
        def run_async():
            import asyncio
            
            async def send():
                try:
                    import discord
                    
                    intents = discord.Intents.default()
                    bot = discord.Client(intents=intents)
                    
                    @bot.event
                    async def on_ready():
                        #print(f"🤖 Bot connesso come: {bot.user}")
                        
                        channel = bot.get_channel(channel_id)
                        
                        if channel is None:
                            #print(f"❌ Canale {channel_id} non trovato")
                            await bot.close()
                            return
                        
                        # ============================================================================
                        # CREA L'EMBED - FORMATO COMPATTO
                        # ============================================================================
                        
                        embed = discord.Embed(
                            title=f"{card_name}",
                            description="",
                            color=embed_color
                        )
                        
                        # ============================================================================
                        # PREPARA I FILE DA INVIARE
                        # ============================================================================
                        
                        files = []
                        
                        # 1️⃣ CARD IMAGE GRANDE (immagine principale sotto il titolo)
                        if thumbnail_blob:
                            try:
                                files.append(
                                    discord.File(
                                        io.BytesIO(thumbnail_blob),
                                        filename="card_image.png"
                                    )
                                )
                                embed.set_image(url="attachment://card_image.png")
                                #print("   ✅ Immagine carta aggiunta")
                            except Exception as e:
                                print(f"   ⚠️ Errore card image: {e}")
                        
                        # 2️⃣ ICONA RARITÀ (thumbnail a destra)
                        if rarity in RARITY_DATA:
                            rarity_icon_relative_path = RARITY_DATA[rarity]
                            rarity_icon_path = get_resource_path(rarity_icon_relative_path)
                            
                            if os.path.exists(rarity_icon_path):
                                try:
                                    with open(rarity_icon_path, 'rb') as f:
                                        rarity_image_data = f.read()
                                    
                                    files.append(
                                        discord.File(
                                            io.BytesIO(rarity_image_data),
                                            filename="rarity_icon.png"
                                        )
                                    )
                                    embed.set_thumbnail(url="attachment://rarity_icon.png")
                                    #print("   ✅ Icona rarità aggiunta")
                                except Exception as e:
                                    print(f"   ⚠️ Errore icona rarità: {e}")
                        
                        # 3️⃣ SET COVER (come author icon)
                        if set_cover_blob:
                            try:
                                files.append(
                                    discord.File(
                                        io.BytesIO(set_cover_blob),
                                        filename="set_cover.png"
                                    )
                                )
                                # Usa il set cover come icona dell'author
                                embed.set_author(
                                    name=f"{set_code}",
                                    icon_url="attachment://set_cover.png"
                                )
                                #print("   ✅ Cover set aggiunta")
                            except Exception as e:
                                print(f"   ⚠️ Errore set cover: {e}")
                        else:
                            # Se non c'è set cover, usa solo il testo
                            embed.set_author(name=f"{set_code}")
                        
                        # ============================================================================
                        # FOOTER CON ACCOUNT
                        # ============================================================================
                        
                        embed.set_footer(
                            text=f"👤 {account_name}"
                        )
                        embed.timestamp = datetime.now()
                        
                        # ============================================================================
                        # INVIA IL MESSAGGIO
                        # ============================================================================
                        
                        try:
                            if files:
                                await channel.send(embed=embed, files=files)
                                #print(f"✅ Notifica Discord inviata! ({len(files)} allegati)")
                            else:
                                await channel.send(embed=embed)
                                #print("✅ Notifica Discord inviata (senza immagini)")
                        except Exception as e:
                            #print(f"❌ Errore invio messaggio: {e}")
                            import traceback
                            traceback.print_exc()
                        finally:
                            await bot.close()
                    
                    await bot.start(bot_token)
                    
                except Exception as e:
                    #print(f"❌ Errore connessione bot: {e}")
                    import traceback
                    traceback.print_exc()
            
            asyncio.run(send())
        
        # Avvia in thread per non bloccare l'UI
        thread = threading.Thread(target=run_async, daemon=True)
        thread.start()
        
        return True
        
    except Exception as e:
        #print(f"❌ Errore send_discord_bot_message: {e}")
        import traceback
        traceback.print_exc()
        return False



class TunnelURLDialog(QDialog):
    """Dialog per mostrare l'URL del tunnel con QR code e copia."""
    
    def __init__(self, url: str, parent=None):
        super().__init__(parent)
        self.url = url
        self.setWindowTitle(t("ui.cloudflare_ready"))
        self.setMinimumWidth(500)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Titolo
        title = QLabel(t("ui.collection_ready"))
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2ecc71; margin-bottom: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # QR Code
        try:
            import qrcode
            from io import BytesIO
            
            qr = qrcode.QRCode(version=1, box_size=10, border=2)
            qr.add_data(self.url)
            qr.make(fit=True)
            
            img = qr.make_image(fill_color="black", back_color="white")
            
            # Converti in QPixmap
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            
            pixmap = QPixmap()
            pixmap.loadFromData(buffer.read())
            
            qr_label = QLabel()
            qr_label.setPixmap(pixmap.scaled(250, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            qr_label.setAlignment(Qt.AlignCenter)
            qr_label.setStyleSheet("margin: 10px; background: white; padding: 10px; border-radius: 5px;")
            layout.addWidget(qr_label)
            
        except ImportError:
            error_label = QLabel(t("cloudflare.qr_code_error"))
            error_label.setStyleSheet("color: #e67e22; font-style: italic;")
            error_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(error_label)
        
        # URL con sfondo
        url_container = QLabel(self.url)
        url_container.setStyleSheet("""
            QLabel {
                background-color: #34495e;
                color: #ecf0f1;
                padding: 15px;
                border-radius: 8px;
                font-size: 14px;
                font-family: 'Courier New', monospace;
            }
        """)
        url_container.setAlignment(Qt.AlignCenter)
        url_container.setTextInteractionFlags(Qt.TextSelectableByMouse)
        url_container.setWordWrap(True)
        layout.addWidget(url_container)
        
        # Bottoni
        button_layout = QHBoxLayout()
        
        copy_btn = QPushButton(t("ui.copy_url"))
        copy_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        copy_btn.clicked.connect(self.copy_url)
        button_layout.addWidget(copy_btn)
        
        open_btn = QPushButton(t("ui.open_web"))
        open_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
        """)
        open_btn.clicked.connect(self.open_browser)
        button_layout.addWidget(open_btn)
        
        close_btn = QPushButton(t("ui.close_button"))
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
        """)
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        # Info
        info = QLabel(t("ui.show_qr"))
        info.setStyleSheet("color: #7f8c8d; font-size: 12px; margin-top: 10px;")
        info.setWordWrap(True)
        info.setAlignment(Qt.AlignCenter)
        layout.addWidget(info)
        
        self.setLayout(layout)
    
    def copy_url(self):

        clipboard = QApplication.clipboard()
        clipboard.setText(self.url)
        
        sender = self.sender()
        
        # Feedback visivo immediato
        original_text = sender.text()
        
        try:
            sender.setText(t("copy_btn"))
            sender.setStyleSheet("""
                QPushButton {
                    background-color: #27ae60;
                    color: white;
                    padding: 10px 20px;
                    border: none;
                    border-radius: 5px;
                    font-size: 14px;
                    font-weight: bold;
                }
            """)
            
            # Ripristina dopo 2 secondi SOLO se il dialog è ancora aperto
            QTimer.singleShot(2000, lambda: self._restore_button_safe(sender, original_text))
        except RuntimeError:
            # Se il bottone è già stato eliminato, ignora
            pass


    def _restore_button_safe(self, button, original_text):
        """Helper method per ripristinare il bottone in modo sicuro."""
        try:
            # Controlla se il widget esiste ancora
            if button and button.isVisible():
                button.setText(original_text)
                button.setStyleSheet("""
                    QPushButton {
                        background-color: #3498db;
                        color: white;
                        padding: 10px 20px;
                        border: none;
                        border-radius: 5px;
                        font-size: 14px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #2980b9;
                    }
                """)
        except (RuntimeError, AttributeError):
            # Widget è stato eliminato, ignora silenziosamente
            pass


    def open_browser(self):
        """Apri l'URL nel browser."""
        import webbrowser
        webbrowser.open(self.url)


class MainWindow(QMainWindow):
    """Finestra principale dell'applicazione."""

    def __init__(self):
        super().__init__()

        # ✅ VALIDA DATABASE ALL'AVVIO
        #print("\n" + "=" * 70)
        #print("🔧 STARTUP: Validazione Database")
        #print("=" * 70)

        db_manager = DatabaseManager()
        if db_manager.connect():
            db_manager.validate_and_repair_database()
            db_manager.close()
        else:
            QMessageBox.critical(
                self,
                t("error.db_connection_failed_title"),
                t("error.db_connection_failed_body"),
            )
            return
        self.db = DatabaseManager(log_callback=print)  # Usiamo print per i log
        if not self.db.connect():
            QMessageBox.critical(
                self,
                t("error.db_manager_connection_failed_title"),
                t("error.db_manager_connection_failed_body"),
            )
        self.wishlist_manager = WishlistManager(self.db)
        self.setWindowTitle(t("ui.window_title"))
        self.setGeometry(100, 100, 1400, 900)
        global SELECTED_RARITIES
        # Imposta l'icona se esiste
        # ⬇️ AGGIUNGI QUESTA PARTE ⬇️
        # Imposta icona della finestra (taskbar + titlebar)
        if os.path.exists(ICON_PATH):
            app_icon = QIcon(ICON_PATH)
            self.setWindowIcon(app_icon)

            # Imposta anche l'icona dell'applicazione (per Windows taskbar)
            if sys.platform == "win32":
                import ctypes

                myappid = (
                    "pcbisgood.tcgpockettracker.teamrocket.1"  # ID arbitrario univoco
                )
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
                QApplication.setWindowIcon(app_icon)
        self.image_cache = ImageCache(max_size=500)
        self.db_lock = Lock()
        self.conn = sqlite3.connect(DB_FILENAME, check_same_thread=False)
        # ================================================================
        # ✅ PASSO 1: BATCH WRITER SETUP
        # ================================================================
        # Coda thread-safe per le scritture sul DB
        self.db_write_queue = queue.Queue()

        # Timer per svuotare la coda in batch
        self.db_writer_timer = QTimer(self)
        self.db_writer_timer.timeout.connect(self.process_db_write_queue)
        self.db_writer_timer.start(5000)  # Processa ogni 5 secondi
        # ================================================================
        # ✅ PASSO 2: IMAGE LOADER SETUP
        # ================================================================
        self.image_loader_pool = QThreadPool()
        self.image_loader_pool.setMaxThreadCount(10)  # Max 10 download simultanei

        # Crea una pixmap segnaposto
        self.placeholder_pixmap = QPixmap(150, 210)  # Dimensioni standard carta
        self.placeholder_pixmap.fill(QColor("#3a3a3a"))
        # ================================================================
        self.bot_thread = None
        self.scraper_thread = None
        self.channel_loader_thread = None
        self.found_cards = []
        self.collection_loaded = False
        self.active_toasters = []
        self.web_viewer_btn = None

        self.tunnel_btn = None
        self.tunnel_thread = None
        self.collection_card_widgets = {}
        from typing import Optional, Dict

        self.current_account_id = None  # Optional[int]
        self.inventory_map = {}
        self.wishlist_map = {}
        self.channel_list_widget = QWidget()
        self.channel_list_layout = QVBoxLayout(self.channel_list_widget)
        self.channel_list_layout.setContentsMargins(10, 10, 10, 10)
        self.channel_list_layout.setSpacing(5)
        self.channel_list_layout.addWidget(
            QLabel(t("ui.channel_available"))
        )
        self.channel_list_layout.addStretch()
        self.setup_ui()
        #self.set_background_image("gui/background.png")
        # Load settings
        self.load_settings()
        # Setup system tray
        if self.token_input.text().strip():
            self.start_channel_loader()
        self.setup_system_tray()
        self.force_quit = False

    # ui_main_window.py (dentro MainWindow)
    def start_channel_loader(self):
        """Avvia il thread leggero per caricare la lista dei canali dal bot."""
        token = self.token_input.text().strip()
        if not token:
            return

        # Evita il doppio avvio
        if (
            hasattr(self, "channel_loader_thread")
            and self.channel_loader_thread
            and self.channel_loader_thread.isRunning()
        ):
            return

        # Crea e avvia il thread di caricamento canali
        self.channel_loader_thread = DiscordChannelLoaderThread(token)
        self.channel_loader_thread.channels_ready_signal.connect(self.on_channels_ready)
        self.channel_loader_thread.log_signal.connect(self.append_bot_log)
        self.channel_loader_thread.start()

    def create_image_tooltip(self, blob_data, text_fallback=""):
        """
        Crea un tooltip HTML da un BLOB di immagine (JPEG).
        Se il BLOB è nullo, ritorna il testo di fallback.
        """
        if not blob_data:
            return text_fallback  # Ritorna solo il testo

        try:
            # Converti i bytes del BLOB (JPEG) in una stringa base64
            b64_data = base64.b64encode(blob_data).decode("utf-8")

            # Crea un tag HTML <img>. Il tooltip si ridimensionerà
            # automaticamente all'immagine (es. 150x150 per le carte).
            return f'<html><img src="data:image/jpeg;base64,{b64_data}"></html>'

        except Exception as e:
            #print(f"⚠️ Errore creazione tooltip: {e}")
            return text_fallback  # Fallback al testo in caso di errore

    def add_found_card(self, card_data):
        """Aggiunge una carta trovata alla tabella della GUI."""
        try:
            # Ottieni la tabella
            if not hasattr(self, "cards_table"):
                return

            row = self.cards_table.rowCount()
            self.cards_table.insertRow(row)

            # Aggiungi i dati
            columns = [
                card_data.get("card_name", ""),
                card_data.get("set_code", "")
                + "_"
                + str(card_data.get("card_number", "")),
                card_data.get("rarity", ""),
                card_data.get("account_name", ""),
                t("misc.similarity_percent", similarity=str(card_data.get("similarity", 0.0))[:5]),
            ]

            for col_idx, value in enumerate(columns):
                item = QTableWidgetItem(str(value))
                self.cards_table.setItem(row, col_idx, item)

        except Exception as e:
            self.log_callback(f"{t("ui.card_error")} {e}")

    def setup_accounts_tab(self):
        """Configura il tab degli account (NUOVO)."""
        # Crea l'istanza della nuova scheda
        self.accounts_tab_widget = AccountsTab(self)  # <-- Crea l'attributo qui

        # Aggiungi il widget al QTabWidget
        self.tabs.addTab(self.accounts_tab_widget, t("ui.accounts_tab_with_icon"))

    def setup_ui(self):
        """Configura l'interfaccia utente."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # Tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Tab 2: Discord Bot
        self.setup_bot_tab()

        # Tab 3: Cards Found
        self.setup_cards_found_tab()

        # Tab 4: Collection
        self.setup_collection_tab()
        self.load_accounts_from_database()

        # ✅ Tab 1: Accounts (NUOVO)
        self.setup_accounts_tab()

        # Tab 5: Database Setup
        self.setup_database_tab()

        # Tab 6: Statistics
        self.setup_stats_tab()

        # Tab 7: Settings
        self.setup_settings_tab()
        self.tabs.currentChanged.connect(self.on_tab_changed)

    def get_all_accounts(self):
        """
        Recupera tutti gli account ordinati per nome.
        ✅ CORREZIONE: Seleziona device_account e lo aliasa a account_id per la compatibilità.
        """
        try:
            # self.cursor è il cursore dalla connessione di MainWindow
            self.cursor.execute(
                "SELECT device_account AS account_id, account_name FROM accounts ORDER BY account_name"
            )
            # Restituisce una lista di tuple [(device_id, display_name), ...]
            return self.cursor.fetchall()
        except Exception as e:
            # Registra l'errore che viene visualizzato
            #print(f"❌ Errore recupero account: {e}")
            return []

    def load_accounts_from_database(self):
        if not hasattr(self, "collection_account_combo"):
            return

        try:
            with self.db_lock:
                with sqlite3.connect(DB_FILENAME) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        # ✅ CORREZIONE: Seleziona device_account e lo aliasa a account_id
                        "SELECT device_account AS account_id, account_name FROM accounts ORDER BY account_name"
                    )
                    accounts = cursor.fetchall()

            # Pulisci: mantieni "Tutti gli account"
            while self.collection_account_combo.count() > 1:
                self.collection_account_combo.removeItem(1)

            # Aggiungi account (memorizza l'alias 'account_id' come userData)
            for account_id, account_name in accounts:
                self.collection_account_combo.addItem(account_name, account_id)

            #print(f"✅ Caricati {len(accounts)} account")
        except Exception as e:
            # Questo è l'errore che vedi se la SELECT fallisce qui
            print(f"❌ Errore caricamento account: {e}")

    def open_cloudflare_dialog(self):
        """Apre il dialog per configurare Cloudflare"""
        dialog = CloudflarePasswordDialog(self)
        if dialog.exec_() == QDialog.Accepted:  # PyQt5 usa exec_()
            password = dialog.password
            if password:
                # Mostra info su come usare Cloudflare
                info_msg = t("cloudflare.password_configured_body")
                QMessageBox.information(
                    self, t("cloudflare.password_configured_title"), info_msg
                )

    # =========================================================================
    # FLASK WEB SERVER
    # =========================================================================

    def toggle_web_server(self):
        """Avvia o ferma il server web Flask."""
        # ✅ CORRETTO: controlla prima se flask_thread è None
        if (
            not hasattr(self, "flask_thread")
            or self.flask_thread is None
            or not self.flask_thread.isRunning()
        ):
            self.start_web_server()
        else:
            self.stop_web_server()

    def start_web_server(self):
        """Avvia il server web Flask in un thread separato."""
        try:
            if not os.path.exists(DB_FILENAME):
                QMessageBox.warning(
                    self, t("warning.title"), t("warning.db_not_found")
                )
                return

            self.flask_thread = FlaskServerThread()
            self.flask_thread.log_signal.connect(self.on_flask_log)
            self.flask_thread.started_signal.connect(self.on_flask_started)
            self.flask_thread.stopped_signal.connect(self.on_flask_stopped)
            self.flask_thread.error_signal.connect(self.on_flask_error)

            # ✅ Controlla se il bottone esiste prima di usarlo
            if hasattr(self, "web_viewer_btn") and self.web_viewer_btn is not None:
                self.web_viewer_btn.setEnabled(False)
                self.web_viewer_btn.setText(t("ui.starting_server"))

            self.flask_thread.start()

        except Exception as e:
            import traceback

            error_msg = (
                f"{t('ui.failed_start_web_server')}: {str(e)}\n{traceback.format_exc()}"
            )
            QMessageBox.critical(self, t("error.title"), error_msg)

            # ✅ Controlla se il bottone esiste prima di usarlo
            if hasattr(self, "web_viewer_btn") and self.web_viewer_btn is not None:
                self.web_viewer_btn.setEnabled(True)

    def stop_web_server(self):
        """Ferma il server web Flask."""
        try:
            # ✅ Già corretto nel tuo codice
            if (
                hasattr(self, "flask_thread")
                and self.flask_thread
                and self.flask_thread.isRunning()
            ):
                self.web_viewer_btn.setEnabled(False)
                self.web_viewer_btn.setText(t("ui.stop_server"))
                self.flask_thread.stop_server()
        except Exception as e:
            QMessageBox.warning(self, t("warning.title"), f"{t('ui.error_stopping_server')}: {str(e)}")
            self.on_flask_stopped()

    def on_flask_log(self, message):
        """Gestisce i log del server Flask."""
        self.append_bot_log(message)
        print(message)

    def on_flask_started(self):
        """Chiamato quando Flask è avviato."""
        # ✅ Controlla se il bottone esiste prima di usarlo
        if hasattr(self, "web_viewer_btn") and self.web_viewer_btn is not None:
            self.web_viewer_btn.setEnabled(True)
            self.web_viewer_btn.setText(t("ui.stop_web_viewer"))
            self.web_viewer_btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #e74c3c;
                    color: white;
                    padding: 5px 15px;
                    font-weight: bold;
                }
            """
            )

        # Apri il browser dopo 1 secondo
        QTimer.singleShot(1000, self.open_web_browser)

        # Mostra il messaggio di successo
        QMessageBox.information(
            self,
            t("ui.webserver_started"),
            t("ui.webserver_running_message")
            + t("ui.url_label")
            + "http://localhost:5000"
            + t("ui.browser_will_open"),
        )

    def on_flask_stopped(self):
        """Chiamato quando Flask è fermato."""
        # ✅ Controlla se il bottone esiste prima di usarlo
        if hasattr(self, "web_viewer_btn") and self.web_viewer_btn is not None:
            self.web_viewer_btn.setEnabled(True)
            self.web_viewer_btn.setText(t("ui.open_web_viewer"))
            self.web_viewer_btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #27ae60;
                    color: white;
                    padding: 5px 15px;
                }
            """
            )

    def on_flask_error(self, error_message):
        """Chiamato in caso di errore Flask."""
        QMessageBox.critical(self, t("ui.flask_server_error"), error_message)
        self.on_flask_stopped()

    def open_web_browser(self):
        """Apre il browser con l'URL del server Flask."""
        import webbrowser

        try:
            webbrowser.open("http://localhost:5000")
        except Exception as e:
            print(f"{t("ui.cannot_open")} {e}")

    def on_wishlist_changed(self, card_id, is_wishlisted):
        """Gestisce il cambio di stato della wishlist."""
        try:
            with sqlite3.connect(DB_FILENAME) as conn:
                cursor = conn.cursor()

                if is_wishlisted:
                    # Aggiungi alla wishlist
                    cursor.execute(
                        """
                        INSERT OR IGNORE INTO wishlist (card_id, added_date)
                        VALUES (?, ?)
                    """,
                        (card_id, datetime.now().isoformat()),
                    )
                else:
                    # Rimuovi dalla wishlist
                    cursor.execute("DELETE FROM wishlist WHERE card_id = ?", (card_id,))

                conn.commit()
        except Exception as e:
            print(f"{t("ui.error_wishlist")} {e}")

    # ui_main_window.py (dentro MainWindow)
    def toggle_channel_selection(self, channel_id: int, checked: bool):
        """Aggiunge o rimuove un Channel ID dalla lista di selezione e salva."""
        if checked:
            self.selected_channel_ids.add(channel_id)
        else:
            self.selected_channel_ids.discard(channel_id)

        self.save_settings()

    def start_bot(self):
        """
        Avvia il Discord bot principale.
        Verifica il Token e la selezione dei canali (dopo che sono stati caricati).
        """
        token = self.token_input.text().strip()

        # 1. Validazione base: Il Token è l'unico campo obbligatorio per tentare l'avvio
        if not token:
            QMessageBox.warning(self, t("error.title"), t("ui.provide_token"))
            return

        # 2. Recupera la lista degli ID selezionati (dal set salvato/caricato)
        channel_ids = list(self.selected_channel_ids)

        # 3. Validazione Selezione Canali: Deve esserci almeno 1 canale SE LA LISTA è stata caricata.
        # Se il bot è alla prima connessione, channel_ids sarà vuota e la GUI sarà popolata.
        if (
            not channel_ids
            and hasattr(self, "available_channels")
            and self.available_channels
        ):
            QMessageBox.warning(
                self, t("error.title"), t("warning.select_one_channel")
            )
            return

        # 4. Ferma il loader leggero (se attivo)
        if (
            hasattr(self, "channel_loader_thread")
            and self.channel_loader_thread
            and self.channel_loader_thread.isRunning()
        ):
            self.channel_loader_thread.quit()
            self.channel_loader_thread.wait(500)

        # 5. Avvia il thread principale

        # Passa la LISTA di ID interi selezionati al thread (può essere vuota solo al primo avvio)
        self.bot_thread = DiscordBotThread(token, channel_ids)

        # ✅ NUOVO COLLEGAMENTO: Riceve i canali disponibili dal bot (per popolare/ri-popolare la UI)
        self.bot_thread.channels_ready_signal.connect(self.on_channels_ready)

        # Collegamenti standard
        self.bot_thread.log_signal.connect(self.append_bot_log)
        self.bot_thread.progress_signal.connect(self.on_bot_progress)
        self.bot_thread.trade_signal.connect(self.add_trade_to_table)
        self.bot_thread.status_signal.connect(self.update_bot_status)
        self.bot_thread.card_found_signal.connect(self.on_card_found)

        self.bot_thread.start()

        self.start_bot_btn.setEnabled(False)
        self.stop_bot_btn.setEnabled(True)
        self.bot_status_label.setText(t("bot_status.starting"))

        self.save_settings()

    # ui_main_window.py (dentro MainWindow)
    def create_collapsible_channel_group(self):
        """Crea una sezione espandibile/collassabile per la lista dei canali."""

        # === 1. IL CONTENITORE PRINCIPALE ===
        main_container = QWidget()
        main_layout = QVBoxLayout(main_container)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # === 2. PULSANTE DI CONTROLLO (Toggle Button) ===
        self.channel_toggle_btn = QToolButton(self)
        self.channel_toggle_btn.setText(t("ui.channel_available_text"))
        self.channel_toggle_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.channel_toggle_btn.setArrowType(Qt.RightArrow)
        self.channel_toggle_btn.setStyleSheet(
            """
            QToolButton { 
                background: #3a3a3a; /* Dark background */
                border: 1px solid #555; 
                padding: 5px; 
                text-align: left; 
                font-weight: bold; 
            }
        """
        )
        self.channel_toggle_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # === 3. CONTENUTO SCORREVOLE (Utilizza il widget esistente) ===

        # Crea l'area scrollabile e assegna il widget che contiene il layout esistente
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        # 💥 UTILIZZA IL WIDGET ESISTENTE CHE CONTIENE self.channel_list_layout
        scroll_area.setWidget(self.channel_list_widget)

        scroll_area.setStyleSheet("border: 1px solid #555; border-top: none;")
        scroll_area.setFixedHeight(150)

        self.channel_content_scroll_area = scroll_area

        # === 4. LOGICA DI COLLASSO ===
        # Inizialmente collassato
        self.channel_content_scroll_area.setVisible(False)

        def toggle_section(checked):
            # Mostra/nascondi l'area scrollabile (il contenuto)
            self.channel_content_scroll_area.setVisible(checked)
            # Cambia l'icona della freccia
            self.channel_toggle_btn.setArrowType(
                Qt.DownArrow if checked else Qt.RightArrow
            )

        self.channel_toggle_btn.setCheckable(True)
        self.channel_toggle_btn.setChecked(False)  # Inizia chiuso
        self.channel_toggle_btn.toggled.connect(toggle_section)

        # Aggiungi al layout principale
        main_layout.addWidget(self.channel_toggle_btn)
        main_layout.addWidget(self.channel_content_scroll_area)

        return main_container

    def on_channels_ready(self, channels_data: Dict[int, str]):
        """
        Riceve la lista dei canali disponibili dal bot e popola la UI con le Checkbox.
        self.channel_list_layout è ora persistente.
        ✅ FIX: Verifica che il layout non sia stato deletato
        """
        # 🔴 FIX #1: CONTROLLA CHE IL LAYOUT ESISTA E NON SIA DELETATO
        if not hasattr(self, "channel_list_layout"):
            #print("⚠️ Channel list layout non trovato")
            return

        # 🔴 FIX #2: VERIFICA CHE IL LAYOUT NON SIA STATO DISTRUTTO
        try:
            # Tenta di accedere a una proprietà del layout per verificare che sia valido
            _ = self.channel_list_layout.count()
        except RuntimeError:
            #print("⚠️ Channel list layout è stato distrutto, skip update")
            return

        # 💥 PULIZIA SICURA DEL LAYOUT 💥
        while self.channel_list_layout.count() > 0:
            item = self.channel_list_layout.takeAt(0)
            if item is None:
                break

            # Rimuovi widget
            widget = item.widget()
            if widget:
                widget.deleteLater()
            # Rimuovi spacer
            elif item.spacerItem():
                pass  # Gli spacer non hanno bisogno di cleanup esplicito

        self.available_channels = channels_data

        if not channels_data:
            self.channel_list_layout.addWidget(
                QLabel(t("ui.no_channel_found"))
            )
            return

        # Ordina per nome del canale
        sorted_channels = sorted(
            channels_data.items(), key=lambda item: item[1].lower()
        )

        # Crea una Checkbox per ogni canale
        for channel_id, channel_name in sorted_channels:
            cb = QCheckBox(t("channel.checkbox_label", channel_name=channel_name, channel_id=channel_id))

            # Carica lo stato salvato (se l'ID è nella lista salvata)
            if channel_id in self.selected_channel_ids:
                cb.setChecked(True)

            # Connetti il segnale per salvare lo stato al click
            cb.toggled.connect(
                lambda checked, cid=channel_id: self.toggle_channel_selection(
                    cid, checked
                )
            )
            self.channel_list_layout.addWidget(cb)

        # Aggiungi stretch finale
        self.channel_list_layout.addStretch()

        #self.append_bot_log(
        #    f"✅ Caricati {len(channels_data)} canali dal server. Seleziona quelli da scansionare."
        #)
        self.save_settings()

    def update_channel_count_label(self):
        """Aggiorna il contatore dei canali configurati nella UI."""
        if hasattr(self, "channel_configs") and hasattr(self, "channel_count_label"):
            count = len(self.channel_configs)
            self.channel_count_label.setText(t("ui.channels_configured", count=count))

    def setup_bot_tab(self):
        """Configura il tab del Discord bot."""
        bot_widget = QWidget()
        bot_layout = QVBoxLayout(bot_widget)

        # Configuration Group
        config_group = QGroupBox(t("ui.config_group_with_icon"))
        config_layout = QVBoxLayout(config_group)

        # Token
        token_layout = QHBoxLayout()
        token_layout.addWidget(QLabel(t("ui.bot_token_lable")))
        self.token_input = QLineEdit()
        self.token_input.setEchoMode(QLineEdit.Password)
        self.token_input.setPlaceholderText(t("ui.enter_discord_token"))
        token_layout.addWidget(self.token_input)
        config_layout.addLayout(token_layout)
        self.token_input.textChanged.connect(self.save_settings)

        # ✅ AGGIUNGI SOLO IL COLLAPSIBLE (che contiene già lo scroll_area)
        collapsible_channel_section = self.create_collapsible_channel_group()
        config_layout.addWidget(collapsible_channel_section)

        # ❌ RIMUOVI QUESTO BLOCCO INTERO:
        # scroll_area = QScrollArea()
        # scroll_area.setWidgetResizable(True)
        # scroll_area.setWidget(self.channel_list_widget)
        # config_layout.addWidget(scroll_area)

        # Buttons
        button_layout = QHBoxLayout()
        self.start_bot_btn = QPushButton(t("ui.start_bot"))
        self.start_bot_btn.clicked.connect(self.start_bot)
        self.start_bot_btn.setStyleSheet(
            "QPushButton { background-color: #2ecc71; color: white; padding: 8px; font-weight: bold; }"
        )

        self.stop_bot_btn = QPushButton(t("ui.stop_bot"))
        self.stop_bot_btn.clicked.connect(self.stop_bot)
        self.stop_bot_btn.setEnabled(False)
        self.stop_bot_btn.setStyleSheet(
            "QPushButton { background-color: #e74c3c; color: white; padding: 8px; font-weight: bold; }"
        )

        button_layout.addWidget(self.start_bot_btn)
        button_layout.addWidget(self.stop_bot_btn)

        self.recover_history_btn = QPushButton(t("ui.recover_history"))
        config_layout.addLayout(button_layout)

        bot_layout.addWidget(config_group)

        # Status Group
        status_group = QGroupBox(t("ui.status_group_with_icon"))
        status_layout = QVBoxLayout(status_group)
        self.bot_status_label = QLabel(t("bot_status.stopped"))
        self.bot_status_label.setStyleSheet(
            "QLabel { font-size: 14px; font-weight: bold; }"
        )
        status_layout.addWidget(self.bot_status_label)
        bot_layout.addWidget(status_group)

        # Progress
        progress_group = QGroupBox(t("ui.progress"))
        progress_layout = QVBoxLayout(progress_group)
        self.bot_progress_bar = QProgressBar()
        self.bot_progress_bar.setStyleSheet(
            "QProgressBar::chunk { background-color: #3498db; }"
        )
        progress_layout.addWidget(self.bot_progress_bar)
        bot_layout.addWidget(progress_group)

        # Log
        log_group = QGroupBox(t("ui.log_group_with_icon"))
        log_layout = QVBoxLayout(log_group)
        self.bot_log_text = QTextEdit()
        self.bot_log_text.setReadOnly(True)
        self.bot_log_text.setStyleSheet(
            """
            QTextEdit { 
                font-family: 'Consolas', 'Segoe UI Mono', monospace; 
                font-size: 12px; 
            }
        """
        )
        log_layout.addWidget(self.bot_log_text)
        bot_layout.addWidget(log_group)

        # Trades Table
        trades_group = QGroupBox(t("ui.recent_trades_group_with_icon"))
        trades_layout = QVBoxLayout(trades_group)
        self.trades_table = QTableWidget()
        self.trades_table.setColumnCount(5)
        self.trades_table.setHorizontalHeaderLabels(
            [
                t("ui.table.preview"),
                t("ui.table.account"),
                t("ui.table.cards_found"),
                t("ui.table.xml"),
                t("ui.table.image"),
            ]
        )
        self.trades_table.horizontalHeader().setStretchLastSection(True)
        self.trades_table.setAlternatingRowColors(True)
        self.trades_table.setColumnWidth(0, 80)
        self.trades_table.verticalHeader().setDefaultSectionSize(70)
        self.trades_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.trades_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.trades_table.setStyleSheet(
            """
            QTableWidget::item:selected {
                background-color: #f39c12; /* Un bel giallo/arancione */
                color: #000000; /* Testo nero (per leggibilità) */
            }
        """
        )
        trades_layout.addWidget(self.trades_table)
        bot_layout.addWidget(trades_group)

        self.tabs.addTab(bot_widget, t("ui.discord_bot_tab_with_icon"))

    def setup_cards_found_tab(self):
        """Configura il tab delle carte trovate (Refactored)."""
        # Crea l'istanza della nuova scheda
        self.cards_found_tab_widget = CardsFoundTab(self)

        # Aggiungi il widget al QTabWidget
        self.tabs.addTab(self.cards_found_tab_widget, t("ui.cards_found_tab_with_icon"))

    def _find_screenshot_for_message(self, account_name, message_id):
        """Trova il file screenshot per un messaggio specifico."""
        if not message_id:
            return ""

        # ✅ Cartella dell'account
        account_folder = os.path.join(ACCOUNTS_DIR, account_name, "images")

        if not os.path.exists(account_folder):
            return ""

        # ✅ Cerca file con il naming pattern: {message_id}.*
        try:
            files = os.listdir(account_folder)
            for file in files:
                # Il file dovrebbe essere chiamato: {message_id}.png, .jpg, ecc
                if file.startswith(str(message_id)):
                    full_path = os.path.join(account_folder, file)
                    if os.path.exists(full_path):
                        return full_path
        except Exception as e:
            print(f"{t("ui.screenshot_error")} {e}")

        return ""

    def download_cloudflared(self):
        """Scarica cloudflared.exe automaticamente e riavvia l'app."""
        try:
            from urllib.request import urlretrieve
            import shutil

            # URL di download per Windows 64-bit
            cloudflared_url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"

            # Path di destinazione finale (nella cartella dell'app)
            if getattr(sys, "frozen", False):
                # Se è EXE, metti nella stessa cartella dell'EXE
                app_dir = os.path.dirname(sys.executable)
            else:
                # Se è sviluppo, metti nella cartella corrente
                app_dir = os.getcwd()

            final_path = os.path.join(app_dir, "cloudflared.exe")

            # Path temporaneo per il download
            temp_path = os.path.join(
                os.path.expanduser("~"), "Downloads", "cloudflared-windows-amd64.exe"
            )

            # Mostra progress dialog
            progress = QProgressBar()
            progress_dialog = QMessageBox(self)
            progress_dialog.setWindowTitle(t("ui.download_cloudflare"))
            progress_dialog.setText(
                t("ui.downloading_cloudflared")
            )
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
            #self.append_bot_log(f"📥 Downloading cloudflared from GitHub...")
            urlretrieve(cloudflared_url, temp_path, reporthook=report_progress)

            # Copia nella cartella dell'app e rinomina
            #self.append_bot_log(f"📂 Installing to: {final_path}")
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
                t("ui.download_completed"),
                t("ui.installed_successfuly"),
                t("ui.need_restart"),
                t("ui.restart_now"),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )

            if reply == QMessageBox.Yes:
                self.restart_application()
            else:
                QMessageBox.information(
                    self,
                    t("ui.restart_needed"),
                    t("ui.restart_now_app"),
                )

        except Exception as e:
            import traceback

            error_msg = (
                f"Failed to download cloudflared: {str(e)}\n\n{traceback.format_exc()}"
            )
            self.append_bot_log(f"❌ {error_msg}")

            QMessageBox.critical(
                self,
                t("download.failed_title"),
                t(
                    "download.cloudflared_failed",
                    e=str(e),
                    app_dir=app_dir
                ),
            )

    def restart_application(self):
        """Riavvia l'applicazione."""
        try:
            self.append_bot_log(t("app.restarting"))

            # Salva le impostazioni prima di riavviare
            self.save_settings()

            # Ferma tutti i thread attivi
            if (
                hasattr(self, "bot_thread")
                and self.bot_thread
                and self.bot_thread.isRunning()
            ):
                self.bot_thread.stop_bot()
                self.bot_thread.wait(3000)

            if (
                hasattr(self, "scraper_thread")
                and self.scraper_thread
                and self.scraper_thread.isRunning()
            ):
                self.scraper_thread.quit()
                self.scraper_thread.wait(3000)

            if (
                hasattr(self, "flask_thread")
                and self.flask_thread
                and self.flask_thread.isRunning()
            ):
                self.flask_thread.stop_server()
                self.flask_thread.wait(3000)

            # Ottieni path dell'eseguibile
            if getattr(sys, "frozen", False):
                # Se è EXE
                executable = sys.executable
            else:
                # Se è sviluppo (Python script)
                executable = sys.executable
                script = os.path.abspath(sys.argv[0])

            # Chiudi l'app corrente
            QApplication.quit()

            # Riavvia in un nuovo processo
            if getattr(sys, "frozen", False):
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
                t("app.restart_failed_title"),
                t("app.restart_failed_body", e=str(e)),
            )

    def open_info_dialog(self):
        """
        Apre un dialogo separato per mostrare le "Extra Info" (crediti e link).
        """
        # Creiamo un QDialog al volo
        dialog = QDialog(self)
        dialog.setWindowTitle(t("credits.dialog_title"))
        dialog.setMinimumSize(600, 450)
        dialog.setModal(True)

        layout = QVBoxLayout(dialog)

        info_text = QTextBrowser()
        info_text.setReadOnly(True)
        info_text.setOpenExternalLinks(True)
        info_text.setStyleSheet(
            """
            QTextEdit {
                background-color: #2a2a2a; border: 1px solid #555;
                border-radius: 5px; padding: 15px; font-size: 11px; line-height: 1.6;
            }
        """
        )

        # Incolliamo qui il contenuto HTML che abbiamo rimosso
        info_text.setHtml(t("credits.html_content"))

        layout.addWidget(info_text)

        # Pulsante OK
        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)

        dialog.exec_()

    def select_bot_folder(self):
        """Apre un dialogo per selezionare la cartella del Bot."""
        # Legge il percorso attuale, se esiste, per aprire il dialogo lì
        current_path = self.bot_folder_input.text()
        if not os.path.isdir(current_path):
            current_path = os.path.expanduser("~")  # Fallback alla home

        folder_path = QFileDialog.getExistingDirectory(
            self, t("ui.select_folder_bot"), current_path
        )

        if folder_path:
            self.bot_folder_input.setText(folder_path)
            # Il segnale textChanged si occuperà di salvare

    def open_app_data_folder(self):
        """
        Apre la cartella dei dati dell'applicazione (AppData)
        nel file explorer del sistema operativo.
        """
        try:
            # Usiamo DB_FILENAME come riferimento per trovare la cartella
            # get_app_data_path() ci dà il percorso completo del file
            app_data_dir = os.path.dirname(get_app_data_path(DB_FILENAME))

            if not os.path.exists(app_data_dir):
                QMessageBox.warning(self, t("warning.title"), t("ui.app_data_error"))
                return

            #print(f"ℹ️ Apertura cartella dati: {app_data_dir}")

            # Usa il metodo nativo del SO per aprire la cartella
            if sys.platform == "win32":
                os.startfile(app_data_dir)
            elif sys.platform == "darwin":  # macOS
                subprocess.Popen(["open", app_data_dir])
            else:  # Linux
                subprocess.Popen(["xdg-open", app_data_dir])

        except Exception as e:
            QMessageBox.critical(
                self,
                t("error.folder_open_failed_title"),
                t("error.folder_open_failed_body", e=e),
            )

    def setup_collection_tab(self):
        """
        Configura il tab della collezione (Refactored).
        Crea semplicemente l'istanza di CollectionTab e la aggiunge.
        """
        # Crea il widget della scheda passando 'self' (la MainWindow)
        self.collection_tab_widget = CollectionTab(self)

        # Aggiungi il widget al QTabWidget
        self.tabs.addTab(self.collection_tab_widget, t("ui.collection_tab_with_icon"))

    # =========================================================================
    # CLOUDFLARE TUNNEL
    # =========================================================================

    def toggle_cloudflare_tunnel(self):
        """Avvia o ferma il tunnel Cloudflare."""
        # ✅ Usa tunnel_thread invece di cloudflare_thread
        if (
            hasattr(self, "tunnel_thread")
            and self.tunnel_thread
            and self.tunnel_thread.isRunning()
        ):
            self.stop_cloudflare_tunnel()
        else:
            self.start_cloudflare_tunnel()

    def on_cloudflare_url(self, url: str):
        """Chiamato quando il tunnel è pronto con l'URL pubblico."""
        if hasattr(self, 'tunnel_btn') and self.tunnel_btn is not None:
            self.tunnel_btn.setEnabled(True)
            self.tunnel_btn.setText(t("cloudflare.stop_tunnel_button"))
            self.tunnel_btn.setStyleSheet("""
                QPushButton {
                    background-color: #e74c3c;
                    color: white;
                    padding: 5px 15px;
                    font-weight: bold;
                }
            """)
        
        self.append_bot_log(t("cloudflare.public_url_log", url=url))
        
        # ✅ Mostra dialog con QR code e copy button
        dialog = TunnelURLDialog(url, self)
        dialog.exec_()


    def on_cloudflare_log(self, message: str):
        """Log dal tunnel Cloudflare."""
        self.append_bot_log(t("cloudflare.log_prefix", message=message))

    def on_cloudflare_error(self, error_message: str):
        """Errore dal tunnel Cloudflare."""
        QMessageBox.critical(self, t("cloudflare.tunnel_error_title"), error_message)

        if hasattr(self, "tunnel_btn") and self.tunnel_btn is not None:
            self.tunnel_btn.setEnabled(True)
            self.tunnel_btn.setText(t("ui.start_tunnel"))

    def on_cloudflare_stopped(self):
        """Tunnel Cloudflare fermato."""
        if hasattr(self, "tunnel_btn") and self.tunnel_btn is not None:
            self.tunnel_btn.setEnabled(True)
            self.tunnel_btn.setText(t("ui.start_tunnel"))
            self.tunnel_btn.setStyleSheet("")

        self.append_bot_log(f"✅ {t("ui.stop_tunnel")}")

    def on_cloudflare_error(self, error_message: str):
        """Chiamato in caso di errore."""
        QMessageBox.critical(self, t("cloudflare.tunnel_error_title"), error_message)

        if hasattr(self, "tunnel_btn") and self.tunnel_btn is not None:
            self.tunnel_btn.setEnabled(True)
            self.tunnel_btn.setText(t("ui.start_tunnel"))

    def on_cloudflare_stopped(self):
        """Chiamato quando il tunnel viene fermato."""
        if hasattr(self, "tunnel_btn") and self.tunnel_btn is not None:
            self.tunnel_btn.setEnabled(True)
            self.tunnel_btn.setText(t("ui.start_tunnel"))
            self.tunnel_btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #3498db;
                    color: white;
                    padding: 5px 15px;
                }
            """
            )

    def on_cloudflare_log(self, message: str):
        """Riceve i log dal tunnel."""
        print(t("cloudflare.log_prefix", message=message))


    def start_cloudflare_tunnel(self):
        """Avvia il tunnel Cloudflare con auto-start Flask."""
        try:
            # ✅ AUTO-START Flask se non è in esecuzione
            if (
                not hasattr(self, "flask_thread")
                or not self.flask_thread
                or not self.flask_thread.isRunning()
            ):
                self.append_bot_log(t("ui.starting_cloudflare"))
                self.start_web_server()
                # Aspetta 2 secondi che Flask sia pronto
                QTimer.singleShot(2000, self._continue_cloudflare_start)
                return

            self._continue_cloudflare_start()

        except Exception as e:
            import traceback

            error_msg = (
                f"Failed to start Cloudflare Tunnel: {str(e)}\n{traceback.format_exc()}"
            )
            QMessageBox.critical(self, t("error.title"), error_msg)

            if hasattr(self, "tunnel_btn") and self.tunnel_btn is not None:
                self.tunnel_btn.setEnabled(True)
                self.tunnel_btn.setText(t("ui.start_tunnel"))

    def _continue_cloudflare_start(self):
        """Continua l'avvio di Cloudflare dopo che Flask è pronto."""
        try:
            from .cloudflare import CloudflareTunnelThread

            self.tunnel_thread = CloudflareTunnelThread(local_port=5000)

            # ✅ CORREZIONE: Usa url_ready_signal invece di url_signal
            self.tunnel_thread.log_signal.connect(self.on_cloudflare_log)
            self.tunnel_thread.url_ready_signal.connect(self.on_cloudflare_url)
            self.tunnel_thread.stopped_signal.connect(self.on_cloudflare_stopped)
            self.tunnel_thread.error_signal.connect(self.on_cloudflare_error)

            if hasattr(self, "tunnel_btn") and self.tunnel_btn is not None:
                self.tunnel_btn.setEnabled(False)
                self.tunnel_btn.setText(t("ui.starting_cf"))

            self.tunnel_thread.start()

        except Exception as e:
            import traceback

            error_msg = (
                f"Failed to start Cloudflare Tunnel: {str(e)}\n{traceback.format_exc()}"
            )
            QMessageBox.critical(self, t("error.title"), error_msg)

            if hasattr(self, "tunnel_btn") and self.tunnel_btn is not None:
                self.tunnel_btn.setEnabled(True)
                self.tunnel_btn.setText(t("ui.start_tunnel"))

    def _start_tunnel_after_flask(self):
        """Avvia il tunnel dopo che Flask è partito."""
        # Verifica che Flask sia effettivamente partito
        if hasattr(self, "flask_thread") and self.flask_thread.isRunning():
            self._start_tunnel_now()
        else:
            QMessageBox.warning(self, t("error.title"), t("ui.failed_start_web_server"))
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
            self.tunnel_btn.setText(t("ui.starting_server"))

            # Avvia thread
            self.tunnel_thread.start()

        except Exception as e:
            QMessageBox.critical(self, t("error.title"), f"{str(e)}")
            self.tunnel_btn.setEnabled(True)

    def stop_cloudflare_tunnel(self):
        """Ferma il tunnel Cloudflare."""
        try:
            if (
                hasattr(self, "tunnel_thread")
                and self.tunnel_thread
                and self.tunnel_thread.isRunning()
            ):
                if hasattr(self, "tunnel_btn") and self.tunnel_btn is not None:
                    self.tunnel_btn.setEnabled(False)
                    self.tunnel_btn.setText(t("ui.stop_server"))

                self.tunnel_thread.stop_tunnel()
                self.tunnel_thread.wait(5000)

        except Exception as e:
            QMessageBox.warning(self, t("warning.title"), f"{str(e)}")
            self.on_cloudflare_stopped()

    def on_tunnel_log(self, message):
        """Gestisce i log del tunnel."""
        self.append_bot_log(message)

    def on_tunnel_url_ready(self, public_url):
        """Chiamato quando l'URL pubblico è pronto."""
        self.tunnel_btn.setEnabled(True)
        self.tunnel_btn.setText(t("ui.stop_web_viewer"))
        self.tunnel_btn.setStyleSheet(
            "QPushButton { background-color: #e74c3c; color: white; padding: 5px 15px; font-weight: bold; }"
        )

        # Mostra dialog con URL
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle(t("cloudflare.url_dialog_title"))
        msg.setText(t("cloudflare.url_dialog_text", public_url=public_url))
        msg.setDetailedText(t("ui.tunnel_success"))

        # Pulsante per copiare URL
        copy_btn = msg.addButton(t("cloudflare.copy_button"), QMessageBox.ActionRole)
        open_btn = msg.addButton(t("cloudflare.open_button"), QMessageBox.ActionRole)
        msg.addButton(QMessageBox.Ok)

        msg.exec_()

        clicked = msg.clickedButton()
        if clicked == copy_btn:
            QApplication.clipboard().setText(public_url)
            #self.append_bot_log("📋 Public URL copied to clipboard")
        elif clicked == open_btn:
            import webbrowser

            webbrowser.open(public_url)

    def on_tunnel_stopped(self):
        """Chiamato quando il tunnel è fermato."""
        self.tunnel_btn.setEnabled(True)
        self.tunnel_btn.setText(t("ui.expose_online"))
        self.tunnel_btn.setStyleSheet(
            "QPushButton { background-color: #2c3e50; color: white; padding: 5px 15px; }"
        )

    def on_tunnel_error(self, error_message):
        """Chiamato in caso di errore tunnel."""
        QMessageBox.critical(
            self, t("cloudflare.tunnel_error_critical_title"), error_message
        )
        self.on_tunnel_stopped()

    def setup_database_tab(self):
        """Configura il tab del database (Refactored)."""
        # Crea l'istanza della nuova scheda
        self.scraper_tab_widget = ScraperTab(self)

        # Aggiungi il widget al QTabWidget
        self.tabs.addTab(self.scraper_tab_widget, t("ui.database_setup_tab_with_icon"))

    def setup_stats_tab(self):
        """Configura il tab delle statistiche."""
        stats_widget = QWidget()
        stats_layout = QVBoxLayout(stats_widget)

        # Buttons
        button_layout = QHBoxLayout()
        self.refresh_stats_btn = QPushButton(t("stats_ui.refresh_statistics"))
        self.refresh_stats_btn.clicked.connect(self.refresh_stats)
        self.refresh_stats_btn.setStyleSheet(
            "QPushButton { background-color: #3498db; color: white; padding: 8px; font-weight: bold; }"
        )
        button_layout.addWidget(self.refresh_stats_btn)
        button_layout.addStretch()
        stats_layout.addLayout(button_layout)

        # Stats Text
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setStyleSheet(
            "QTextEdit { font-family: 'Courier New'; font-size: 11px; }"
        )
        stats_layout.addWidget(self.stats_text)

        self.tabs.addTab(stats_widget, t("ui.statistics_tab_with_icon"))

    def on_language_changed(self):
        """Callback per cambio lingua."""
        new_lang = self.language_combo.currentData()
        set_language(new_lang)
        self.save_settings()

    def clear_account_inventory(self):
        """Svuota la tabella account_inventory."""

        # Conferma 1
        reply = QMessageBox.question(
            self,
            t("ui.reset_inventory"),
            t("ui.warning_message"),
            t("ui.cannot_undone"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply != QMessageBox.Yes:
            return

        # Conferma 2 (Ancora più forte)
        final_reply = QMessageBox.warning(
            self,
            t("ui.are_sure"),
            t("ui.last_hope"),
            QMessageBox.Ok | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )

        if final_reply == QMessageBox.Ok:
            try:
                with sqlite3.connect(DB_FILENAME, timeout=10.0) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM account_inventory")
                    count = cursor.fetchone()[0]

                    if count == 0:
                        QMessageBox.information(
                            self, t("info.title"), t("ui.inventory_full")
                        )
                        return

                    # Svuota la tabella
                    cursor.execute("DELETE FROM account_inventory")
                    conn.commit()

                    QMessageBox.information(
                        self,
                        t("ui.success"),
                        t("ui.delete_done"),
                    )


            except Exception as e:
                QMessageBox.critical(
                    self, t("error.title"), t("error.generic_body", e=e)
                )

    def clear_trades_log(self):
        """Svuota la tabella trades."""

        reply = QMessageBox.question(
            self,
            t("trade_log.reset_title"),
            t("trade_log.reset_warning"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            try:
                with sqlite3.connect(DB_FILENAME, timeout=10.0) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM trades")
                    count = cursor.fetchone()[0]

                    if count == 0:
                        QMessageBox.information(
                            self, t("info.title"), t("trade_log.already_empty")
                        )
                        return

                    # Svuota la tabella
                    cursor.execute("DELETE FROM trades")
                    conn.commit()

                    QMessageBox.information(
                        self,
                        t("ui.success"),
                        t("trade_log.reset_success", count=count),
                    )

            except Exception as e:
                QMessageBox.critical(
                    self,
                    t("trade_log.reset_error_title"),
                    t("trade_log.reset_error_body", e=e),
                )

    def setup_settings_tab(self):
        """Configura il tab delle impostazioni con selezione rarità."""
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)

        # ================================================================
        # SEZIONE RARITÀ (Invariata)
        # ================================================================
        rarity_group = QGroupBox(t("ui.settings_tab.rarity_filter_group"))
        rarity_layout = QHBoxLayout(rarity_group)
        rarity_layout.setSpacing(15)
        rarity_layout.setContentsMargins(10, 10, 10, 10)
        saved_rarities = (
            SELECTED_RARITIES if SELECTED_RARITIES else list(RARITY_DATA.keys())
        )
        self.rarity_checkboxes = {}
        self.rarity_labels = {}
        ICON_HEIGHT = 30
        for rarity_name, icon_filename in RARITY_DATA.items():
            rarity_widget = QWidget()
            rarity_widget.setCursor(Qt.PointingHandCursor)
            rarity_widget.setProperty("rarity_name", rarity_name)
            rarity_widget.setProperty("selected", rarity_name in saved_rarities)
            rarity_container = QVBoxLayout(rarity_widget)
            rarity_container.setContentsMargins(0, 0, 0, 0)
            rarity_container.setSpacing(0)
            icon_label = QLabel()
            icon_label.setAlignment(Qt.AlignCenter)
            icon_path = get_resource_path(icon_filename)
            if os.path.exists(icon_path):
                pixmap = QPixmap(icon_path)
                pixmap = pixmap.scaledToHeight(ICON_HEIGHT, Qt.SmoothTransformation)
                icon_label.setPixmap(pixmap)
                icon_label.setProperty("original_pixmap", pixmap)
            else:
                icon_label.setText(rarity_name.replace(" ", "\n"))
                icon_label.setFixedSize(ICON_HEIGHT, ICON_HEIGHT)
                icon_label.setStyleSheet(
                    """
                    QLabel {
                        font-weight: bold; font-size: 10px; border: 2px solid #ccc;
                        background: #f0f0f0; border-radius: 5px; padding: 5px;
                    }
                """
                )
            icon_label.setToolTip(rarity_name)
            if rarity_name not in saved_rarities:
                self.apply_grayscale_filter(icon_label, rarity_name)
            rarity_container.addWidget(icon_label, alignment=Qt.AlignCenter)
            self.rarity_labels[rarity_name] = icon_label
            self.rarity_checkboxes[rarity_name] = rarity_widget
            rarity_widget.mousePressEvent = (
                lambda event, name=rarity_name: self.toggle_rarity(name)
            )
            rarity_layout.addWidget(rarity_widget)
        rarity_layout.addStretch()
        settings_layout.addWidget(rarity_group)

        # ================================================================
        # SEZIONE KEVIN BOT FOLDER (Aggiunta)
        # ================================================================
        bot_folder_group = QGroupBox(t("ui.settings_tab.bot_path_group"))
        bot_folder_layout = QHBoxLayout(bot_folder_group)
        self.bot_folder_input = QLineEdit()
        self.bot_folder_input.setPlaceholderText(
            t("ui.settings_tab.bot_path_placeholder")
        )
        self.bot_folder_input.textChanged.connect(self.save_settings)
        bot_folder_layout.addWidget(self.bot_folder_input)
        self.bot_folder_btn = QPushButton(t("ui.settings_tab.browse_button"))
        self.bot_folder_btn.clicked.connect(self.select_bot_folder)
        bot_folder_layout.addWidget(self.bot_folder_btn)
        settings_layout.addWidget(bot_folder_group)

        # ================================================================
        # SEZIONE NOTIFICATION SETTINGS (Aggiunta)
        # ================================================================
        notification_group = QGroupBox(t("ui.settings_tab.notification_group_title"))
        notification_layout = QVBoxLayout(notification_group)
        self.notification_enable_cb = QCheckBox(
            t("ui.settings_tab.notification_enable_checkbox")
        )
        self.notification_enable_cb.toggled.connect(self.save_settings)
        notification_layout.addWidget(self.notification_enable_cb)
        notif_channel_layout = QHBoxLayout()
        notif_channel_layout.addWidget(QLabel(t("ui.settings_tab.notification_channel_label")))
        self.notification_channel_input = QLineEdit()
        self.notification_channel_input.setPlaceholderText(
            t("ui.settings_tab.notification_channel_placeholder")
        )
        self.notification_channel_input.textChanged.connect(self.save_settings)
        notif_channel_layout.addWidget(self.notification_channel_input)
        notification_layout.addLayout(notif_channel_layout)
        settings_layout.addWidget(notification_group)

        # ================================================================
        # ℹ️ SEZIONE INFO & LINK (RIMOSSA)
        # ================================================================
        # Questo blocco è stato completamente rimosso.
        # Il suo contenuto è ora in open_info_dialog()

        # ================================================================
        # ⚙️ APPLICATION SETTINGS (Con offset)
        # ================================================================
        settings_group = QGroupBox(t("ui.application_settings"))
        settings_group_layout = QVBoxLayout(settings_group)
        settings_group_layout.setContentsMargins(10, 10, 10, 10)  # Offset
        settings_group_layout.setSpacing(10)  # Offset
        language_layout = QHBoxLayout()
        language_layout.addWidget(QLabel(t("ui.language") + ":"))
        self.language_combo = QComboBox()
        self.language_combo.addItem("🇬🇧 English", "en")
        self.language_combo.addItem("🇫🇷 Français", "fr")
        self.language_combo.addItem("🇮🇹 Italiano", "it")
        self.language_combo.currentIndexChanged.connect(self.on_language_changed)
        language_layout.addWidget(self.language_combo)
        language_layout.addStretch()
        settings_group_layout.addLayout(language_layout)
        self._language_combo_initialized = False
        theme_layout = QHBoxLayout()
        theme_layout.addWidget(QLabel(t("ui.settings_tab.theme_label")))
        self.theme_combo = QCheckBox(t("ui.settings_tab.dark_theme_checkbox"))
        self.theme_combo.setChecked(True)
        theme_layout.addWidget(self.theme_combo)
        theme_layout.addStretch()
        settings_group_layout.addLayout(theme_layout)
        self.autostart_cb = QCheckBox(t("ui.settings_tab.autostart_checkbox"))
        settings_group_layout.addWidget(self.autostart_cb)
        self.minimize_tray_cb = QCheckBox(t("ui.settings_tab.minimize_to_tray_checkbox"))
        self.minimize_tray_cb.setChecked(True)
        settings_group_layout.addWidget(self.minimize_tray_cb)
        settings_layout.addWidget(settings_group)

        # ================================================================
        # 🗄️ DATABASE MANAGEMENT (Modificato con pulsante Info)
        # ================================================================
        db_management_group = QGroupBox(t("ui.database_management"))
        db_management_layout = QVBoxLayout(db_management_group)

        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)

        # --- Pulsanti Esistenti (allineati a sinistra) ---
        self.clear_found_cards_btn = QPushButton(t("ui.settings_tab.clear_cards_log_button"))
        self.clear_found_cards_btn.clicked.connect(self.clear_found_cards)
        self.clear_found_cards_btn.setStyleSheet(
            "QPushButton { background-color: #e67e22; color: white; padding: 8px; font-weight: bold; border-radius: 5px; }"
        )
        self.clear_found_cards_btn.setToolTip(
            t("ui.settings_tab.clear_cards_log_tooltip")
        )
        buttons_layout.addWidget(self.clear_found_cards_btn)

        self.clear_inventory_btn = QPushButton(t("ui.settings_tab.reset_inventory_button"))
        self.clear_inventory_btn.clicked.connect(self.clear_account_inventory)
        self.clear_inventory_btn.setStyleSheet(
            "QPushButton { background-color: #e74c3c; color: white; padding: 8px; font-weight: bold; border-radius: 5px; }"
        )
        self.clear_inventory_btn.setToolTip(
            t("ui.settings_tab.reset_inventory_tooltip")
        )
        buttons_layout.addWidget(self.clear_inventory_btn)

        self.clear_trades_btn = QPushButton(t("ui.settings_tab.reset_trades_button"))
        self.clear_trades_btn.clicked.connect(self.clear_trades_log)
        self.clear_trades_btn.setStyleSheet(
            "QPushButton { background-color: #c0392b; color: white; padding: 8px; font-weight: bold; border-radius: 5px; }"
        )
        self.clear_trades_btn.setToolTip(
            t("ui.settings_tab.reset_trades_tooltip")
        )
        buttons_layout.addWidget(self.clear_trades_btn)

        self.open_appdata_btn = QPushButton(t("ui.settings_tab.open_data_folder_button"))
        self.open_appdata_btn.clicked.connect(self.open_app_data_folder)
        self.open_appdata_btn.setStyleSheet(
            "QPushButton { background-color: #3498db; color: white; padding: 8px; font-weight: bold; border-radius: 5px; }"
        )
        self.open_appdata_btn.setToolTip(
            t("ui.settings_tab.open_data_folder_tooltip")
        )
        buttons_layout.addWidget(self.open_appdata_btn)

        # Aggiunge spazio flessibile tra i pulsanti di sinistra e quello di destra
        buttons_layout.addStretch()

        # ================================================================
        # ✅ NUOVO PULSANTE INFO (Aggiunto a destra)
        # ================================================================
        self.info_btn = QPushButton()
        self.info_btn.setIcon(
            self.style().standardIcon(QStyle.SP_MessageBoxInformation)
        )
        self.info_btn.setFixedSize(32, 32)
        self.info_btn.setToolTip(t("ui.settings_tab.show_info_tooltip"))
        self.info_btn.setStyleSheet(
            """
            QPushButton {
                border-radius: 16px; /* Metà della dimensione fissa (32/2) */
                border: 1px solid #555;
                background-color: #3a3a3a;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
        """
        )
        # Collega il click alla nuova funzione
        self.info_btn.clicked.connect(self.open_info_dialog)
        buttons_layout.addWidget(self.info_btn)  # Aggiunto all'estrema destra
        # ================================================================

        db_management_layout.addLayout(buttons_layout)
        settings_layout.addWidget(db_management_group)

        # ================================================================
        # FINALIZE
        # ================================================================
        settings_layout.addStretch()  # Rimuove lo stretch proporzionale
        self.tabs.addTab(settings_widget, t("ui.settings"))

    # ================================================================
    # ✅ FUNZIONI PER GESTIRE RARITÀ
    # ================================================================

    def toggle_rarity(self, rarity_name):
        """Toggle rarità al click."""
        try:
            widget = self.rarity_checkboxes[rarity_name]
            label = self.rarity_labels[rarity_name]

            is_selected = widget.property("selected")
            new_state = not is_selected
            widget.setProperty("selected", new_state)

            if new_state:
                # ✅ MODIFICATO: Passa rarity_name
                self.remove_grayscale_filter(label, rarity_name)
            else:
                # ✅ MODIFICATO: Passa rarity_name
                self.apply_grayscale_filter(label, rarity_name)

            self.save_settings()

        except Exception as e:
            print(f"Error toggling rarity: {e}")

    def apply_grayscale_filter(self, label, rarity_name):
        """
        Applica quando DISATTIVATA:
        1. Scala di grigi all'immagine
        2. Bordo ROSSO ai pixel trasparenti
        """
        try:
            pixmap = label.property("original_pixmap")
            if pixmap and not pixmap.isNull():
                image = pixmap.toImage()

                if image.format() != image.Format_RGBA8888:
                    image = image.convertToFormat(image.Format_RGBA8888)

                border_radius = 2

                # ============================================================
                # ✅ STEP 1: APPLICA GRAYSCALE A TUTTI I PIXEL VISIBILI
                # ============================================================
                for y in range(image.height()):
                    for x in range(image.width()):
                        pixel = image.pixel(x, y)
                        alpha = (pixel >> 24) & 0xFF

                        # Se il pixel è visibile (non trasparente), converti a grayscale
                        if alpha > 100:
                            r = (pixel >> 16) & 0xFF
                            g = (pixel >> 8) & 0xFF
                            b = pixel & 0xFF

                            # Calcola luminanza (grayscale)
                            gray = int(0.299 * r + 0.587 * g + 0.114 * b)

                            # Reimposta con grayscale
                            new_pixel = (
                                (alpha << 24) | (gray << 16) | (gray << 8) | gray
                            )
                            image.setPixel(x, y, int(new_pixel))

                # ============================================================
                # ✅ STEP 2: RILEVA I BORDI (pixel trasparenti)
                # ============================================================
                border_mask = [[False] * image.width() for _ in range(image.height())]

                for y in range(image.height()):
                    for x in range(image.width()):
                        pixel = image.pixel(x, y)
                        alpha = (pixel >> 24) & 0xFF

                        if alpha == 0:
                            # Controlla se è vicino a un pixel opaco
                            for dx in [-1, 0, 1]:
                                for dy in [-1, 0, 1]:
                                    nx, ny = x + dx, y + dy
                                    if (
                                        0 <= nx < image.width()
                                        and 0 <= ny < image.height()
                                    ):
                                        neighbor = image.pixel(nx, ny)
                                        if ((neighbor >> 24) & 0xFF) > 100:
                                            border_mask[y][x] = True
                                            break

                # ============================================================
                # ✅ STEP 3: APPLICA ROSSO AI BORDI RILEVATI
                # ============================================================
                for y in range(image.height()):
                    for x in range(image.width()):
                        if border_mask[y][x]:
                            # ✅ ROSSO SEMI-TRASPARENTE
                            new_pixel = (180 << 24) | (255 << 16) | (0 << 8) | 0
                            image.setPixel(x, y, int(new_pixel))

                painter_pixmap = QPixmap.fromImage(image)
                label.setPixmap(painter_pixmap)

        except Exception as e:
            print(f"Error applying red border: {e}")

    def remove_grayscale_filter(self, label, rarity_name):
        """
        Applica quando ATTIVATA:
        1. Mantieni colori originali (niente grayscale)
        2. Bordo VERDE ai pixel trasparenti
        """
        try:
            pixmap = label.property("original_pixmap")
            if pixmap and not pixmap.isNull():
                # ✅ COPIA ORIGINALE SENZA MODIFICHE (colori originali)
                image = pixmap.toImage()

                if image.format() != image.Format_RGBA8888:
                    image = image.convertToFormat(image.Format_RGBA8888)

                border_radius = 2

                # ============================================================
                # STEP 1: RILEVA I BORDI (pixel trasparenti)
                # ============================================================
                border_mask = [[False] * image.width() for _ in range(image.height())]

                for y in range(image.height()):
                    for x in range(image.width()):
                        pixel = image.pixel(x, y)
                        alpha = (pixel >> 24) & 0xFF

                        if alpha == 0:
                            # Controlla se è vicino a un pixel opaco
                            for dx in [-1, 0, 1]:
                                for dy in [-1, 0, 1]:
                                    nx, ny = x + dx, y + dy
                                    if (
                                        0 <= nx < image.width()
                                        and 0 <= ny < image.height()
                                    ):
                                        neighbor = image.pixel(nx, ny)
                                        if ((neighbor >> 24) & 0xFF) > 100:
                                            border_mask[y][x] = True
                                            break

                # ============================================================
                # ✅ STEP 2: APPLICA VERDE AI BORDI RILEVATI
                # ============================================================
                for y in range(image.height()):
                    for x in range(image.width()):
                        if border_mask[y][x]:
                            # ✅ VERDE SEMI-TRASPARENTE
                            new_pixel = (180 << 24) | (0 << 16) | (255 << 8) | 0
                            image.setPixel(x, y, int(new_pixel))
                painter_pixmap = QPixmap.fromImage(image)
                label.setPixmap(painter_pixmap)

        except Exception as e:
            print(f"Error applying green border: {e}")

    # ================================================================
    # ✅ FUNZIONE HELPER - OPEN URL
    # ================================================================

    def open_url(self, url):
        """Apre un URL nel browser."""
        try:
            webbrowser.open(url)
        except Exception as e:
            self.append_bot_log(f"❌ Error opening URL: {e}")

    def clear_found_cards(self):
        """Vide la table found_cards de la base de données."""

        # ✅ MODIFICATO: Spiegazione aggiunta al popup
        reply = QMessageBox.question(
            self,
            t("ui.dialog.clear_found_cards_log_title"),
            t("ui.dialog.clear_found_cards_log_text"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            try:
                # Compte d'abord le nombre d'entrées
                with sqlite3.connect(DB_FILENAME, timeout=10.0) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM found_cards")
                    count = cursor.fetchone()[0]

                    if count == 0:
                        QMessageBox.information(
                            self, t("ui.info"), t("ui.no_found_cards_to_clear")
                        )
                        return

                    # ❌ RIMOSSA: Seconda conferma (ora è ridondante)

                    # Svuota la tabella
                    cursor.execute("DELETE FROM found_cards")
                    conn.commit()

                    QMessageBox.information(
                        self, t("ui.success"), t("ui.found_cards_cleared", count=count)
                    )

                    # Log dans l'interface
                    if hasattr(self, "append_bot_log"):
                        self.append_bot_log(
                            f"🗑️ {t('ui.found_cards_cleared', count=count)}"
                        )

            except Exception as e:
                QMessageBox.critical(
                    self, t("ui.error"), t("ui.error_clearing_found_cards") + f": {e}"
                )
                import traceback

                print(traceback.format_exc())

    def show_from_tray(self):
        """Mostra la finestra dal system tray."""
        self.show()
        self.raise_()
        self.activateWindow()

    def quit_application(self):
        """
        Avvia la sequenza di chiusura completa dell'applicazione.
        (Chiamato dal menu della Tray Icon)
        """
        print("ℹ️ Uscita forzata richiesta dalla tray icon...")
        self.force_quit = True  # Imposta il flag
        self.close()  # Chiama closeEvent per gestire lo shutdown

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

        show_action = tray_menu.addAction(t("ui.tray.show"))
        show_action.triggered.connect(self.show_from_tray)  # ⬅️ Cambia qui

        quit_action = tray_menu.addAction(t("ui.tray.quit"))
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

    # ui_main_window.py (dentro MainWindow)

    def start_bot(self):
        """
        Avvia il Discord bot principale.
        Verifica il Token e la selezione dei canali (dopo che sono stati caricati).
        """
        token = self.token_input.text().strip()

        # 1. Validazione del Token (unico check rigido)
        if not token:
            QMessageBox.warning(self, "Error", "Please provide Bot Token.")
            return

        # 2. Recupera la lista degli ID selezionati (dal set salvato/caricato)
        channel_ids = list(self.selected_channel_ids)

        # 3. Validazione Selezione Canali (solo se la lista dei canali è già stata caricata)
        # Questa condizione controlla che:
        # A) Nessun canale è stato selezionato (not channel_ids) E
        # B) La lista dei canali disponibili (available_channels) è già stata popolata dal loader
        #    Questo impedisce il messaggio di errore al primo avvio assoluto.
        if (
            not channel_ids
            and hasattr(self, "available_channels")
            and self.available_channels
        ):
            QMessageBox.warning(
                self, "Error", "Please select at least one Channel ID to scan."
            )
            return

        # 4. Ferma il loader leggero (se attivo)
        if (
            hasattr(self, "channel_loader_thread")
            and self.channel_loader_thread
            and self.channel_loader_thread.isRunning()
        ):
            self.channel_loader_thread.quit()
            self.channel_loader_thread.wait(500)

        # 5. Avvia il thread principale

        # Passa la LISTA di ID interi selezionati al thread
        self.bot_thread = DiscordBotThread(token, channel_ids)

        # Collegamenti standard
        self.bot_thread.channels_ready_signal.connect(self.on_channels_ready)
        self.bot_thread.log_signal.connect(self.append_bot_log)
        self.bot_thread.progress_signal.connect(self.on_bot_progress)
        self.bot_thread.trade_signal.connect(self.add_trade_to_table)
        self.bot_thread.status_signal.connect(self.update_bot_status)
        self.bot_thread.card_found_signal.connect(self.on_card_found)

        self.bot_thread.start()

        self.start_bot_btn.setEnabled(False)
        self.stop_bot_btn.setEnabled(True)
        self.bot_status_label.setText(t("bot_status.starting"))

        self.save_settings()

    def stop_bot(self):
        """Ferma il Discord bot."""
        if self.bot_thread:
            self.bot_thread.stop()
            self.bot_thread = None

        self.start_bot_btn.setEnabled(True)
        self.stop_bot_btn.setEnabled(False)
        self.bot_status_label.setText(t("bot_status.stopped"))
        self.append_bot_log(t("discord_bot.bot_stopped"))

    def recover_history(self):
        """Lance manuellement la récupération de l'historique depuis le dernier message traité."""
        if (
            not hasattr(self, "bot_thread")
            or not self.bot_thread
            or not self.bot_thread.isRunning()
        ):
            QMessageBox.warning(self, t("ui.warning"), t("ui.bot_must_be_running"))
            return

        if not hasattr(self.bot_thread, "client") or not self.bot_thread.client:
            QMessageBox.warning(self, t("ui.warning"), t("ui.bot_not_ready"))
            return

        # Vérifie que le scan initial est terminé
        if (
            not hasattr(self.bot_thread.client, "initial_scan_done")
            or not self.bot_thread.client.initial_scan_done
        ):
            QMessageBox.warning(self, t("ui.warning"), t("ui.wait_for_initial_scan"))
            return

        # Demande confirmation
        reply = QMessageBox.question(
            self,
            t("ui.recover_history"),
            t("ui.recover_history_confirmation"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            self.append_bot_log("🔄 " + t("discord_bot.starting_manual_recovery"))
            self.recover_history_btn.setEnabled(False)

            # Émet un signal pour demander la récupération au thread du bot
            try:
                self.bot_thread.recover_history_signal.emit()
                # Réactive le bouton après un délai (le scan peut prendre du temps)
                QTimer.singleShot(
                    10000, lambda: self.recover_history_btn.setEnabled(True)
                )
            except Exception as e:
                self.append_bot_log("❌ " + t("discord_bot.bot_error") + f": {e}")
                import traceback

                self.append_bot_log(traceback.format_exc())
                self.recover_history_btn.setEnabled(True)

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

    def on_bot_progress(self, progress_info: dict):
        """
        Aggiorna la progress bar del bot con info (testo e valore).
        Gestisce sia lo stato 'percentuale' che 'indeterminato'.
        """
        try:
            percent = progress_info.get("percent", 0)
            status = progress_info.get("status", t("misc.processing"))

            # Mostra il testo
            self.bot_progress_bar.setFormat(status)
            self.bot_progress_bar.setTextVisible(True)

            # Se 'percent' è -1, attiviamo la modalità "busy" (indeterminata)
            if percent == -1:
                self.bot_progress_bar.setRange(0, 0)  # Min=0, Max=0
            else:
                # Altrimenti, impostiamo la percentuale normale
                self.bot_progress_bar.setRange(0, 100)  # Min=0, Max=100
                self.bot_progress_bar.setValue(percent)

        except Exception as e:
            print(f"❌ Errore progress bot: {e}")

    def update_scraper_progress(self, progress_info):
        """Aggiorna la progress bar con informazioni ricche dal dizionario."""

        try:
            # Estrai i dati dal dict
            set_code = progress_info.get("set_code", "N/A")
            set_name = progress_info.get("set_name", "N/A")
            cards_done = progress_info.get("cards_done", 0)
            cards_total = progress_info.get("cards_total", 0)
            sets_completed = progress_info.get("sets_completed", 0)
            sets_total = progress_info.get("sets_total", 1)
            percent = progress_info.get("percent", 0)

            # Crea la barra di progresso
            bar_length = 16
            filled = int((percent / 100) * bar_length) if percent > 0 else 0
            bar = "█" * filled + "░" * (bar_length - filled)

            progress_text = (
                f"[{set_code}] {set_name}: {cards_done}/{cards_total} "
                f"[{sets_completed}/{sets_total}] [{bar}] {percent}%"
            )

            # ✅ Stampa il testo nel tooltip della progress bar
            self.bot_progress_bar.setFormat(progress_text)

            # ✅ Aggiorna la progress bar con la percentuale
            self.bot_progress_bar.setValue(percent)

        except Exception as e:
            print(f"❌ Errore update_scraper_progress: {e}")

    def update_bot_status(self, status):
        """Aggiorna lo status del bot."""
        status_icons = {
            "Connected": "🟢",
            "Monitoring": "🟢",
            "Scanning": "🟡",
            "Error": "🔴",
        }
        icon = status_icons.get(status, "⚫")
        self.bot_status_label.setText(t("bot_status.status_text", icon=icon, status=status))

    # =========================================================================
    # TRADE AND CARD MANAGEMENT
    # =========================================================================

    def add_trade_to_table(self, trade_data):
        """
        Aggiunge un trade alla tabella CON MINIATURA (dal BLOB).
        MODIFICATO: Aggiunge .scaled() alla miniatura.
        """
        row = self.trades_table.rowCount()
        self.trades_table.insertRow(row)

        # Colonna 0: Miniatura (dal BLOB)
        preview_label = QLabel()
        preview_label.setAlignment(Qt.AlignCenter)

        image_blob = trade_data.get("screenshot_thumbnail_blob")
        if image_blob:
            pixmap = QPixmap()
            pixmap.loadFromData(image_blob)
            if not pixmap.isNull():
                # ✅ CORREZIONE: Ridimensiona il pixmap per la cella
                pixmap = pixmap.scaled(
                    60, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                preview_label.setPixmap(pixmap)
            else:
                preview_label.setText(t("misc.cross_mark"))
        else:
            preview_label.setText(t("misc.picture_mark"))

        # ✅ Tooltip (Usa il BLOB originale, grande)
        preview_label.setToolTip(self.create_image_tooltip(image_blob, t("misc.screenshot")))

        self.trades_table.setCellWidget(row, 0, preview_label)

        # Altre colonne
        account_item = QTableWidgetItem(trade_data.get("account_name", ""))
        account_item.setData(Qt.UserRole, trade_data.get("image_url"))

        cards_item = QTableWidgetItem(trade_data.get("cards_found", ""))
        xml_item = QTableWidgetItem(t("misc.check_mark") if trade_data.get("xml_path") else t("misc.ballot_x"))
        image_item = QTableWidgetItem(t("misc.check_mark") if trade_data.get("image_url") else t("misc.ballot_x"))

        self.trades_table.setItem(row, 1, account_item)
        self.trades_table.setItem(row, 2, cards_item)
        self.trades_table.setItem(row, 3, xml_item)
        self.trades_table.setItem(row, 4, image_item)

        while self.trades_table.rowCount() > 100:
            self.trades_table.removeRow(0)

        self.trades_table.scrollToBottom()

    # ui_main_window.py (dentro MainWindow)
    def _get_or_create_account(
        self,
        account_id: str,
        account_name: Optional[str] = None,
        device_password: Optional[str] = None,
    ):
        """
        Ottiene o crea un account nel database (Thread-safe).
        Questa versione usa account_id come unica PK, poiché è stata determinata dal bot.
        """
        final_pk = account_id

        if not final_pk:
            return None

        try:
            # Usiamo self.db_lock e self.conn (connessione della MainWindow)
            with self.db_lock:
                cursor = self.conn.cursor()

                # STEP 1: Inserisci o Ignora. account_name è qui il nome di fallback.
                # Se è già stato creato dal bot (con deviceId o Fallback), ignora.
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO accounts 
                    (device_account, account_name, device_password) 
                    VALUES (?, ?, ?)
                """,
                    (final_pk, final_pk, device_password),
                )

                # STEP 2: Aggiorna solo se necessario
                if device_password or account_name:
                    # Usiamo final_pk come account_name se non fornito (per consistenza)
                    name_to_set = account_name if account_name else final_pk

                    cursor.execute(
                        """
                        UPDATE accounts 
                        SET device_password = ?, 
                            account_name = ?
                        WHERE device_account = ?
                    """,
                        (device_password, name_to_set, final_pk),
                    )

                self.conn.commit()

                # STEP 3: Ritorna la PK utilizzata
                return final_pk

        except Exception as e:
            print(f"⚠️ Errore gestione account '{final_pk}' in MainWindow: {e}")
            return final_pk

    def process_db_write_queue(self):
        """
        Processa la coda di scrittura del database.
        Inserisce i log in 'found_cards' utilizzando le tuple fornite.
        """
        if self.db_write_queue.empty():
            return

        cards_to_process = []

        # Svuota la coda in modo sicuro
        while not self.db_write_queue.empty():
            try:
                # item_tuple è: (card_id, account_id, message_id, confidence, path)
                cards_to_process.append(self.db_write_queue.get_nowait())
            except queue.Empty:
                break

        if not cards_to_process:
            return

        try:
            # Eseguiamo tutto in un'unica transazione
            with sqlite3.connect(DB_FILENAME, timeout=10.0) as conn:
                cursor = conn.cursor()

                # SQL: Inserisci nel log 'found_cards' (5 valori)
                # La tupla in coda è: (card_id, account_id, message_id, confidence, path)
                sql_found_log = """
                    INSERT OR IGNORE INTO found_cards 
                    (card_id, account_id, message_id, confidence_score, source_image_path)
                    VALUES (?, ?, ?, ?, ?)
                """

                # Esegue l'inserimento batch di tutte le tuple raccolte
                cursor.executemany(sql_found_log, cards_to_process)

                # Finalizza la transazione
                conn.commit()

        except Exception as e:
            # L'errore 'account_id' era probabilmente causato da un errore precedente nel ciclo.
            # Stampiamo l'errore reale:
            print(f"❌ Errore Batch Writer: {e}")
            # Se fallisce, rimetti gli elementi in coda per il prossimo tentativo
            print(
                f"⚠️ Dati non inseriti, {len(cards_to_process)} elementi verranno riprovati."
            )
            for item in cards_to_process:
                self.db_write_queue.put(item)

        except Exception as e:
            print(f"❌ Errore Batch Writer: {e}")
            # Se fallisce, rimetti gli elementi in coda per il prossimo tentativo
            print(
                f"⚠️ Dati non inseriti, {len(cards_to_process)} elementi verranno riprovati."
            )
            for item in cards_to_process:
                self.db_write_queue.put(item)

    # In ui_main_window.py

    # ui_main_window.py (dentro MainWindow)
    def on_card_found(self, card_data):
        """
        Chiamato quando viene trovata una carta.
        Recupera i BLOB e aggiunge l'elemento alla coda di scrittura DB.
        """

        rarity_name = card_data.get("rarity", "Unknown")
        card_id = None
        thumbnail_blob = None
        screenshot_blob = None
        set_cover_blob = None
        is_wishlisted = False

        # ================================================================
        # PASSO 1: LETTURA DB (Card ID, Blobs, Wishlist Check)
        # ================================================================
        try:
            with self.db_lock:
                cursor = self.conn.cursor()

                # Recupera card_id e thumbnail_blob
                cursor.execute(
                    "SELECT id, thumbnail_blob FROM cards WHERE set_code = ? AND card_number = ?",
                    (card_data["set_code"], card_data["card_number"]),
                )
                result = cursor.fetchone()
                if result:
                    card_id = result[0]
                    thumbnail_blob = result[1]

                # Recupera screenshot_blob
                message_id = card_data.get("message_id")
                if message_id:
                    cursor.execute(
                        "SELECT screenshot_thumbnail_blob FROM trades WHERE message_id = ?",
                        (message_id,),
                    )
                    result_trade = cursor.fetchone()
                    if result_trade:
                        screenshot_blob = result_trade[0]

                # Recupera cover_image_blob DAL DB
                set_code = card_data.get("set_code")
                if set_code:
                    cursor.execute(
                        "SELECT cover_image_blob FROM sets WHERE set_code = ?",
                        (set_code,),
                    )
                    result_set = cursor.fetchone()
                    if result_set:
                        set_cover_blob = result_set[0]

                # CONTROLLA SE È IN WISHLIST
                if card_id:
                    cursor.execute(
                        "SELECT 1 FROM wishlist WHERE card_id = ?", (card_id,)
                    )
                    is_wishlisted = cursor.fetchone() is not None

        except Exception as e:
            print(f"❌ Error checking wishlist/blobs: {e}")
            return

        if not card_id:
            return


        # ================================================================
        # PASSO 3: AGGIORNAMENTO UI + NOTIFICA WISHLIST
        # ================================================================
        try:
            # Recupera la logica del filtro rarità
            try:
                from config import get_app_data_path, RARITY_DATA

                settings_path = get_app_data_path("settings.json")
            except:
                settings_path = "settings.json"

            saved_rarities = list(RARITY_DATA.keys())
            if os.path.exists(settings_path):
                try:
                    with open(settings_path, "r", encoding="utf-8") as f:
                        settings = json.load(f)
                    saved_rarities = settings.get(
                        "selected_rarities", list(RARITY_DATA.keys())
                    )
                except:
                    pass

            if rarity_name not in saved_rarities:
                return

            self.append_bot_log(
                f"   → {card_data['set_code']}_{card_data['card_number']} [{rarity_name}]"
            )

            # AGGIUNGI I BLOB AL DIZIONARIO
            card_data["thumbnail_blob"] = thumbnail_blob
            card_data["screenshot_thumbnail_blob"] = screenshot_blob
            card_data["set_cover_blob"] = set_cover_blob

            # ORA INVIA LA NOTIFICA WISHLIST
            if is_wishlisted:
                self.send_wishlist_notification(card_data)

            # DELEGA L'AGGIORNAMENTO UI
            if hasattr(self, "cards_found_tab_widget"):
                self.cards_found_tab_widget.add_new_card(card_data)

        except Exception as e:
            print(f"❌ Errore aggiornamento UI CardsFound: {e}")
            import traceback

            traceback.print_exc()

            print(t("misc.image_not_available"))
            return


    def open_pack_image(self, event, path):
        """
        Apre l'URL dello screenshot.
        MODIFICATO: Apre gli URL nel browser.
        """
        if not path:
            print(t("misc.image_not_available"))
            return

        try:
            if path.startswith("http"):
                # È un URL, apri nel browser
                print(t("misc.opening_screenshot_url_browser", path=path))
                import webbrowser

                webbrowser.open(path)
            elif os.path.exists(path):
                # È un path locale (logica vecchia)
                print(t("misc.opening_screenshot_file_internal", path=path))
                dialog = ImageViewerDialog(path, self)
                dialog.exec_()
            else:
                print(t("misc.file_or_url_not_found", path=path))
        except Exception as e:
            print(t("misc.error_opening_screenshot_dialog", e=e))




    def send_wishlist_notification(self, card_data):
        """
        Invia notifiche per la wishlist.
        ORDINE:
        1. Toast (Windows notification)
        2. Discord (webhook) - SE ABILITATO
        3. Tray (fallback)
        """
        print(
            f"🔔 send_wishlist_notification chiamata per: {card_data.get('card_name', 'Unknown')}"
        )

        try:
            from windows_toasts import (
                Toast,
                WindowsToaster,
                InteractableWindowsToaster,
                ToastDisplayImage,
                ToastImagePosition,
            )
            windows_toast_available = True
            print("   ✅ windows_toasts importato con successo")
        except ImportError as e:
            windows_toast_available = False
            print(f"   ❌ windows_toasts NON disponibile: {e}")

        # ============================================================================
        # STEP 1: Invia Toast Notification (Windows)
        # ============================================================================
        
        if windows_toast_available:
            try:
                print("   📱 Tentativo notifica Toast...")
                from .notification_manager import send_toast_notification

                success = send_toast_notification(card_data)
                print(f"   Toast success: {success}")

                if not success:
                    print("⚠️ Notifica Toast fallita, uso il fallback (Tray).")
                    
            except Exception as e:
                print(f"❌ Errore imprevisto in send_toast_notification: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("⚠️ Windows Toast NON disponibile")

        # ============================================================================
        # STEP 2: Invia Discord Notification (SE ABILITATO NELLE SETTINGS)
        # ============================================================================
        
        try:
            import json
            import os
            from config import get_app_data_path
            
            settings_path = get_app_data_path("settings.json")
            
            if os.path.exists(settings_path):
                with open(settings_path, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                
                # Leggi la flag per Discord notifications
                discord_enabled = settings.get('custom_notification_enabled', False)
                discord_channel_id = settings.get('notification_channel', '')
                
                print(f"   🔵 Discord notifications abilitato: {discord_enabled}")
                print(f"   🔵 Discord channel ID: {discord_channel_id if discord_channel_id else 'NOT SET'}")
                
                if discord_enabled and discord_channel_id:
                    print("   📤 Invio notifica Discord...")
                    discord_success = send_discord_bot_message(card_data)
                    print(f"   Discord success: {discord_success}")
                else:
                    print("   ⏭️  Discord notifications disabilitato o channel non configurato")
                    
        except Exception as e:
            print(f"❌ Errore Discord notifications: {e}")
            import traceback
            traceback.print_exc()

        # ============================================================================
        # STEP 3: Invia Tray Notification (FALLBACK)
        # ============================================================================
        
        print("   📌 Invio notifica Tray (fallback)...")
        #self.send_tray_notification(card_data)



    # =========================================================================
    # IMAGE PREVIEW AND INTERACTION
    # =========================================================================

    def show_image_preview(self, image_path):
        """Mostra un'anteprima piccola dell'immagine (tooltip o status bar)."""
        # Implementazione opzionale per tooltip hover
        pass

    #   # =========================================================================
    #   # STATISTICS
    #   # =========================================================================

    def refresh_stats(self):
        """Aggiorna le statistiche.
        ✅ CORRETTO: Aggiornate le JOIN per usare la nuova chiave primaria device_account.
        """
        try:
            # ================================================================
            # ✅ CORREZIONE: Carica le rarità dinamicamente dalle impostazioni
            # ================================================================
            try:
                from config import get_app_data_path, RARITY_DATA

                settings_path = get_app_data_path("settings.json")
            except:
                settings_path = "settings.json"

            saved_rarities = list(RARITY_DATA.keys())  # Default
            if os.path.exists(settings_path):
                try:
                    with open(settings_path, "r", encoding="utf-8") as f:
                        settings = json.load(f)
                    saved_rarities = settings.get(
                        "selected_rarities", list(RARITY_DATA.keys())
                    )
                except Exception:
                    pass

            with sqlite3.connect(DB_FILENAME) as conn:
                cursor = conn.cursor()

                # Sets
                cursor.execute("SELECT COUNT(*) FROM sets")
                sets_count = cursor.fetchone()[0]

                # Cards
                cursor.execute("SELECT COUNT(*) FROM cards")
                cards_count = cursor.fetchone()[0]

                # Rarity Counts (Invariato)
                placeholders = ", ".join("?" for _ in saved_rarities)
                rarity_query = f"""
                    SELECT rarity, COUNT(*) 
                    FROM cards 
                    WHERE rarity IN ({placeholders})
                    GROUP BY rarity
                """
                cursor.execute(rarity_query, saved_rarities)
                rarity_counts = cursor.fetchall()

                # Accounts
                cursor.execute("SELECT COUNT(*) FROM accounts")
                accounts_count = cursor.fetchone()[0]

                # Total inventory (Invariato)
                cursor.execute("SELECT SUM(quantity) FROM account_inventory")
                total_inventory = cursor.fetchone()[0] or 0

                # Found cards
                cursor.execute("SELECT COUNT(*) FROM found_cards")
                found_count = cursor.fetchone()[0]

                # ✅ CORREZIONE: Top accounts - JOIN e GROUP BY devono usare device_account (PK)
                cursor.execute(
                    """
                    SELECT a.account_name, SUM(ai.quantity) as total
                    FROM accounts a
                    -- JOIN su account_id (in inventory) = device_account (in accounts)
                    JOIN account_inventory ai ON a.device_account = ai.account_id
                    -- GROUP BY usa la PK testuale
                    GROUP BY a.device_account
                    ORDER BY total DESC
                    LIMIT 10
                """
                )
                top_accounts = cursor.fetchall()

                # Top cards (Invariato)
                cursor.execute(
                    """
                    SELECT c.card_name, c.set_code, c.rarity, COUNT(*) as times_found
                    FROM found_cards fc
                    JOIN cards c ON fc.card_id = c.id
                    GROUP BY fc.card_id
                    ORDER BY times_found DESC
                    LIMIT 10
                """
                )
                top_cards = cursor.fetchall()

                # --- Costruzione del testo (invariata) ---

                stats_text = """
╔═══════════════════════════════════════════════════════════════════╗
║                    DATABASE STATISTICS                            ║
╚═══════════════════════════════════════════════════════════════════╝
"""

                self.stats_text.setPlainText(stats_text)

        except Exception as e:
            self.stats_text.setPlainText(t("error.loading_stats", error=str(e)))
            import traceback

            traceback.print_exc()

    # =========================================================================
    # SETTINGS
    # =========================================================================

    # ui_main_window.py (Aggiungi questi metodi alla classe MainWindow)

    def _get_account_id_by_name(self, account_name: str) -> Optional[int]:
        """Ottiene l'account_id dal nome dell'account. Restituisce None se è 'All Accounts'."""
        # Assumiamo che la traduzione 't' sia disponibile per i nomi di account come "All Accounts"
        if account_name.lower() in [self.t("ui.all_accounts").lower(), "all accounts"]:
            return None

        try:
            # self.conn è la tua connessione attiva al DB
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT device_account FROM accounts WHERE account_name = ?
            """,
                (account_name,),
            )
            result = cursor.fetchone()
            return int(result[0]) if result else None
        except Exception as e:
            self.log_callback(f"Errore recupero account ID: {e}")
            return None

    def update_collection_data(self, selected_account_name: str):
        """Centralizza il caricamento dell'inventario."""
        is_all_accounts = selected_account_name.lower() in [
            "tutti gli account",
            "all accounts",
        ]
        account_id = (
            self.get_account_id_by_name(selected_account_name)
            if not is_all_accounts
            else None
        )
        self.current_account_id = account_id

        # Ricarica l'inventario
        self.inventory_map = {}
        try:
            with sqlite3.connect(DB_FILENAME) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Query corretta per ottenere le quantità
                if is_all_accounts:
                    query = """
                        SELECT card_id, SUM(quantity) AS total 
                        FROM account_inventory 
                        GROUP BY card_id
                    """
                    cursor.execute(query)
                else:
                    query = """
                        SELECT ai.card_id, SUM(ai.quantity) AS total 
                        FROM account_inventory ai
                        JOIN accounts a ON ai.account_id = a.account_id
                        WHERE a.account_name = ?
                        GROUP BY ai.card_id
                    """
                    cursor.execute(query, (selected_account_name,))

                for row in cursor.fetchall():
                    self.inventory_map[int(row["card_id"])] = int(row["total"])

            print(f"✅ Inventario caricato: {len(self.inventory_map)} carte")
        except Exception as e:
            print(f"❌ Errore caricamento inventario: {e}")
            import traceback

            traceback.print_exc()

    def update_collection_view(self):
        """
        Forza il ricaricamento della vista della collezione.
        Deve chiamare la funzione che ricarica la tab corrente.
        """
        if (
            hasattr(self, "collection_tab_widget")
            and self.collection_tab_widget is not None
        ):
            current_index = self.collection_tab_widget.currentIndex()
            # Assumendo che on_tab_changed sia la funzione che gestisce il refresh
            self.on_tab_changed(current_index)

    def get_account_id_by_name(self, account_name):
        """
        Recupera l'ID dell'account dal nome.
        ✅ MODIFICATO: Cerca il Device ID (PK) per corrispondenza e lo ritorna.
        """
        if account_name.lower() in ["tutti gli account", "all accounts"]:
            return None

        try:
            with sqlite3.connect(DB_FILENAME) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    # ✅ Seleziona la PK (device_account) e aliasala a account_id per la riga
                    "SELECT device_account AS account_id FROM accounts WHERE account_name = ?",
                    (account_name,),
                )
                row = cursor.fetchone()

                # Ritorna il Device ID (come stringa, che è il nuovo ID)
                return row["account_id"] if row else None
        except Exception as e:
            print(f"❌ Errore recupero account ID: {e}")
            return None

    def on_set_ready_batch(
        self, set_code, set_name, total_cards, cover_path, owned_count, total_copies
    ):
        """Aggiunge un set alla collection."""
        try:
            set_widget = self.create_set_section_fast(
                set_code,
                set_name,
                total_cards,
                cover_path,
                owned_count,
                total_copies,
                None,  # cursor non è più necessario
            )

            if set_widget:
                self.collection_container_layout.insertWidget(
                    self.collection_container_layout.count() - 1, set_widget
                )
        except Exception as e:
            print(f"❌ Errore aggiunta set: {e}")

    def on_collection_progress(self, message):
        """Aggiorna il messaggio di progresso."""
        cache_info = f" (Cache: {self.image_cache.size()}/500)"
        self.collection_stats_label.setText(f"⏳ {message}{cache_info}")
        self.collection_stats_label.setText(f"⏳ {message}")

    def load_set_cards(
        self, content_widget, set_code, inventory, cursor, account_name=None
    ):
        """
        Carica le carte di un set (chiamato solo quando necessario - lazy loading).
        ✅ CORRETTO: Modificate tutte le JOIN con acc.device_account.
        """
        try:
            # Query per ottenere info set (inclusa cover)
            cursor.execute(
                """
                SELECT set_name, total_cards, cover_image_path 
                FROM sets 
                WHERE set_code = ?
            """,
                (set_code,),
            )

            set_info = cursor.fetchone()
            set_name = set_info[0] if set_info else set_code
            total_cards = set_info[1] if set_info and set_info[1] else 0
            cover_path = (
                set_info[2] if set_info and len(set_info) > 2 and set_info[2] else None
            )

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
                        scaled_cover = cover_pixmap.scaled(
                            80, 112, Qt.KeepAspectRatio, Qt.SmoothTransformation
                        )
                        cover_label.setPixmap(scaled_cover)
                        cover_label.setFixedSize(80, 112)
                        cover_label.setStyleSheet(
                            """
                            QLabel {
                                border: 2px solid #555;
                                border-radius: 5px;
                                background-color: #2a2a2a;
                                padding: 2px;
                            }
                        """
                        )
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
            set_label.setStyleSheet(
                "QLabel { font-size: 16px; font-weight: bold; color: #e0e0e0; }"
            )
            info_layout.addWidget(set_label)

            # Set code
            code_label = QLabel(f"Set Code: {set_code}")
            code_label.setStyleSheet("QLabel { color: #888; font-size: 11px; }")
            info_layout.addWidget(code_label)

            # ✅ CORRETTO - Calcola stats in modo sicuro
            owned_count = 0
            total_owned_copies = 0

            try:
                if (
                    account_name
                    and account_name != "All Accounts"
                    and account_name != t("ui.all_accounts")
                ):
                    # Query per un account specifico (Owned Count)
                    cursor.execute(
                        """
                        SELECT COUNT(DISTINCT c.id) as distinct_cards, COALESCE(SUM(ai.quantity), 0) as total_copies
                        FROM cards c
                        INNER JOIN account_inventory ai ON c.id = ai.card_id
                        -- ✅ CORREZIONE JOIN CRUCIALE: ai.account_id (stringa) = acc.device_account (PK)
                        INNER JOIN accounts acc ON ai.account_id = acc.device_account
                        WHERE c.set_code = ? AND ai.quantity > 0 AND acc.account_name = ?
                    """,
                        (set_code, account_name),
                    )
                else:
                    # Query per tutti i conti (non richiede la tabella accounts)
                    cursor.execute(
                        """
                        SELECT COUNT(DISTINCT c.id) as distinct_cards, COALESCE(SUM(ai.quantity), 0) as total_copies
                        FROM cards c
                        INNER JOIN account_inventory ai ON c.id = ai.card_id
                        WHERE c.set_code = ? AND ai.quantity > 0
                    """,
                        (set_code,),
                    )

                result = cursor.fetchone()
                if result:
                    owned_count = result[0] if result[0] else 0
                    total_owned_copies = int(result[1]) if result[1] else 0
                else:
                    owned_count = 0
                    total_owned_copies = 0

            except Exception as e:
                print(f"❌ Error calculating stats for {set_code}: {e}")
                import traceback

                traceback.print_exc()
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
            separator.setStyleSheet(
                "QFrame { background-color: #555; max-height: 1px; }"
            )
            content_widget.layout().addWidget(separator)

            # =========================================================================
            # GRID DELLE CARTE (Calcolo quantità per ogni carta)
            # =========================================================================

            try:
                # Query per ottenere le carte del set
                cursor.execute(
                    """
                    SELECT id, card_number, card_name, rarity, local_image_path
                    FROM cards
                    WHERE set_code = ?
                    ORDER BY CAST(card_number AS INTEGER)
                """,
                    (set_code,),
                )

                cards = cursor.fetchall()

                # Recupera wishlist
                cursor.execute("SELECT card_id FROM wishlist")
                wishlist_ids = set(row[0] for row in cursor.fetchall())

                # Initialise la structure pour ce set dans le filtrage
                if set_code not in self.collection_card_widgets:
                    self.collection_card_widgets[set_code] = {
                        "widgets": [],
                        "layout": None,
                    }

                # Container per le carte
                cards_widget = QWidget()
                cards_grid = QGridLayout(cards_widget)
                cards_grid.setSpacing(10)
                cards_grid.setContentsMargins(10, 10, 10, 10)

                # Stocke le layout per ce set
                self.collection_card_widgets[set_code]["layout"] = cards_grid

                # Crea widget per ogni carta
                row, col = 0, 0
                max_cols = 6

                for card_id, card_number, card_name, rarity, image_path in cards:
                    try:
                        # ✅ Recalcule quantity da BDD per ogni carta
                        quantity = 0
                        try:
                            if (
                                account_name
                                and account_name != "All Accounts"
                                and account_name != t("ui.all_accounts")
                            ):
                                # Per un account specifico
                                cursor.execute(
                                    """
                                    SELECT COALESCE(SUM(ai.quantity), 0) as total_qty
                                    FROM account_inventory ai
                                    -- ✅ CORREZIONE JOIN: ai.account_id = acc.device_account
                                    INNER JOIN accounts acc ON ai.account_id = acc.device_account
                                    WHERE ai.card_id = ? AND acc.account_name = ?
                                """,
                                    (card_id, account_name),
                                )
                            else:
                                # Per tutti i conti (invariato)
                                cursor.execute(
                                    """
                                    SELECT COALESCE(SUM(ai.quantity), 0) as total_qty
                                    FROM account_inventory ai
                                    WHERE ai.card_id = ?
                                """,
                                    (card_id,),
                                )

                            result = cursor.fetchone()
                            if result and result[0]:
                                quantity = int(result[0])

                        except Exception as e:
                            print(f"⚠️ Error fetching quantity for card {card_id}: {e}")
                            quantity = inventory.get(card_id, 0) if inventory else 0

                        is_wishlisted = card_id in wishlist_ids

                        card_data = {
                            "id": card_id,
                            "card_number": card_number,
                            "card_name": card_name,
                            "rarity": rarity,
                            "local_image_path": image_path,
                            "set_code": set_code,
                        }

                        card_widget = CardWidget(card_data, quantity, is_wishlisted)
                        card_widget.wishlist_changed.connect(self.on_wishlist_changed)

                        # Stocke les métadonnées per il filtrage
                        self.collection_card_widgets[set_code]["widgets"].append(
                            (card_widget, card_name, rarity, quantity, card_number)
                        )

                        cards_grid.addWidget(card_widget, row, col)

                        col += 1
                        if col >= max_cols:
                            col = 0
                            row += 1

                    except Exception as e:
                        print(f"❌ Error creating widget for card {card_id}: {e}")
                        import traceback

                        traceback.print_exc()
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
        """Gestisce il cambio di tab."""

        # Tab Discord Bot (Assumendo l'indice 0)
        if index == 0:
            # Se il bot NON è in esecuzione e la lista canali non è stata caricata
            if not (
                hasattr(self, "bot_thread")
                and self.bot_thread
                and self.bot_thread.isRunning()
            ):
                # Avvia il caricatore leggero per ricaricare la lista canali
                self.start_channel_loader()
        if index == 2:
            # Chiama il metodo sul widget della scheda, non su self
            if not self.collection_tab_widget.collection_loaded:
                print("📂 Caricando collection per la prima volta...")
                self.collection_tab_widget.collection_loaded = True
                self.collection_tab_widget.refresh_collection()
            else:
                print("✅ Collection già caricata")

    def on_language_changed(self, index):
        """Gestisce il cambio di lingua."""
        # Skip if called during initialization
        if not hasattr(self, "_language_combo_initialized"):
            return

        if hasattr(self, "language_combo"):
            new_language = self.language_combo.currentData()
            if new_language:
                set_language(new_language)
                self.save_settings()

                # Show message to restart application for full language change
                QMessageBox.information(
                    self,
                    t("ui.info"),
                    t("ui.language_changed_message", language=t("ui.language"))
                    + "\n\n"
                    + t("ui.restart_required"),
                )

    def save_settings(self):
        """Salva tutte le impostazioni nel settings.json principale."""
        try:
            selected_rarities = [
                rarity_name
                for rarity_name, widget in self.rarity_checkboxes.items()
                if widget.property("selected")
            ]
            if not selected_rarities:
                selected_rarities = SELECTED_RARITIES or list(RARITY_DATA.keys())

            settings = {
                "token": (
                    self.token_input.text() if hasattr(self, "token_input") else ""
                ),
                # ✅ SALVA SOLO LA LISTA DI ID SELEZIONATI (nuova struttura)
                "selected_channel_ids": (
                    list(self.selected_channel_ids)
                    if hasattr(self, "selected_channel_ids")
                    else []
                ),
                "autostart": (
                    self.autostart_cb.isChecked()
                    if hasattr(self, "autostart_cb")
                    else False
                ),
                "minimize_to_tray": (
                    self.minimize_tray_cb.isChecked()
                    if hasattr(self, "minimize_tray_cb")
                    else True
                ),
                "dark_theme": (
                    self.theme_combo.isChecked()
                    if hasattr(self, "theme_combo")
                    else True
                ),
                "language": (
                    self.language_combo.currentData()
                    if hasattr(self, "language_combo")
                    else DEFAULT_LANGUAGE
                ),
                "selected_rarities": selected_rarities,
                "last_updated": datetime.now().isoformat(),
                # ✅ NUOVI VALORI (Bot Folder, Notifiche)
                "bot_folder": (
                    self.bot_folder_input.text()
                    if hasattr(self, "bot_folder_input")
                    else ""
                ),
                "custom_notification_enabled": (
                    self.notification_enable_cb.isChecked()
                    if hasattr(self, "notification_enable_cb")
                    else False
                ),
                "notification_channel": (
                    self.notification_channel_input.text()
                    if hasattr(self, "notification_channel_input")
                    else ""
                ),
            }
            # ❌ Rimosso "channel_id" e "channel_configs" dalla scrittura per pulizia.

            settings_path = get_app_data_path("settings.json")
            os.makedirs(os.path.dirname(settings_path), exist_ok=True)

            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=4, ensure_ascii=False)

            print(f"✅ Settings saved!")

        except Exception as e:
            print(f"⚠️ Error saving settings: {e}")

    def load_settings(self):
        """Carica tutte le impostazioni dal settings.json principale."""
        try:
            from config import get_app_data_path

            settings_path = get_app_data_path("settings.json")
        except:
            settings_path = "settings.json"

        if not os.path.exists(settings_path):
            print("⚠️ Settings file not found, using defaults")
            # ... (gestione rarità di default)
            return

        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)

            # ✅ 1. CARICA BOT TOKEN
            if hasattr(self, "token_input"):
                token = settings.get("token", "")
                self.token_input.blockSignals(True)
                self.token_input.setText(token)
                self.token_input.blockSignals(False)
                print(
                    f"✅ Loaded bot token: {'*' * len(token) if token else '(empty)'}"
                )

            # ✅ 2. CARICA CHANNEL SELEZIONATI (Solo lo stato)
            self.selected_channel_ids = set()

            # 1. Prova la nuova chiave (prioritaria)
            if settings.get("selected_channel_ids"):
                self.selected_channel_ids = set(settings["selected_channel_ids"])

            # 2. Fallback Legacy: Vecchia struttura di Channel Configs (lista di dict)
            elif settings.get("channel_configs"):
                # Converte i vecchi dizionari in un set di ID
                self.selected_channel_ids = set(
                    c["id"] for c in settings["channel_configs"] if c.get("id")
                )

            # 3. Fallback Legacy 2: Vecchia stringa singola o multipla separata da virgola
            elif settings.get("channel_id") or settings.get("channel_ids"):
                id_string = settings.get("channel_id", "") or settings.get(
                    "channel_ids", ""
                )
                if id_string:
                    # Filtra e converte in interi
                    id_list = [
                        int(id_raw.strip())
                        for id_raw in id_string.split(",")
                        if id_raw.strip().isdigit()
                    ]
                    self.selected_channel_ids = set(id_list)

            print(f"✅ Loaded {len(self.selected_channel_ids)} selected channel IDs.")

            # ✅ 3. CARICA LINGUA
            if hasattr(self, "language_combo"):
                saved_lang = settings.get("language", "en")
                index = self.language_combo.findData(saved_lang)
                if index != -1:
                    self.language_combo.blockSignals(True)
                    self.language_combo.setCurrentIndex(index)
                    self.language_combo.blockSignals(False)
                    from .translations import set_language

                    set_language(saved_lang)
                    print(f"✅ Loaded language: {saved_lang}")

            # ✅ 4. CARICA TEMA
            if hasattr(self, "theme_combo"):
                dark_theme = settings.get("dark_theme", True)
                self.theme_combo.setChecked(dark_theme)
                print(f"✅ Loaded theme: {'Dark' if dark_theme else 'Light'}")

            # ✅ 5. CARICA AUTOSTART
            if hasattr(self, "autostart_cb"):
                autostart = settings.get("autostart", False)
                self.autostart_cb.setChecked(autostart)
                print(f"✅ Loaded autostart: {autostart}")

            # ✅ 6. CARICA MINIMIZE TO TRAY
            if hasattr(self, "minimize_tray_cb"):
                minimize_tray = settings.get("minimize_to_tray", True)
                self.minimize_tray_cb.setChecked(minimize_tray)
                print(f"✅ Loaded minimize to tray: {minimize_tray}")

            # ✅ 7. CARICA RARITÀ SELEZIONATE
            if hasattr(self, "rarity_checkboxes") and hasattr(self, "rarity_labels"):
                try:
                    from config import RARITY_DATA

                    selected_rarities = settings.get(
                        "selected_rarities", list(RARITY_DATA.keys())
                    )
                    print(f"✅ Loading rarities: {selected_rarities}")
                    for rarity_name, widget in self.rarity_checkboxes.items():
                        is_selected = rarity_name in selected_rarities
                        widget.setProperty("selected", is_selected)
                        label = self.rarity_labels.get(rarity_name)
                        if label:
                            if is_selected:
                                self.remove_grayscale_filter(label, rarity_name)
                            else:
                                self.apply_grayscale_filter(label, rarity_name)
                    print(f"✅ Loaded {len(selected_rarities)} selected rarities")
                except Exception as e:
                    print(f"⚠️ Error loading rarities: {e}")
                    import traceback

                    traceback.print_exc()

            # ✅ 8. CARICA I NUOVI VALORI (Bot Folder, Notifiche)
            if hasattr(self, "bot_folder_input"):
                bot_folder = settings.get("bot_folder", "")
                self.bot_folder_input.blockSignals(True)
                self.bot_folder_input.setText(bot_folder)
                self.bot_folder_input.blockSignals(False)
                print(f"✅ Loaded Bot Folder: {bot_folder}")

            if hasattr(self, "notification_enable_cb"):
                enable_notif = settings.get("custom_notification_enabled", False)
                self.notification_enable_cb.blockSignals(True)
                self.notification_enable_cb.setChecked(enable_notif)
                self.notification_enable_cb.blockSignals(False)
                print(f"✅ Loaded Enable Notifications: {enable_notif}")

            if hasattr(self, "notification_channel_input"):
                notif_channel = settings.get("notification_channel", "")
                self.notification_channel_input.blockSignals(True)
                self.notification_channel_input.setText(notif_channel)
                self.notification_channel_input.blockSignals(False)
                print(f"✅ Loaded Notification Channel: {notif_channel}")

            print("✅ All settings loaded successfully")

        except json.JSONDecodeError as e:
            print(f"❌ Settings file corrupted: {e}")
        except Exception as e:
            print(f"❌ Error loading settings: {e}")
            import traceback

            traceback.print_exc()

    # =========================================================================
    # WINDOW EVENTS
    # =========================================================================

    def closeEvent(self, event):
        """
        Gestisce la chiusura della finestra (click sulla 'X').
        Se 'Minimize to tray' è attivo, nasconde la finestra.
        Altrimenti, o se 'force_quit' è True, chiude l'app.
        """

        # Controlla se l'utente vuole veramente chiudere (dal menu tray)
        # O se l'opzione "minimize" è disattivata
        if self.force_quit or not self.minimize_tray_cb.isChecked():
            print("🛑 Avvio shutdown completo...")
            print("...attesa notifiche...")
            time.sleep(0.5)
            # 1. Svuota la coda di scrittura DB
            if hasattr(self, "db_writer_timer"):
                print("...svuotamento coda DB...")
                self.db_writer_timer.stop()
                self.process_db_write_queue()

            # 2. Ferma i thread principali
            print("...arresto thread...")
            if (
                hasattr(self, "bot_thread")
                and self.bot_thread
                and self.bot_thread.isRunning()
            ):
                self.bot_thread.stop()  # 'stop()' è il metodo corretto per DiscordBotThread
                self.bot_thread.wait(2000)

            if (
                hasattr(self, "scraper_tab_widget")
                and self.scraper_tab_widget.scraper_thread
                and self.scraper_tab_widget.scraper_thread.isRunning()
            ):
                self.scraper_tab_widget.stop_scraper()  # Usa il metodo della scheda
                self.scraper_tab_widget.scraper_thread.wait(2000)

            if (
                hasattr(self, "flask_thread")
                and self.flask_thread
                and self.flask_thread.isRunning()
            ):
                self.flask_thread.stop_server()
                self.flask_thread.wait(2000)

            if (
                hasattr(self, "tunnel_thread")
                and self.tunnel_thread
                and self.tunnel_thread.isRunning()
            ):
                self.tunnel_thread.stop_tunnel()
                self.tunnel_thread.wait(2000)

            print("✅ Shutdown completato. Chiusura.")
            event.accept()  # Permetti alla finestra di chiudersi

        else:
            # L'utente ha cliccato 'X' e 'Minimize to tray' è ATTIVO
            print("ℹ️ Minimizzazione nella tray icon...")
            event.ignore()  # Impedisci la chiusura
            self.hide()  # Nascondi la finestra

            # (Opzionale) Mostra una notifica
            if hasattr(self, "tray_icon"):
                self.tray_icon.showMessage(
                    "TCGP Team Rocket Tool",
                    t(
                        "ui.app_running_in_background"
                    ),  # "L'app è in esecuzione in background"
                    QSystemTrayIcon.Information,
                    2000,  # 2 secondi
                )

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
            if img.mode != "RGBA":
                img = img.convert("RGBA")

            # Crea overlay grigio scuro semitrasparente
            overlay = Image.new(
                "RGBA", img.size, (30, 30, 30, 180)
            )  # RGB + Alpha (0-255)

            # Combina immagine e overlay
            img_with_overlay = Image.alpha_composite(img, overlay)

            # Salva temporaneamente
            temp_bg_path = os.path.join("gui/background_with_overlay.png")
            img_with_overlay.save(temp_bg_path, "PNG")

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
        if hasattr(self, "background_label") and hasattr(self, "original_background"):
            window_size = self.size()

            # Scala in modalità cover
            scaled_pixmap = self.original_background.scaled(
                window_size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
            )

            self.background_label.setPixmap(scaled_pixmap)

            # Centra l'immagine
            x_offset = (scaled_pixmap.width() - window_size.width()) // 2
            y_offset = (scaled_pixmap.height() - window_size.height()) // 2
            self.background_label.setGeometry(
                -x_offset, -y_offset, scaled_pixmap.width(), scaled_pixmap.height()
            )

    def resizeEvent(self, event):
        """Chiamato quando la finestra viene ridimensionata."""
        super().resizeEvent(event)
        self.update_background_size()
