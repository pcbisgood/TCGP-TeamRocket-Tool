"""
Modulo ottimizzato per la scheda Account con design moderno e performance migliorate.
"""

# ============================================================================
# IMPORTS
# ============================================================================
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QScrollArea,
    QFrame, QPushButton, QSizePolicy, QToolButton, QMessageBox, QStyle,
    QLineEdit, QGraphicsDropShadowEffect, QLayout, QDialog, QApplication,
    QFileDialog  # Aggiungi questo
)
from PyQt5.QtCore import (
    Qt, pyqtSignal, QSize, pyqtSlot, QRunnable, QObject, QThreadPool, QRect, QPoint, QTimer
)
from PyQt5.QtGui import QPixmap, QIcon, QFont, QColor
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Dict, Optional, Tuple
import os
import sqlite3

from config import DB_FILENAME, get_resource_path, TCG_IMAGES_DIR
from .translations import t
from .database import DatabaseManager
from .sync_shiny_dust import sync_shiny_dust
from PyQt5.QtCore import  QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QMovie

if TYPE_CHECKING:
    from .ui_main_window import MainWindow

# ============================================================================
# CONSTANTS
# ============================================================================
SHINY_DUST_ICON_PATH = get_resource_path("gui/shiny_dust.png")
HOURGLASS_ICON_PATH = get_resource_path("gui/hourglass.png")

# Dimensioni widget
COVER_WIDTH = 120
COVER_HEIGHT = 160
TEXT_WIDTH = 150
MAX_COLS_VISIBLE = 6

# Colori tema
class Colors:
    PRIMARY = "#f39c12"
    SUCCESS = "#2ecc71"
    DANGER = "#e74c3c"
    INFO = "#3498db"
    MUTED = "#95a5a6"
    DARK_BG = "#252525"
    DARKER_BG = "#1a1a1a"
    BORDER = "#3a3a3a"

# ============================================================================
# STYLESHEETS
# ============================================================================
# ============================================================================
# STYLESHEETS
# ============================================================================
class Styles:
    MAIN_WIDGET = """
        CollapsibleAccountWidget {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #2d2d2d, stop:1 #252525);
            border: 1px solid #3a3a3a;
            border-radius: 12px;
            margin: 2px;
        }
    """
    
    HEADER_HOVER = """
        QFrame {
            background-color: transparent;
            border: none;
        }
        QFrame:hover {
            background-color: rgba(255, 255, 255, 0.03);
        }
    """
    
    # ✅ NUOVI STILI AGGIUNTI PER L'UNIFORMITÀ DELL'HEADER
    ACCOUNT_NAME_LABEL = "color: #ecf0f1; font-size: 15px; font-weight: bold;" # 15px
    ACCOUNT_ID_LABEL = "color: #bdc3c7; font-size: 11px; font-weight: normal;" # 11px
    
    
    ALIAS_INPUT = """
        QLineEdit {{
            background-color: rgba(52, 73, 94, 0.3);
            border: 2px solid transparent;
            border-radius: 6px;
            padding: 6px 10px;
            color: {color};
            font-size: 12px; /* ✅ Aumentato a 12px */
        }}
        QLineEdit:focus {{
            border: 2px solid #3498db;
            background-color: rgba(52, 73, 94, 0.5);
        }}
        QLineEdit::placeholder {{
            color: #7f8c8d;
            font-style: italic;
        }}
    """
    
    BADGE = """
        QLabel {{
            background-color: {bg_color};
            border: 1px solid {border_color};
            border-radius: 8px;
            padding: 5px 12px;
            color: {text_color};
            font-weight: bold;
        }}
    """
    
    CURRENCY_WIDGET = """
        QWidget {{
            background-color: rgba({rgb}, 0.15);
            border: 1px solid {color};
            border-radius: 8px;
        }}
    """
    
    TOGGLE_BUTTON = """
        QToolButton {
            background-color: rgba(52, 152, 219, 0.2);
            border: 2px solid #3498db;
            border-radius: 18px;
            padding: 8px;
        }
        QToolButton:hover {
            background-color: rgba(52, 152, 219, 0.4);
            border-color: #5dade2;
        }
        QToolButton:pressed {
            background-color: rgba(52, 152, 219, 0.6);
        }
    """
    
    SCROLLBAR = """
        QScrollArea {
            border: none;
            background: transparent;
        }
        QScrollBar:vertical {
            background: #2a2a2a;
            width: 12px;
            border-radius: 6px;
            margin: 2px;
        }
        QScrollBar::handle:vertical {
            background: #6a6a6a;
            border-radius: 6px;
            min-height: 30px;
        }
        QScrollBar::handle:vertical:hover {
            background: #8a8a8a;
        }
        QScrollBar::handle:vertical:pressed {
            background: #5a5a5a;
        }
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {
            border: none;
            background: none;
            height: 0px;
        }
        QScrollBar::add-page:vertical,
        QScrollBar::sub-page:vertical {
            background: none;
        }
        
        QScrollBar:horizontal {
            background: #2a2a2a;
            height: 12px;
            border-radius: 6px;
            margin: 2px;
        }
        QScrollBar::handle:horizontal {
            background: #6a6a6a;
            border-radius: 6px;
            min-width: 30px;
        }
        QScrollBar::handle:horizontal:hover {
            background: #8a8a8a;
        }
        QScrollBar::handle:horizontal:pressed {
            background: #5a5a5a;
        }
        QScrollBar::add-line:horizontal,
        QScrollBar::sub-line:horizontal {
            border: none;
            background: none;
            width: 0px;
        }
        QScrollBar::add-page:horizontal,
        QScrollBar::sub-page:horizontal {
            background: none;
        }
    """
    
    GLOBAL_HEADER = """
        QFrame {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #252525, stop:1 #2d2d2d);
            border: 2px solid #f39c12;
            border-radius: 10px;
            padding: 15px;
        }
    """
    
    REFRESH_BUTTON = """
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #f39c12, stop:1 #e67e22);
            color: black;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: bold;
        }
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #f1c40f, stop:1 #f39c12);
        }
        QPushButton:pressed {
            background: #d35400;
        }
        QPushButton:disabled {
            background: #7f8c8d;
            color: #bdc3c7;
        }
    """
# ============================================================================
# DATA CLASSES
# ============================================================================
@dataclass
class AccountData:
    account_id: str
    account_name: str
    alias: str
    shiny_dust: int
    hourglasses: int

@dataclass
class SetData:
    set_code: str
    set_name: str
    total: int
    owned: int
    cover_blob: Optional[bytes]

# ============================================================================
# WORKER SIGNALS
# ============================================================================
class WorkerSignals(QObject):
    accounts_ready = pyqtSignal(list, int)
    sets_ready = pyqtSignal(list, int, int)
    error = pyqtSignal(str)

# ============================================================================
# OPTIMIZED WORKERS
# ============================================================================
class AccountsLoaderWorker(QRunnable):
    """Worker ottimizzato per il caricamento account."""
    
    def __init__(self):
        super().__init__()
        self.signals = WorkerSignals()
    
    @pyqtSlot()
    def run(self):
        try:
            with sqlite3.connect(DB_FILENAME) as db:
                db.row_factory = sqlite3.Row
                cursor = db.cursor()
                
                # ✅ CORREZIONE DEFINITIVA: Seleziona la PK device_account e aliasala
                cursor.execute("""
                    SELECT device_account AS account_id, account_name AS account_name, alias, shiny_dust AS shiny_dust, hourglasses 
                    FROM accounts ORDER BY account_name ASC
                """)
                accounts = [dict(row) for row in cursor.fetchall()]
                
                cursor.execute("SELECT COUNT(*) FROM sets")
                total_sets = cursor.fetchone()[0]
                
                self.signals.accounts_ready.emit(accounts, total_sets)
        except Exception as e:
            self.signals.error.emit(f"Errore caricamento account: {e}")


# accounts_tab.py (Sostituisci l'intera classe SetDetailsWorker)

class SetDetailsWorker(QRunnable):
    """Worker ottimizzato con query JOIN singola invece di loop."""
    
    def __init__(self, account_id: int):
        super().__init__()
        self.account_id = account_id
        self.signals = WorkerSignals()
    
    @pyqtSlot()
    def run(self):
        try:
            with sqlite3.connect(DB_FILENAME) as db:
                db.row_factory = sqlite3.Row
                cursor = db.cursor()
                
                # ✅ MODIFICATO: Seleziona s.cover_image_blob
                cursor.execute("""
                    SELECT s.set_code, s.set_name, s.total_cards, s.cover_image_blob,
                        COALESCE(SUM(ai.quantity), 0) as owned_cards 
                    FROM sets s 
                    LEFT JOIN cards c ON c.set_code = s.set_code 
                    LEFT JOIN account_inventory ai ON ai.card_id = c.id AND ai.account_id = ? 
                    GROUP BY s.set_code, s.set_name, s.total_cards, s.cover_image_blob 
                    ORDER BY s.release_date DESC
                """, (self.account_id,))
                
                sets_data = []
                total_owned = 0
                total_cards = 0
                
                for row in cursor.fetchall():
                    owned = int(row['owned_cards'])
                    total = row['total_cards'] or 0
                    total_owned += owned
                    total_cards += total
                    
                    # ✅ Rimosso: logica path locale/remoto. Ora usiamo solo il BLOB.
                    cover_blob = row['cover_image_blob']
                    
                    sets_data.append(SetData(
                        set_code=row['set_code'],
                        set_name=row['set_name'],
                        total=total,
                        owned=owned,
                        cover_blob=cover_blob # ✅ PASSA IL BLOB
                    ))
                
                self.signals.sets_ready.emit(sets_data, total_owned, total_cards)
                
        except Exception as e:
            self.signals.error.emit(f"Errore caricamento set: {e}")
            import traceback
            print(traceback.format_exc())

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def hex_to_rgb(hex_color: str) -> str:
    """Converte colore hex in stringa RGB."""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"{r}, {g}, {b}"


def get_progress_colors(owned: int, total: int) -> Tuple[str, str]:
    """Ritorna colori basati sul progresso (text_color, bg_color)."""
    if owned == total and total > 0:
        return Colors.SUCCESS, f"rgba(46, 204, 113, 0.2)"
    elif owned > 0:
        return Colors.PRIMARY, f"rgba(243, 156, 18, 0.2)"
    else:
        return Colors.MUTED, f"rgba(149, 165, 166, 0.2)"


def create_shadow_effect(blur_radius: int = 15, offset: Tuple[int, int] = (0, 3)) -> QGraphicsDropShadowEffect:
    """Crea un effetto ombra riusabile."""
    shadow = QGraphicsDropShadowEffect()
    shadow.setBlurRadius(blur_radius)
    shadow.setColor(QColor(0, 0, 0, 80))
    shadow.setOffset(*offset)
    return shadow

# ============================================================================
# COLLAPSIBLE ACCOUNT WIDGET
# ============================================================================
class CollapsibleAccountWidget(QWidget):
    """Widget ottimizzato per singolo account con lazy loading."""
    
    expand_signal = pyqtSignal()
    

    
    # accounts_tab.py (Dentro la classe CollapsibleAccountWidget)

    def __init__(self, main_window, account, total_sets):
        """Inizializza il widget collapsibile."""
        super().__init__()
        self.main_window = main_window
        self.account_data = account
        self.total_sets = total_sets
        self.is_expanded = False
        
        # ✅ CORREZIONE DEFINITIVA: STILE BADGE UNIFORME
        self.total_cards_label = QLabel()
        self.total_cards_label.setAlignment(Qt.AlignCenter)
        self.total_cards_label.setFixedHeight(28) 
        self.total_cards_label.setFixedWidth(85) # Imposta una larghezza fissa per stabilità
        
        # Stile di base del badge (sarà aggiornato con i colori)
        self.BASE_BADGE_STYLE = """
            QLabel {{
                font-weight: bold;
                border-radius: 9px;
                font-size: 11px;
                padding: 4px 10px;
                margin: 0 3px;
                
                /* Questi colori saranno formattati da _update_badge_style */
                color: {text_color};
                background-color: {bg_color};
                border: 1px solid {border_color};
            }}
        """
        self.total_cards_label.setFont(QFont("Segoe UI", 11, QFont.Bold)) 
        self.total_cards_label.setText(f"(0/{self.total_sets})")
        
        # Applica lo stile iniziale (colore MUTED)
        self._update_badge_style(self.total_cards_label, 0, 1)

        
        self.currencies_section = self._create_currencies_section()
        
        # Setup UI
        self._setup_ui()
 
    # accounts_tab.py (Dentro la classe CollapsibleAccountWidget)

    def _setup_ui(self):
        """Setup UI della tab Account."""
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Header
        self.header_frame = self._create_header()
        self.main_layout.addWidget(self.header_frame)
        
        # Content area (Area che contiene i set, il placeholder e i suoi margini)
        self.content_area = QWidget()
        self.content_layout = QVBoxLayout(self.content_area)
        
        # ✅ CORREZIONE: Imposta margini e spacing a zero sul layout del contenuto.
        # Questo elimina lo spazio riservato al Content Widget quando è nascosto.
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(0)
        
        self.main_layout.addWidget(self.content_area)
        
        # ❌ RIMOSSO: Label per spinner (come richiesto)
        
        # ✅ CORREZIONE FINALE: Forza la chiusura completa all'inizio
        self.content_area.hide()


# accounts_tab.py (Dentro la classe CollapsibleAccountWidget)

    def _load_set_image_from_blob(self, cover_blob: bytes, label: QLabel):
        """Carica l'immagine dal BLOB e la imposta sul QLabel."""
        try:
            if not cover_blob:
                raise ValueError("BLOB non disponibile.")
            
            pixmap = QPixmap()
            pixmap.loadFromData(cover_blob)
            
            if not pixmap.isNull():
                # Scala mantenendo le proporzioni
                scaled_pixmap = pixmap.scaled(
                    COVER_WIDTH, COVER_HEIGHT, 
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                label.setPixmap(scaled_pixmap)
                label.setFixedSize(scaled_pixmap.size()) # Imposta la dimensione corretta
                label.setText("") # Rimuovi il placeholder
            else:
                raise Exception("Impossibile caricare QPixmap dal BLOB.")
                
        except Exception as e:
            # print(f"❌ Errore caricamento BLOB cover: {e}")
            # Fallback al messaggio di errore
            label.setFixedSize(COVER_WIDTH, COVER_HEIGHT)
            label.setText("❌\nBLOB Err")
            label.setStyleSheet("""
                QLabel {
                    color: #e74c3c; 
                    border: 2px dashed #c0392b;
                    border-radius: 6px;
                    background-color: rgba(231, 76, 60, 0.1);
                    font-size: 10px;
                }
            """)


    def _apply_styles(self):
        """Applica stili."""
        self.setStyleSheet(Styles.MAIN_WIDGET)
        self.setGraphicsEffect(create_shadow_effect())
    

    # accounts_tab.py (Dentro la classe CollapsibleAccountWidget)

    def _create_header(self) -> QFrame:
        """Crea header ottimizzato."""
        header = QFrame()
        header.setStyleSheet(Styles.HEADER_HOVER)
        header.setCursor(Qt.PointingHandCursor)
        header.mousePressEvent = lambda event: self.toggle_content()
        
        layout = QHBoxLayout(header)
        # ✅ CORREZIONE: Rimuovi padding verticale eccessivo (15, 10, 15, 10)
        layout.setContentsMargins(15, 6, 15, 6) 
        layout.setSpacing(12)
        
        # Left - Nome, ID, Alias
        layout.addWidget(self._create_name_section())
        layout.addStretch()
        
        # Center - Totali carte
        layout.addWidget(self.total_cards_label)
        
        # Right - Valute
        layout.addWidget(self.currencies_section)
        
        return header


    def _handle_inject_account(self):
        """Gestisce l'injection dell'account - riusa metodi esistenti."""
        try:
            # Recupera credenziali 
            with sqlite3.connect(DB_FILENAME) as db:
                cursor = db.cursor()
                cursor.execute("""
                    SELECT device_account, device_password 
                    FROM accounts WHERE device_account = ?
                """, (self.account_data.account_id,)) 
                result = cursor.fetchone()
                
                if not result:
                    QMessageBox.warning(
                        self.main_window,
                        "Errore",
                        "Impossibile recuperare le credenziali dell'account."
                    )
                    return
                
                device_account, device_password = result
                
                if not device_account or not device_password:
                    QMessageBox.warning(
                        self.main_window,
                        "Dati Mancanti",
                        f"Le credenziali deviceAccount o devicePassword non sono impostate per l'account '{self.account_data.account_name}'.\n\n"
                        "Aggiungile nella gestione account."
                    )
                    return
            
            # Importa i dialog necessari
            from .ui_dialogs import SimpleSelectionDialog, InjectAccountDialog
            from .ui_widgets import CollectionCardDialog
            from pathlib import Path
            
            # Crea un'istanza temporanea per accedere ai metodi
            # Usa db_manager fittizio
            temp_dialog = CollectionCardDialog(
                card_id=0,
                db_manager=None,  # Non serve per questi metodi
                parent=self.main_window
            )
            
            # Definisci percorsi ADB
            adb_path_str = "C:\\Program Files\\Netease\\MuMuPlayerGlobal-12.0\\shell\\adb.exe"
            adb_path_obj = Path(adb_path_str)
            
            if not adb_path_obj.exists():
                # Prova percorsi alternativi
                alternative_paths = [
                    r"C:\Program Files (x86)\Netease\MuMuPlayer-12.0\shell\adb.exe",
                    r"D:\Program Files\Netease\MuMuPlayer-12.0\shell\adb.exe",
                ]
                
                found = False
                for alt_path in alternative_paths:
                    if Path(alt_path).exists():
                        adb_path_str = alt_path
                        adb_path_obj = Path(alt_path)
                        found = True
                        break
                
                if not found:
                    from PyQt5.QtWidgets import QFileDialog
                    
                    reply = QMessageBox.question(
                        self.main_window,
                        "ADB Non Trovato",
                        "Il percorso ADB non è stato trovato automaticamente.\n\n"
                        "Vuoi selezionare manualmente il file adb.exe?",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    
                    if reply == QMessageBox.Yes:
                        adb_path_str, _ = QFileDialog.getOpenFileName(
                            self.main_window,
                            "Seleziona adb.exe",
                            "C:\\Program Files",
                            "File eseguibile (*.exe)"
                        )
                        
                        if not adb_path_str:
                            return
                        
                        adb_path_obj = Path(adb_path_str)
                    else:
                        return
            
            adb_path = str(adb_path_obj)
            mumu_player_path = str(adb_path_obj.parent.parent)
            
            # USA I METODI DELLA CLASSE CollectionCardDialog
            QApplication.setOverrideCursor(Qt.WaitCursor)
            configured_instances = temp_dialog._get_mumu_instances(mumu_player_path)
            connected_device_ids = temp_dialog._run_adb_devices(adb_path)
            QApplication.restoreOverrideCursor()
            
            if not configured_instances:
                QMessageBox.warning(
                    self.main_window,
                    "Errore Configurazione",
                    f"Nessuna istanza MuMu trovata in:\n{mumu_player_path}"
                )
                return
            
            active_ports_set = {dev_id.split(":")[-1] for dev_id in connected_device_ids}
            
            # Filtra istanze attive
            active_instances = {
                name: port for name, port in configured_instances.items()
                if port in active_ports_set
            }
            
            if not active_instances:
                QMessageBox.warning(
                    self.main_window,
                    "Nessuna Istanza Attiva",
                    "Nessuna istanza MuMu risulta connessa tramite ADB.\n\n"
                    "Avvia almeno un'istanza e riprova."
                )
                return
            
            # USA IL METODO STATICO DI SELEZIONE da ui_dialogs
            selected_port = SimpleSelectionDialog.get_selected_port(
                configured_instances,
                active_ports_set,
                self.main_window
            )
            
            if not selected_port:
                return  # Utente ha annullato
            
            # Trova il nome dell'istanza
            selected_name = None
            for name, port in configured_instances.items():
                if port == selected_port:
                    selected_name = name
                    break
            
            if not selected_name:
                selected_name = f"Istanza {selected_port}"
            
            # Crea file XML temporaneo
            import tempfile
            
            xml_content = f"""<?xml version="1.0" encoding="utf-8" standalone="yes" ?>
    <map>
        <string name="deviceAccount">{device_account}</string>
        <string name="devicePassword">{device_password}</string>
    </map>"""
            
            temp_xml_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".xml", mode='w', encoding='utf-8') as temp_file:
                    temp_file.write(xml_content)
                    temp_xml_path = temp_file.name
                
                # USA IL DIALOG DI INJECTION da ui_dialogs
                inject_dialog = InjectAccountDialog(
                    account_name=self.account_data.account_name,
                    temp_xml_path=temp_xml_path,
                    adb_path=adb_path,
                    selected_port=selected_port,
                    account_id=self.account_data.account_id,
                    card_id=0, 
                    parent=self.main_window
                )
                inject_dialog.exec_()
                
            except Exception as e:
                QMessageBox.critical(
                    self.main_window,
                    "Errore Creazione File Temporaneo",
                    f"Impossibile creare il file XML temporaneo:\n{e}"
                )
            finally:
                # Cleanup file temporaneo
                if temp_xml_path and os.path.exists(temp_xml_path):
                    try:
                        os.remove(temp_xml_path)
                    except:
                        pass
                        
        except Exception as e:
            QMessageBox.critical(
                self.main_window,
                "Errore",
                f"Errore durante l'injection:\n{e}"
            )
            import traceback
            traceback.print_exc()




    def _execute_injection(self, adb_path: str, selected_port: str, temp_xml_path: str, selected_name: str):
        """Esegue i comandi ADB per l'injection."""
        import subprocess
        
        device_id = f"127.0.0.1:{selected_port}"
        remote_path = "/data/data/com.YoStarEN.Arknights/shared_prefs/SdkDeviceId.xml"
        
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        try:
            # 1. Push file
            push_result = subprocess.run(
                [adb_path, "-s", device_id, "push", temp_xml_path, remote_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if push_result.returncode != 0:
                raise Exception(f"Push fallito: {push_result.stderr}")
            
            # 2. Kill app
            kill_result = subprocess.run(
                [adb_path, "-s", device_id, "shell", "am", "force-stop", "com.YoStarEN.Arknights"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            QApplication.restoreOverrideCursor()
            
            QMessageBox.information(
                self.main_window,
                "Injection Completata",
                f"Account '{self.account_data.account_name}' iniettato con successo su:\n{selected_name} (porta {selected_port})\n\n"
                "L'app è stata chiusa. Riaprila per vedere l'account."
            )
            
        except subprocess.TimeoutExpired:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(
                self.main_window,
                "Timeout",
                "Operazione ADB scaduta. Controlla che l'istanza sia attiva."
            )
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(
                self.main_window,
                "Errore Injection",
                f"Errore durante l'injection:\n{e}"
            )


    def _get_mumu_instances(self, mumu_player_path: str) -> Dict[str, str]:
        """Legge i file vmconfig.json per la porta e extraconfig.json per playerName di MuMu."""
        from pathlib import Path
        import json
        
        instances_map = {}
        vms_path_found = Path(mumu_player_path) / "vms"
        
        if not vms_path_found.exists():
            print(f"❌ ERRORE: Cartella vms non trovata in {vms_path_found}")
            return {}
        
        print(f"🔍 Scansione istanze in {vms_path_found}")
        
        for vm_folder in vms_path_found.iterdir():
            if not vm_folder.is_dir() or vm_folder.name.endswith("-base"):
                continue
            
            print(f"--- Trovata cartella istanza: {vm_folder.name}")
            
            config_dir = vm_folder / "configs"
            vm_config_path = config_dir / "vmconfig.json"
            extra_config_path = config_dir / "extraconfig.json"
            
            if not vm_config_path.exists():
                print(f"⚠ AVVISO: vmconfig.json non trovato in {config_dir}. Salto.")
                continue
            
            port = None
            name = None
            
            try:
                # 1. Leggi la porta da vmconfig.json
                with open(vm_config_path, 'r', encoding='utf-8') as f:
                    vm_config_data = json.load(f)
                    try:
                        port = vm_config_data["vm"]["nat"]["port_forward"]["adb"]["host_port"]
                    except KeyError:
                        print(f"❌ ERRORE: Impossibile trovare la porta ADB in {vm_config_path.name}")
                        continue
                
                # 2. Leggi il nome da extraconfig.json se esiste
                if extra_config_path.exists():
                    with open(extra_config_path, 'r', encoding='utf-8') as f:
                        extra_config_data = json.load(f)
                        name = extra_config_data.get("playerName")
                        print(f"✅ Trovato playerName: {name}")
                else:
                    print(f"⚠ AVVISO: extraconfig.json non trovato, uso nome di fallback.")
                
                # 3. Fallback se playerName non è trovato
                if not name:
                    name = vm_folder.name  # Fallback finale al nome cartella
                
                print(f"✅ Istanza '{name}' usa porta {port}")
                instances_map[name] = str(port)
                
            except Exception as e:
                print(f"❌ ERRORE lettura JSON per {vm_folder.name}: {e}")
                continue
        
        if not instances_map:
            print("⚠ Scansione completata, ma nessuna istanza valida trovata.")
        
        return instances_map

    def _run_adb_devices(self, adb_path: str) -> set:
        """Esegue adb devices e restituisce un set di porte connesse."""
        import subprocess
        
        connected_ports = set()
        
        try:
            command = [adb_path, "devices"]
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                timeout=5
            )
            
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip().endswith("device"):
                    device_id = line.split()[0]
                    if device_id.startswith("127.0.0.1"):
                        connected_ports.add(device_id)
                        
        except FileNotFoundError:
            QMessageBox.warning(
                self.main_window,
                "Errore ADB",
                f"Comando ADB non trovato al percorso:\n{adb_path}\n\n"
                "Controlla il percorso nelle impostazioni."
            )
        except subprocess.TimeoutExpired:
            QMessageBox.warning(
                self.main_window,
                "Errore ADB",
                "Timeout durante l'esecuzione di 'adb devices'."
            )
        except Exception as e:
            QMessageBox.warning(
                self.main_window,
                "Errore ADB",
                f"Errore scansione ADB:\n{e}"
            )
        
        return connected_ports

    def filter_accounts(self, text: str) -> List[dict]:
        text = text.lower().strip()
        results = [
            acc for acc in self.all_accounts
            if text in acc['account_name'].lower()
            or text in acc['account_id'].lower()
            or (acc.get('alias') or '').lower().find(text) != -1
        ]
        return results


    def _on_alias_edit(self):
        new_alias = self.alias_edit.text().strip()
        if new_alias != (self.account_data.get('alias') or ""):
            self.account_data['alias'] = new_alias
            # Salva nel DB (dict/row → chiave/colonna 'alias')
            with sqlite3.connect(DB_FILENAME) as db:
                cursor = db.cursor()
                cursor.execute("UPDATE accounts SET alias = ? WHERE device_account = ?", (new_alias, self.account_data['account_id']))
                db.commit()

# accounts_tab.py (Dentro la classe CollapsibleAccountWidget)

    # accounts_tab.py (Dentro la classe CollapsibleAccountWidget)

    def _create_name_section(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Nome account
        name_label = QLabel(self.account_data['account_name'])
        name_label.setFont(QFont("Segoe UI", 14, QFont.Bold)) 
        name_label.setStyleSheet(Styles.ACCOUNT_NAME_LABEL) 
        layout.addWidget(name_label)

        # ID account
        account_id_label = QLabel(f"ID: {self.account_data['account_id']}")
        account_id_label.setFont(QFont("Segoe UI", 11, QFont.Normal)) 
        account_id_label.setStyleSheet(Styles.ACCOUNT_ID_LABEL) 
        layout.addWidget(account_id_label)

        # Alias (EDITABILE)
        self.alias_edit = QLineEdit(self.account_data.get('alias') or "")
        self.alias_edit.setPlaceholderText("Alias account…")
        self.alias_edit.setFont(QFont("Segoe UI", 12, QFont.StyleItalic)) 
        
        # Applica stile Alias
        alias_style = Styles.ALIAS_INPUT.format(color=Colors.PRIMARY).replace(
            "border: 2px solid transparent;", 
            "border: none;"
        ).replace(
            "padding: 6px 10px;", 
            "padding: 3px 6px;" 
        )
        self.alias_edit.setStyleSheet(alias_style)
        self.alias_edit.editingFinished.connect(self._on_alias_edit)
        layout.addWidget(self.alias_edit)

        widget.setLayout(layout)
        return widget


    
    def _update_badge_style(self, label: QLabel, owned: int, total: int):
        """Aggiorna stile badge basato su progresso mantenendo le dimensioni fisse."""
        text_color, bg_color = get_progress_colors(owned, total)
        
        # ✅ AGGIORNAMENTO DEFINITIVO: Usa lo stile base con i nuovi colori
        style = self.BASE_BADGE_STYLE.format(
            bg_color=bg_color,
            border_color=text_color,
            text_color=text_color
        )
        label.setStyleSheet(style)
    


    def _create_currencies_section(self) -> QWidget:
        """Crea la sezione con shiny_dust e hourglasses."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        # ================================================================
        # SHINY DUST - CON ICONA PNG
        # ================================================================
        dust_icon_label = QLabel()
        dust_pixmap = QPixmap(SHINY_DUST_ICON_PATH)
        # ✅ Ridimensiona l'icona a 24x24 pixel
        dust_pixmap = dust_pixmap.scaledToHeight(24, Qt.SmoothTransformation)
        dust_icon_label.setPixmap(dust_pixmap)
        
        dust_label = QLabel(f"{self.account_data.get('shiny_dust') or 0}")
        dust_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        dust_label.setStyleSheet(f"color: {Colors.PRIMARY};")
        
        layout.addWidget(dust_icon_label)
        layout.addWidget(dust_label)
        layout.addSpacing(15)
        
        # ================================================================
        # HOURGLASSES - CON ICONA PNG
        # ================================================================
        hourglasses_icon_label = QLabel()
        hourglasses_pixmap = QPixmap(HOURGLASS_ICON_PATH)
        # ✅ Ridimensiona l'icona a 24x24 pixel
        hourglasses_pixmap = hourglasses_pixmap.scaledToHeight(24, Qt.SmoothTransformation)
        hourglasses_icon_label.setPixmap(hourglasses_pixmap)
        
        hourglasses_label = QLabel(f"{self.account_data.get('hourglasses') or 0}")
        hourglasses_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        hourglasses_label.setStyleSheet("color: #ffc107;")
        
        layout.addWidget(hourglasses_icon_label)
        layout.addWidget(hourglasses_label)
        
        return widget
    
    def _create_currency_widget(self, icon_path: str, value: str, tooltip: str, color: str) -> QWidget:
        """Crea widget valuta riusabile senza bordo."""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)
        
        # Icona
        icon_label = QLabel()
        icon_pixmap = QPixmap(icon_path)
        if not icon_pixmap.isNull():
            icon_label.setPixmap(icon_pixmap.scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        layout.addWidget(icon_label)
        
        # Valore
        value_label = QLabel(value)
        value_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        value_label.setStyleSheet(f"color: {color};")
        layout.addWidget(value_label)
        
        container.setToolTip(tooltip)
        # RIMUOVI IL BORDO - solo background trasparente o leggero
        container.setStyleSheet("""
            QWidget {
                background-color: transparent;
                border: none;
            }
        """)
        
        return container

    
    def _create_toggle_button(self) -> QToolButton:
        """Crea pulsante toggle."""
        button = QToolButton()
        button.setIcon(self.main_window.style().standardIcon(QStyle.SP_ArrowDown))
        button.setIconSize(QSize(20, 20))
        button.setStyleSheet(Styles.TOGGLE_BUTTON)
        button.clicked.connect(self.toggle_content)
        return button
    
    def _update_alias_style(self, text: str):
        """Aggiorna stile alias dinamicamente."""
        color = Colors.PRIMARY if text else Colors.MUTED
        self.alias_input.setStyleSheet(Styles.ALIAS_INPUT.format(color=color))
    
    def _save_alias(self):
        """Salva alias nel database."""
        new_alias = self.alias_input.text().strip()
        if new_alias == self.account_data.alias:
            return
        
        try:
            with sqlite3.connect(DB_FILENAME) as db:
                cursor = db.cursor()
                cursor.execute(
                    "UPDATE accounts SET alias = ? WHERE account_id = ?",
                    (new_alias, self.account_data.account_id)
                )
                db.commit()
                self.account_data.alias = new_alias
        except Exception as e:
            QMessageBox.critical(
                self.main_window,
                "Errore",
                f"Impossibile salvare l'alias: {e}"
            )
    
    def toggle_content(self):
        """Toggle con lazy loading (senza toggle button)."""
        if self.is_expanded:
            self.content_area.hide()
            self.is_expanded = False
            # Aggiorna badge con freccia destra
            current_text = self.total_cards_label.text()
            if " ▼" in current_text:
                self.total_cards_label.setText(current_text.replace(" ▼", " ▶"))
        else:
            
            self._load_sets_lazy()
            self.content_area.show()
            self.is_expanded = True
            # Aggiorna badge con freccia giù
            current_text = self.total_cards_label.text()
            if " ▶" in current_text:
                self.total_cards_label.setText(current_text.replace(" ▶", " ▼"))
            self.expand_signal.emit()
    
    def _load_sets_lazy(self):
        """Carica i set solo quando necessario (lazy loading)."""
        placeholder = QLabel("⏳ Caricamento...")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("color: #95a5a6; font-size: 13px; padding: 30px;")
        self.content_layout.addWidget(placeholder)
        
        worker = SetDetailsWorker(self.account_data['account_id'])
        worker.signals.sets_ready.connect(
            lambda sets, owned, total: self._on_sets_loaded(sets, owned, total, placeholder)
        )
        worker.signals.error.connect(
            lambda msg: QMessageBox.critical(self.main_window, "Errore", msg)
        )
        self.main_window.image_loader_pool.start(worker)
    
    @pyqtSlot(list, int, int)
    def _on_sets_loaded(self, sets_data: List[SetData], total_owned: int, total_cards: int, placeholder: QLabel):
        """Callback quando i set sono caricati."""
        self.sets_loaded = True
        
        # Rimuovi placeholder
        self.content_layout.removeWidget(placeholder)
        placeholder.deleteLater()
        
        # Aggiorna totali
        self.total_cards_label.setText(f"({total_owned}/{total_cards})")
        self._update_badge_style(self.total_cards_label, total_owned, total_cards)
        
        # Container principale con scroll
        scroll_widget = QWidget()
        scroll_widget.setStyleSheet("""
            QWidget {
                background-color: rgba(0, 0, 0, 0.1);
                border-radius: 8px;
            }
        """)
        
        # Layout FLOW (wrapping automatico)
        from PyQt5.QtWidgets import QLayout
        
        class FlowLayout(QLayout):
            """Layout che wrappa automaticamente i widget."""
            def __init__(self, parent=None, margin=0, spacing=-1):
                super().__init__(parent)
                self.itemList = []
                self.m_hSpace = spacing
                self.m_vSpace = spacing
                self.setContentsMargins(margin, margin, margin, margin)
            
            def __del__(self):
                item = self.takeAt(0)
                while item:
                    item = self.takeAt(0)
            
            def addItem(self, item):
                self.itemList.append(item)
            
            def count(self):
                return len(self.itemList)
            
            def itemAt(self, index):
                if 0 <= index < len(self.itemList):
                    return self.itemList[index]
                return None
            
            def takeAt(self, index):
                if 0 <= index < len(self.itemList):
                    return self.itemList.pop(index)
                return None
            
            def expandingDirections(self):
                return Qt.Orientations(Qt.Orientation(0))
            
            def hasHeightForWidth(self):
                return True
            
            def heightForWidth(self, width):
                height = self._doLayout(QRect(0, 0, width, 0), True)
                return height
            
            def setGeometry(self, rect):
                super().setGeometry(rect)
                self._doLayout(rect, False)
            
            def sizeHint(self):
                return self.minimumSize()
            
            def minimumSize(self):
                size = QSize()
                for item in self.itemList:
                    size = size.expandedTo(item.minimumSize())
                margin = self.contentsMargins().left()
                size += QSize(2 * margin, 2 * margin)
                return size
            
            def _doLayout(self, rect, testOnly):
                x = rect.x()
                y = rect.y()
                lineHeight = 0
                spacing = self.spacing()
                
                for item in self.itemList:
                    wid = item.widget()
                    spaceX = spacing
                    spaceY = spacing
                    
                    nextX = x + item.sizeHint().width() + spaceX
                    if nextX - spaceX > rect.right() and lineHeight > 0:
                        x = rect.x()
                        y = y + lineHeight + spaceY
                        nextX = x + item.sizeHint().width() + spaceX
                        lineHeight = 0
                    
                    if not testOnly:
                        item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))
                    
                    x = nextX
                    lineHeight = max(lineHeight, item.sizeHint().height())
                
                return y + lineHeight - rect.y()
            
            def spacing(self):
                if self.m_hSpace >= 0:
                    return self.m_hSpace
                else:
                    return self.smartSpacing()
            
            def smartSpacing(self):
                parent = self.parent()
                if not parent:
                    return -1
                elif parent.isWidgetType():
                    return parent.style().pixelMetric(QStyle.PM_LayoutHorizontalSpacing, None, parent)
                else:
                    return parent.spacing()
        
        # Usa FlowLayout
        flow_layout = FlowLayout(scroll_widget, margin=15, spacing=25)
        
        # Aggiungi tutti i set
        for set_data in sets_data:
            set_widget = self._create_set_widget(set_data)
            flow_layout.addWidget(set_widget)
        
        self.content_layout.addWidget(scroll_widget)




    # accounts_tab.py (Sostituisci l'intero metodo _create_set_widget)

    def _create_set_widget(self, set_data: SetData) -> QWidget:
        """Crea widget set con tutto centrato."""
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        
        # === COVER LABEL ===
        cover_label = QLabel()
        cover_label.setAlignment(Qt.AlignCenter)
        cover_label.setScaledContents(False)
        cover_label.setStyleSheet("QLabel { border: none; background: transparent; }")
        
        # ✅ NUOVA LOGICA DI CARICAMENTO BLOB
        if set_data.cover_blob:
            # Caricamento BLOB sincrono (molto veloce)
            self._load_set_image_from_blob(set_data.cover_blob, cover_label)
        else:
            cover_label.setFixedSize(COVER_WIDTH, COVER_HEIGHT)
            cover_label.setText("🖼️\nNo Cover")
            cover_label.setStyleSheet("""
                QLabel {
                    color: #7f8c8d; 
                    border: 2px dashed #555;
                    border-radius: 6px;
                    background-color: rgba(52, 73, 94, 0.2);
                    font-size: 11px;
                }
            """)
        # ❌ Rimosso: self._load_set_image_original(set_data.cover_path, cover_label)
        # ❌ Rimosso: tutta la logica di download/cache per l'URL
        
        layout.addWidget(cover_label, 0, Qt.AlignHCenter)
        
        # === NOME SET ===
        name_label = QLabel(set_data.set_name)
        name_label.setAlignment(Qt.AlignCenter)
        name_label.setWordWrap(True)
        name_label.setFixedWidth(max(COVER_WIDTH, TEXT_WIDTH))
        name_label.setMaximumHeight(35)
        name_label.setStyleSheet("""
            color: #f39c12; 
            font-weight: bold; 
            font-size: 11px; 
            background: transparent;
        """)
        layout.addWidget(name_label, 0, Qt.AlignHCenter)
        
        # === SET CODE ===
        code_label = QLabel(set_data.set_code)
        code_label.setAlignment(Qt.AlignCenter)
        code_label.setFixedWidth(max(COVER_WIDTH, TEXT_WIDTH))
        code_label.setStyleSheet("""
            color: #95a5a6; 
            font-size: 10px; 
            font-style: italic; 
            background: transparent;
        """)
        layout.addWidget(code_label, 0, Qt.AlignHCenter)
        
        # === PROGRESS ===
        progress_label = QLabel(f"{set_data.owned} / {set_data.total}")
        progress_label.setAlignment(Qt.AlignCenter)
        progress_label.setFixedWidth(min(COVER_WIDTH - 10, TEXT_WIDTH - 20))
        progress_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        
        text_color, bg_color = get_progress_colors(set_data.owned, set_data.total)
        progress_label.setStyleSheet(f"""
            color: {text_color};
            font-weight: bold;
            background-color: {bg_color};
            border: 1px solid {text_color};
            border-radius: 6px;
            padding: 5px 8px;
        """)
        layout.addWidget(progress_label, 0, Qt.AlignHCenter)
        
        container.setStyleSheet("QWidget { background: transparent; }")
        container.setToolTip(f"{set_data.set_name}\n{set_data.set_code}")
        
        # Imposta dimensione fissa del container basata sul contenuto più largo
        max_width = max(COVER_WIDTH, TEXT_WIDTH) + 16  # +16 per margini
        container.setFixedWidth(max_width)
        
        return container




    

# ============================================================================
# ACCOUNTS TAB
# ============================================================================
class AccountsTab(QWidget):
    """Tab principale ottimizzato."""
    
    def __init__(self, main_window: 'MainWindow'):
        super().__init__()
        self.main_window = main_window
        self.thread_pool = QThreadPool()
        self.accounts_widgets: List[CollapsibleAccountWidget] = []
        self.all_accounts = []  # Tutti gli account caricati
        self.displayed_accounts_count = 0  # Quanti sono mostrati
        self.accounts_per_page = 20  # Mostra 20 alla volta
        self.total_sets = 0       
        self.first_tab_entry = True 

  
        self.thread_pool = QThreadPool()
        self._setup_ui()
        self.load_accounts()
    
    def _setup_ui(self):
        """Setup UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Header globale
        main_layout.addWidget(self._create_global_header())
        main_layout.addWidget(self._create_search_bar())
        
        # Scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setStyleSheet(Styles.SCROLLBAR)
        
        self.accounts_container = QWidget()
        self.accounts_layout = QVBoxLayout(self.accounts_container)
        self.accounts_layout.setContentsMargins(0, 0, 0, 0)
        self.accounts_layout.setSpacing(0)
        self.accounts_layout.addStretch(1)
        
        self.scroll_area.setWidget(self.accounts_container)
        main_layout.addWidget(self.scroll_area)







    def sync_shiny_dust_from_ui(self):
        """Sincronizza shiny_dust in thread separato."""
        try:
            worker = SyncWorkerAsync()
            worker.signals.finished.connect(self._on_sync_finished)
            worker.signals.error.connect(self._on_sync_error)
            self.thread_pool.start(worker)
        except Exception as e:
            self._safe_log(f"❌ Errore lancio sincronizzazione: {e}")


    def _create_global_header(self) -> QFrame:
        """Crea header globale minimale."""
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background: transparent;
                border: none;
                padding: 5px;
            }
        """)
        
        layout = QHBoxLayout(header)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(15)
        
        # Titolo (senza icona, font ridotto)
        self.title_label = QLabel("Gestione Account")
        self.title_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.title_label.setStyleSheet("color: #f39c12;")
        layout.addWidget(self.title_label)
        
        layout.addStretch()
        
        # Totale accounts
        self.total_accounts_label = QLabel("Totale Account: 0")
        self.total_accounts_label.setFont(QFont("Segoe UI", 9))
        self.total_accounts_label.setFixedHeight(32)
        self.total_accounts_label.setStyleSheet("""
            QLabel {
                background-color: rgba(243, 156, 18, 0.2);
                border: 1px solid #f39c12;
                border-radius: 6px;
                padding: 0px 12px;
                color: #f39c12;
            }
        """)
        layout.addWidget(self.total_accounts_label)
        
        # Pulsante refresh (giallo con icona centrata)
        self.refresh_btn = QPushButton()
        self.refresh_btn.setFixedSize(36, 36)
        refresh_icon = self.style().standardIcon(QStyle.SP_BrowserReload)
        self.refresh_btn.setIcon(refresh_icon)
        self.refresh_btn.setIconSize(QSize(18, 18))
        self.refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(243, 156, 18, 0.3);
                border: 2px solid #f39c12;
                border-radius: 18px;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: rgba(243, 156, 18, 0.5);
                border: 2px solid #f1c40f;
            }
            QPushButton:pressed {
                background-color: rgba(243, 156, 18, 0.7);
            }
            QPushButton:disabled {
                background-color: rgba(149, 165, 166, 0.3);
                border: 2px solid #95a5a6;
            }
        """)
        self.refresh_btn.setToolTip("Ricarica account")
        self.refresh_btn.clicked.connect(self.load_accounts)
        self.refresh_btn.setCursor(Qt.PointingHandCursor)
        layout.addWidget(self.refresh_btn)

        return header
    



    
    def load_accounts(self):
        """Carica gli account dal DB."""
        self.refresh_btn.setEnabled(False)
        self.refresh_btn.setText("Caricamento...")
        
        # Lancia il worker per caricare gli account dal DB
        worker = AccountsLoaderWorker()
        worker.signals.accounts_ready.connect(self.on_accounts_loaded)
        worker.signals.error.connect(self.on_error)
        self.thread_pool.start(worker)
        sync_shiny_dust()
        



    def _on_sync_finished(self, result):
        """Callback quando sincronizzazione è finita."""
        try:
            updated = result.get('updated', 0)
            total = result.get('total', 0)
            not_found = result.get('not_found', 0)
            
            msg = f"✅ Sincronizzazione: {updated}/{total} aggiornati"
            if not_found > 0:
                msg += f" ({not_found} non trovati nel DB)"
            
            self._safe_log(msg)
        except Exception as e:
            print(f"❌ Errore callback: {e}")
        finally:
            if hasattr(self, 'refresh_btn'):
                self.refresh_btn.setText("")
                self.refresh_btn.setEnabled(True)






    def _safe_log(self, message):
        """Log sicuro che non crasha."""
        try:
            if hasattr(self, 'log_callback') and callable(self.log_callback):
                self.log_callback(message)
            else:
                print(message)
        except Exception as e:
            print(f"Log error: {e}")


    def _on_sync_error(self, error):
        """Callback per errori sincronizzazione - versione sicura."""
        try:
            if hasattr(self, 'log_callback') and callable(self.log_callback):
                self.log_callback(f"❌ Errore sincronizzazione: {error}")
            else:
                print(f"❌ Errore: {error}")
        except Exception as e:
            print(f"❌ Errore callback: {e}")
        finally:
            # Nascondi spinner
            self._show_spinner(False)
            
            if hasattr(self, 'refresh_btn'):
                self.refresh_btn.setText("")
                self.refresh_btn.setEnabled(True)
            if hasattr(self, 'load_accounts'):
                try:
                    self.load_accounts()
                except:
                    pass



    def _load_more_accounts(self):
        """Carica i prossimi N account."""
        if not self.all_accounts:
            self._add_empty_state()
            return
        
        # Determina range da caricare
        start_idx = self.displayed_accounts_count
        end_idx = min(start_idx + self.accounts_per_page, len(self.all_accounts))
        
        if start_idx >= len(self.all_accounts):
            return  # Nessun altro account da caricare
        
        # Rimuovi il pulsante "Load More" se esiste
        if hasattr(self, 'load_more_button') and self.load_more_button:
            self.accounts_layout.removeWidget(self.load_more_button)
            self.load_more_button.deleteLater()
            self.load_more_button = None
        
        # Rimuovi stretch temporaneamente
        stretch_item = self.accounts_layout.takeAt(self.accounts_layout.count() - 1)
        
        # Aggiungi nuovi account
        accounts_to_add = self.all_accounts[start_idx:end_idx]
        
        for i, account in enumerate(accounts_to_add):
            widget = CollapsibleAccountWidget(self.main_window, account, self.total_sets)
            widget.expand_signal.connect(self._update_layout)
            self.accounts_layout.addWidget(widget)
            self.accounts_widgets.append(widget)
            
            # Aggiungi separatore se non è l'ultimo
            if start_idx + i < len(self.all_accounts) - 1:
                self.accounts_layout.addWidget(self._create_separator())
        
        self.displayed_accounts_count = end_idx
        
        # Aggiungi pulsante "Load More" se ci sono altri account
        if end_idx < len(self.all_accounts):
            self._add_load_more_button()
        
        # Rimetti stretch
        self.accounts_layout.addItem(stretch_item)

    def _add_load_more_button(self):
        """Aggiunge pulsante per caricare altri account."""
        load_more_container = QWidget()
        load_more_layout = QHBoxLayout(load_more_container)
        load_more_layout.setContentsMargins(0, 20, 0, 20)
        
        self.load_more_button = QPushButton(
            f"⬇ Carica altri {min(self.accounts_per_page, len(self.all_accounts) - self.displayed_accounts_count)} account"
        )
        self.load_more_button.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.load_more_button.setCursor(Qt.PointingHandCursor)
        self.load_more_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f39c12, stop:1 #e67e22);
                color: black;
                border: none;
                border-radius: 10px;
                padding: 15px 30px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f1c40f, stop:1 #f39c12);
            }
            QPushButton:pressed {
                background: #d35400;
            }
        """)
        self.load_more_button.clicked.connect(self._load_more_accounts)
        
        load_more_layout.addStretch()
        load_more_layout.addWidget(self.load_more_button)
        load_more_layout.addStretch()
        
        # Inserisci prima dello stretch finale
        insert_pos = self.accounts_layout.count() - 1
        self.accounts_layout.insertWidget(insert_pos, load_more_container)

    def _clear_accounts_layout(self):
        """Pulisce completamente il layout degli account."""
        # Rimuovi stretch
        stretch = self.accounts_layout.takeAt(self.accounts_layout.count() - 1)
        
        # Rimuovi tutti i widget
        while self.accounts_layout.count() > 0:
            item = self.accounts_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Resetta variabili
        self.accounts_widgets = []
        self.load_more_button = None
        
        # Rimetti stretch
        self.accounts_layout.addItem(stretch)

    def _add_empty_state(self):
        """Aggiunge lo stato vuoto."""
        self.accounts_layout.insertWidget(
            self.accounts_layout.count() - 1,
            self._create_empty_state()
        )



    @pyqtSlot(list, int)
    def on_accounts_loaded(self, accounts, total_sets):
        """Callback quando gli account sono caricati dal DB."""
        self.refresh_btn.setEnabled(True)
        self.refresh_btn.setText("")
        
        # Aggiorna il totale account
        try:
            with sqlite3.connect(DB_FILENAME) as db:
                cursor = db.cursor()
                cursor.execute("SELECT COUNT(DISTINCT device_account) as total FROM accounts")
                total_unique = cursor.fetchone()[0]
                self.total_accounts_label.setText(f"Totale Account: {total_unique}")
        except Exception as e:
            self.total_accounts_label.setText(f"Totale Account: {len(accounts)}")
        
        # Visualizza gli account nella GUI
        self._display_accounts(accounts, total_sets)



    def _display_accounts(self, accounts: List[Dict], total_sets: int):
        """Visualizza la lista di account con lazy loading."""
        self.all_accounts = accounts
        self.total_sets = total_sets
        self.displayed_accounts_count = 0
        
        self._clear_accounts_layout()
        self._load_more_accounts()

    
    # accounts_tab.py (Dentro la classe AccountsTab)

    def _create_separator(self) -> QFrame:
        """Crea separatore."""
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setMaximumHeight(2)
        line.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 transparent, stop:0.5 #f39c12, stop:1 transparent);
                /* ✅ CORREZIONE: Aggiungi un margine verticale uniforme */
                margin: 10px 30px 10px 30px; 
                border: none;
            }
        """)
        return line
    
    def _create_empty_state(self) -> QWidget:
        """Crea widget stato vuoto."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(20)
        
        icon = QLabel("📭")
        icon.setFont(QFont("Segoe UI", 72))
        icon.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon)
        
        title = QLabel("Nessun Account Trovato")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title.setStyleSheet(f"color: {Colors.DANGER};")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        message = QLabel("Aggiungi un account per iniziare.")
        message.setFont(QFont("Segoe UI", 12))
        message.setStyleSheet(f"color: {Colors.MUTED};")
        message.setAlignment(Qt.AlignCenter)
        layout.addWidget(message)
        
        widget.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2d2d2d, stop:1 #1a1a1a);
                border: 2px dashed #555;
                border-radius: 15px;
                padding: 60px;
            }
        """)
        
        return widget
    
    @pyqtSlot(str)
    
    def on_error(self, error_msg):
        """Callback per errori durante caricamento."""
        self._safe_log(f"❌ Errore: {error_msg}")
        self.refresh_btn.setEnabled(True)
        self.refresh_btn.setText("")

    
    @pyqtSlot()
    def _update_layout(self):
        """Aggiorna layout."""
        self.accounts_container.adjustSize()
        self.scroll_area.update()


    def _on_search_text_changed(self, text: str):
        """Gestisce il cambio di testo nella ricerca."""
        text = text.strip()
        
        if len(text) == 0:
            # Ricarica tutti gli account
            self.load_accounts()
            return
        
        if len(text) < 3:
            # Non fare nulla se meno di 3 caratteri
            return
        
        # Avvia ricerca con debouncing
        if hasattr(self, '_search_timer'):
            self._search_timer.stop()
        
        self._search_timer = QTimer()
        self._search_timer.setSingleShot(True)
        self._search_timer.timeout.connect(lambda: self._perform_search(text))
        self._search_timer.start(300)  # 300ms di debounce

    def _perform_search(self, search_term: str):
        """Esegue la ricerca in background."""
        self.title_label.setText("Gestione Account (Ricerca...)")
        
        worker = SearchAccountsWorker(search_term)
        worker.signals.accounts_ready.connect(self._on_search_results)
        worker.signals.error.connect(self.on_error)
        self.main_window.image_loader_pool.start(worker)

    @pyqtSlot(list, int)
    def _on_search_results(self, accounts: List[Dict], total_sets: int):
        """Callback risultati ricerca."""
        self.title_label.setText(f"Gestione Account (Trovati: {len(accounts)})")
        self._display_accounts(accounts, total_sets)

    def _clear_search(self):
        """Pulisce la ricerca e ricarica tutti gli account."""
        self.search_input.clear()
        
        # Deseleziona tutti i filtri
        for btn in self.sort_buttons:
            btn.setChecked(False)
        
        self.current_sort = None
        self.current_order = None
        
        self.load_accounts()

    def _apply_sort(self, sort_type: str, order: str):
        """Applica ordinamento agli account."""
        # Deseleziona altri pulsanti
        sender = self.sender()
        for btn in self.sort_buttons:
            if btn != sender:
                btn.setChecked(False)
        
        # Se stesso pulsante cliccato di nuovo, resetta
        if self.current_sort == sort_type and self.current_order == order:
            sender.setChecked(False)
            self.current_sort = None
            self.current_order = None
            self.load_accounts()
            return
        
        self.current_sort = sort_type
        self.current_order = order
        
        # Ricarica con ordinamento
        self._load_accounts_sorted()

    def _load_accounts_sorted(self):
        """Carica account con ordinamento applicato."""
        self.refresh_btn.setEnabled(False)
        
        worker = SortedAccountsLoaderWorker(self.current_sort, self.current_order)
        worker.signals.accounts_ready.connect(self.on_accounts_loaded)
        worker.signals.error.connect(self.on_error)
        self.main_window.image_loader_pool.start(worker)



    def _create_search_bar(self) -> QWidget:
        """Crea barra di ricerca minimale."""
        search_container = QFrame()
        search_container.setStyleSheet("""
            QFrame {
                background: transparent;
                border: none;
                padding: 5px;
            }
        """)
        
        layout = QHBoxLayout(search_container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(15)
        
        # === CAMPO RICERCA (senza icone, senza bordo colorato) ===
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Cerca account per nome o alias (min. 3 caratteri)...")
        self.search_input.setFont(QFont("Segoe UI", 10))
        self.search_input.setMinimumWidth(300)
        self.search_input.setFixedHeight(32)
        self.search_input.setStyleSheet("""
            QLineEdit {
                background-color: rgba(52, 73, 94, 0.3);
                border: 1px solid #3a3a3a;
                border-radius: 6px;
                padding: 0px 12px;
                color: #ecf0f1;
                font-size: 10px;
            }
            QLineEdit:focus {
                border: 1px solid #555;
                background-color: rgba(52, 73, 94, 0.5);
            }
            QLineEdit::placeholder {
                color: #7f8c8d;
                font-style: italic;
            }
        """)
        self.search_input.textChanged.connect(self._on_search_text_changed)
        layout.addWidget(self.search_input, 1)
        
        # === SEPARATORE ===
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #3a3a3a;")
        separator.setFixedWidth(1)
        layout.addWidget(separator)
        
        # === LABEL ORDINAMENTO ===
        sort_label = QLabel("Ordina:")
        sort_label.setFont(QFont("Segoe UI", 9))
        sort_label.setStyleSheet("color: #bdc3c7;")
        layout.addWidget(sort_label)
        
        # === FILTRO SHINY DUST (TOGGLE) ===
        self.sort_dust_btn = QPushButton()
        self.sort_dust_btn.setToolTip("Ordina per Shiny Dust (clicca per alternare)")
        self.sort_dust_btn.setCursor(Qt.PointingHandCursor)
        self.sort_dust_btn.clicked.connect(lambda: self._toggle_sort('dust'))
        self._create_sort_button_with_icon(self.sort_dust_btn, SHINY_DUST_ICON_PATH)
        layout.addWidget(self.sort_dust_btn)
        
        # === FILTRO HOURGLASSES (TOGGLE) ===
        self.sort_hourglass_btn = QPushButton()
        self.sort_hourglass_btn.setToolTip("Ordina per Clessidre (clicca per alternare)")
        self.sort_hourglass_btn.setCursor(Qt.PointingHandCursor)
        self.sort_hourglass_btn.clicked.connect(lambda: self._toggle_sort('hourglass'))
        self._create_sort_button_with_icon(self.sort_hourglass_btn, HOURGLASS_ICON_PATH)
        layout.addWidget(self.sort_hourglass_btn)
        
        # Variabili per tenere traccia del sort attivo
        self.current_sort = None
        self.current_order = None
        self.sort_buttons = {
            'dust': self.sort_dust_btn,
            'hourglass': self.sort_hourglass_btn
        }
        
        # Aggiorna label iniziali
        self._update_sort_button_label(self.sort_dust_btn, None)
        self._update_sort_button_label(self.sort_hourglass_btn, None)
        
        return search_container


    def _create_sort_button_with_icon(self, button: QPushButton, icon_path: str):
        """Crea pulsante di ordinamento con icona."""
        button.setFixedSize(80, 40)
        button.setFont(QFont("Segoe UI", 9, QFont.Bold))
        
        # Carica icona
        if os.path.exists(icon_path):
            icon = QIcon(icon_path)
            button.setIcon(icon)
            button.setIconSize(QSize(20, 20))
        
        button.setStyleSheet("""
            QPushButton {
                background-color: rgba(243, 156, 18, 0.2);
                border: 2px solid #f39c12;
                border-radius: 8px;
                color: #f39c12;
                font-weight: bold;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: rgba(243, 156, 18, 0.4);
                border-color: #f1c40f;
            }
            QPushButton:pressed {
                background-color: rgba(243, 156, 18, 0.6);
            }
        """)

    def _update_sort_button_label(self, button: QPushButton, order: Optional[str]):
        """Aggiorna label del pulsante in base all'ordine."""
        if order == 'asc':
            button.setText("↑")
            button.setStyleSheet("""
                QPushButton {
                    background-color: rgba(243, 156, 18, 0.6);
                    border: 2px solid #f39c12;
                    border-radius: 8px;
                    color: #ecf0f1;
                    font-weight: bold;
                    font-size: 16px;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: rgba(243, 156, 18, 0.7);
                }
            """)
        elif order == 'desc':
            button.setText("↓")
            button.setStyleSheet("""
                QPushButton {
                    background-color: rgba(243, 156, 18, 0.6);
                    border: 2px solid #f39c12;
                    border-radius: 8px;
                    color: #ecf0f1;
                    font-weight: bold;
                    font-size: 16px;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: rgba(243, 156, 18, 0.7);
                }
            """)
        else:
            button.setText("—")
            button.setStyleSheet("""
                QPushButton {
                    background-color: rgba(243, 156, 18, 0.2);
                    border: 2px solid #f39c12;
                    border-radius: 8px;
                    color: #f39c12;
                    font-weight: bold;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: rgba(243, 156, 18, 0.4);
                    border-color: #f1c40f;
                }
            """)

    def _toggle_sort(self, sort_type: str):
        """Alterna l'ordinamento per un tipo (none -> asc -> desc -> none)."""
        # Se è un tipo diverso, resetta l'altro
        other_type = 'hourglass' if sort_type == 'dust' else 'dust'
        if self.current_sort == other_type:
            self._update_sort_button_label(self.sort_buttons[other_type], None)
        
        # Determina nuovo ordine
        if self.current_sort != sort_type or self.current_order is None:
            new_order = 'asc'
        elif self.current_order == 'asc':
            new_order = 'desc'
        else:
            new_order = None
        
        # Aggiorna stato
        if new_order is None:
            self.current_sort = None
            self.current_order = None
            self._update_sort_button_label(self.sort_buttons[sort_type], None)
            self.load_accounts()
        else:
            self.current_sort = sort_type
            self.current_order = new_order
            self._update_sort_button_label(self.sort_buttons[sort_type], new_order)
            self._load_accounts_sorted()

    def _clear_search(self):
        """Pulisce la ricerca e ricarica tutti gli account."""
        self.search_input.clear()
        
        # Deseleziona tutti i filtri
        for sort_type, btn in self.sort_buttons.items():
            self._update_sort_button_label(btn, None)
        
        self.current_sort = None
        self.current_order = None
        
        self.load_accounts()



# ============================================================================
# SEARCH WORKER
# ============================================================================
class SearchAccountsWorker(QRunnable):
    """Worker per ricerca account in tempo reale."""
    
    def __init__(self, search_term: str):
        super().__init__()
        self.search_term = search_term
        self.signals = WorkerSignals()
    
    @pyqtSlot()
    def run(self):
        try:
            if len(self.search_term) < 3:
                self.signals.accounts_ready.emit([], 0)
                return
            
            with sqlite3.connect(DB_FILENAME) as db:
                db.row_factory = sqlite3.Row
                cursor = db.cursor()
                
                # Ricerca per nome o alias
                search_pattern = f"%{self.search_term}%"
                cursor.execute("""
                    SELECT device_account AS account_id, account_name AS account_name, alias, shiny_dust AS shiny_dust, hourglasses 
                    FROM accounts WHERE account_name LIKE ? OR alias LIKE ? 
                    ORDER BY device_account ASC
                """, (search_pattern, search_pattern))
                
                accounts = [dict(row) for row in cursor.fetchall()]
                
                cursor.execute("SELECT COUNT(*) FROM sets")
                total_sets = cursor.fetchone()[0]
                
                self.signals.accounts_ready.emit(accounts, total_sets)
        except Exception as e:
            self.signals.error.emit(f"Errore ricerca: {e}")


class SortedAccountsLoaderWorker(QRunnable):
    """Worker per caricamento account ordinati."""
    
    def __init__(self, sort_type: str, order: str):
        super().__init__()
        self.sort_type = sort_type
        self.order = order
        self.signals = WorkerSignals()
    
    @pyqtSlot()
    def run(self):
        try:
            with sqlite3.connect(DB_FILENAME) as db:
                db.row_factory = sqlite3.Row
                cursor = db.cursor()
                
                # Determina campo di ordinamento
                if self.sort_type == 'dust':
                    order_field = 'shiny_dust'
                elif self.sort_type == 'hourglass':
                    order_field = 'hourglasses'
                else:
                    order_field = 'account_id'
                
                order_dir = 'ASC' if self.order == 'asc' else 'DESC'
                
                query = f"""
                    SELECT device_account AS account_id, account_name AS account_name, alias, shiny_dust AS shiny_dust, hourglasses 
                    FROM accounts ORDER BY {order_field} {order_dir}
                """
                
                cursor.execute(query)
                accounts = [dict(row) for row in cursor.fetchall()]
                
                cursor.execute("SELECT COUNT(*) FROM sets")
                total_sets = cursor.fetchone()[0]
                
                self.signals.accounts_ready.emit(accounts, total_sets)
        except Exception as e:
            self.signals.error.emit(f"Errore ordinamento: {e}")


# ================================================================
# CLASSE: SyncSignals
# ================================================================

class SyncSignals(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)


# ================================================================
# CLASSE: SyncWorker
# ================================================================

class SyncWorker(QRunnable):
    """Worker per sincronizzazione shiny_dust in background."""
    
    def __init__(self):
        super().__init__()
        self.signals = SyncSignals()
    
    def run(self):
        try:
            from .sync_shiny_dust import sync_shiny_dust
            result = sync_shiny_dust()
            account_reloaded = self.load_accounts()

            self.signals.finished.emit(account_reloaded)
        except Exception as e:
            self.signals.error.emit(str(e))



class SyncWorkerAsync(QRunnable):
    """Worker per sincronizzazione shiny_dust."""
    
    def __init__(self):
        super().__init__()
        self.signals = SyncSignalsSimple()
    
    def run(self):
        try:
            from .sync_shiny_dust import sync_shiny_dust
            result = sync_shiny_dust()
            self.signals.finished.emit(result)
        except Exception as e:
            self.signals.error.emit(str(e))


class SyncSignalsSimple(QObject):
    """Segnali semplificate."""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
