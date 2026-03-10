"""ui_collection_loader.py - Loader ottimizzato per collection"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QLabel, QFrame
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import sqlite3
from config import DB_FILENAME


class LazyCollectionLoader(QScrollArea):
    """Carica i card solo quando diventano visibili"""
    
    def __init__(self, db_path, parent=None):
        super().__init__(parent)
        self.db_path = db_path
        self.visible_range = {}
        self.loaded_sets = set()
        self.all_sets = []
        
        # Connetti al segnale di scroll
        self.verticalScrollBar().valueChanged.connect(self._on_scroll)
        
    def _on_scroll(self):
        """Carica i set quando scrolli"""
        view_rect = self.viewport().rect()
        # Implementa logica per caricare solo i set nel viewport
        pass


class CollectionLoaderThread(QThread):
    """Carica la collection in background"""
    
    set_ready = pyqtSignal(str, str, int, str, str, dict)  # set_code, name, total, cover, account, inventory
    progress = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, db_path, account_name):
        super().__init__()
        self.db_path = db_path
        self.account_name = account_name
        self._is_running = True
        
    def run(self):
        """Carica i set uno per uno"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Ottieni lista dei set
                cursor.execute("SELECT DISTINCT set_code, set_name FROM sets ORDER BY release_date DESC")
                sets = cursor.fetchall()
                
                self.progress.emit(f"Loading {len(sets)} sets...")
                
                for i, set_row in enumerate(sets):
                    if not self._is_running:
                        break
                    
                    set_code = set_row['set_code']
                    set_name = set_row['set_name']
                    
                    # Carica i dati del set
                    cursor.execute("""
                        SELECT COUNT(*) as total FROM cards WHERE set_code = ?
                    """, (set_code,))
                    total = cursor.fetchone()['total']
                    
                    # Carica inventory
                    inventory = self._load_inventory(cursor, set_code)
                    
                    self.progress.emit(f"Loading {set_name}... ({i+1}/{len(sets)})")
                    
                    # Emetti il signal - il widget verrà creato nel thread main
                    self.set_ready.emit(set_code, set_name, total, "", self.account_name, inventory)
                
                self.progress.emit("✅ Collection loaded!")
                self.finished.emit()
                
        except Exception as e:
            self.error.emit(f"Error loading collection: {str(e)}")
    
    def _load_inventory(self, cursor, set_code):
        """Carica l'inventory per un set"""
        cursor.execute("""
            SELECT card_id, quantity FROM account_inventory 
            WHERE account_id = (SELECT id FROM accounts WHERE account_name = ?)
            AND card_id IN (SELECT id FROM cards WHERE set_code = ?)
        """, (self.account_name, set_code))
        return {row[0]: row[1] for row in cursor.fetchall()}
    
    def stop(self):
        """Ferma il caricamento"""
        self._is_running = False
