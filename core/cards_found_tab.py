# core/cards_found_tab.py
"""
Questo modulo contiene il QWidget per la scheda "Cards Found".
"""

# Import PyQt5
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QLabel, QPushButton, QHeaderView, QFileDialog, QMessageBox, QWidget
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QRunnable, pyqtSlot, QThreadPool
from PyQt5.QtGui import QPixmap, QColor, QFont

# Import standard
import os
import sqlite3
import json
import csv
import base64
import urllib.request
from typing import TYPE_CHECKING

# Import moduli app
from config import DB_FILENAME, RARITY_DATA, get_app_data_path, get_resource_path
from .translations import t

# ================================================================
# ✅ WORKER DI DOWNLOAD (SOLO PER IMMAGINI REMOTE NON BLOB)
# ================================================================

class ImageLoaderSignals(QObject):
    finished = pyqtSignal(bytes, str, QLabel)
    error = pyqtSignal(str, str, QLabel)

class ImageDownloaderWorker(QRunnable):
    def __init__(self, image_url: str, target_label: QLabel):
        super().__init__()
        self.image_url = image_url
        self.target_label = target_label
        self.signals = ImageLoaderSignals()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    @pyqtSlot()
    def run(self):
        if not self.image_url or not self.image_url.startswith('http'):
            err_msg = "URL non valido"
            # print(f"❌ Worker Errore Download: {err_msg} | URL: {self.image_url}")
            self.signals.error.emit(err_msg, self.image_url, self.target_label)
            return
        try:
            # print(f"🔄 Inizio download: {self.image_url}")
            req = urllib.request.Request(self.image_url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                image_data = response.read()
            if image_data:
                # print(f"✅ Download completato ({len(image_data)} bytes): {self.image_url}")
                self.signals.finished.emit(image_data, self.image_url, self.target_label)
            else:
                err_msg = "Dati immagine vuoti"
                # print(f"❌ Worker Errore Download: {err_msg} | URL: {self.image_url}")
                self.signals.error.emit(err_msg, self.image_url, self.target_label)
        except Exception as e:
            # print(f"❌ Worker Errore Download (Exception): {e} | URL: {self.image_url}")
            self.signals.error.emit(str(e), self.image_url, self.target_label)

# ❌ RIMOSSO: SetCoverPreloaderWorker (Non più necessario con i BLOB)

# ================================================================
# FINE WORKERS
# ================================================================

# Type checking
if TYPE_CHECKING:
    from .ui_main_window import MainWindow

class CardsFoundTab(QWidget):
    
    def __init__(self, main_window: 'MainWindow', parent=None):
        super().__init__(parent)
        
        self.main_window = main_window
        self.image_cache = main_window.image_cache
        self.image_loader_pool = main_window.image_loader_pool
        self.placeholder_pixmap = main_window.placeholder_pixmap
        
        self.cards_offset = 0
        self.found_cards_list = [] 
        
        # ❌ RIMOSSO: self.preloader_pool (Non più necessario)
        
        self.setup_ui()
        self.load_found_cards_from_database()

    def setup_ui(self):
        """Configura l'interfaccia utente di questa scheda."""
        cards_layout = QVBoxLayout(self)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # Pulsante Refresh Cache
        self.refresh_btn = QPushButton(t("cards_found.refresh_list"))
        self.refresh_btn.setToolTip(t("cards_found.refresh_list_tooltip"))
        self.refresh_btn.clicked.connect(self.refresh_list_and_cache)
        controls_layout.addWidget(self.refresh_btn)
        
        self.clear_cards_btn = QPushButton(t("ui.clear_list"))
        self.clear_cards_btn.clicked.connect(self.clear_cards_list)
        controls_layout.addWidget(self.clear_cards_btn)
        
        self.export_cards_btn = QPushButton(t("ui.export_csv"))
        self.export_cards_btn.clicked.connect(self.export_cards_to_csv)
        controls_layout.addWidget(self.export_cards_btn)
        
        controls_layout.addStretch()
        
        self.cards_count_label = QLabel(t("ui.total_cards_found", count=0))
        self.cards_count_label.setStyleSheet("QLabel { font-size: 12px; font-weight: bold; }")
        controls_layout.addWidget(self.cards_count_label)
        
        cards_layout.addLayout(controls_layout)
        
        # Cards Table
        self.cards_table = QTableWidget()
        self.cards_table.setColumnCount(7)
        self.cards_table.setHorizontalHeaderLabels([
            t("ui.table.card"), t("ui.table.pack"),
            t("ui.table.set_cover"),
            t("ui.table.card_number"), t("ui.table.card_name"),
            t("ui.table.rarity"), "Similarity"
        ])
        self.cards_table.horizontalHeader().setStretchLastSection(False)
        self.cards_table.setAlternatingRowColors(True)
        self.cards_table.setSortingEnabled(True)

        # ✅ COLONNE RIDIMENSIONATE E CENTRATE
        self.cards_table.setColumnWidth(0, 90)    # Card preview
        self.cards_table.setColumnWidth(1, 90)    # Pack preview
        self.cards_table.setColumnWidth(2, 120)   # Account
        self.cards_table.setColumnWidth(3, 90)    # Set cover
        self.cards_table.setColumnWidth(4, 100)   # Card number
        self.cards_table.setColumnWidth(5, 200)   # Card name
        self.cards_table.setColumnWidth(6, 100)   # Rarity
        self.cards_table.setColumnWidth(7, 110)   # Similarity
        
        self.cards_table.verticalHeader().setDefaultSectionSize(75)
        
        self.cards_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.cards_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.cards_table.setStyleSheet("""
            QTableWidget::item:selected {
                background-color: #f39c12; color: #000000;
            }
            QTableWidget {
                gridline-color: #cccccc;
            }
        """)
        cards_layout.addWidget(self.cards_table)
        
        # Load More Button
        self.load_more_btn = QPushButton("📥 " + t("ui.load_more", count=20))
        self.load_more_btn.clicked.connect(self.on_load_more_cards)
        load_more_layout = QHBoxLayout()
        load_more_layout.addStretch()   
        self.load_more_btn.setMaximumWidth(200)
        load_more_layout.addWidget(self.load_more_btn)
        
        cards_layout.addLayout(load_more_layout)
    
    # --- Metodo Pubblico per MainWindow ---
    
    # core/cards_found_tab.py

    def add_new_card(self, card_data: dict):
        """
        Metodo pubblico chiamato da MainWindow (on_card_found) 
        per aggiungere una carta in cima alla lista.
        
        ✅ MODIFICATO: Recupera il BLOB della cover dal DB se non è presente nei dati ricevuti.
        """
        # 1. Verifica se manca il blob della cover del set
        if not card_data.get('set_cover_blob'):
            set_code = card_data.get('set_code')
            if set_code:
                try:
                    # Recupero rapido del BLOB dal DB
                    with sqlite3.connect(DB_FILENAME) as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT cover_image_blob FROM sets WHERE set_code = ?", (set_code,))
                        row = cursor.fetchone()
                        if row and row[0]:
                            card_data['set_cover_blob'] = row[0] # Aggiungi il blob ai dati
                            # print(f"✅ Cover BLOB recuperata on-the-fly per {set_code}")
                except Exception as e:
                    print(f"⚠️ Impossibile recuperare cover blob per {set_code}: {e}")

        # 2. Verifica se manca il blob dello screenshot (pack) ma c'è l'URL/Path
        # (Opzionale: se lo scraper passa solo il path, proviamo a caricarlo)
        if not card_data.get('screenshot_thumbnail_blob'):
            # Se c'è un path locale per lo screenshot, prova a caricarlo come bytes
            screenshot_path = card_data.get('image_url_screenshot') # A volte usato per il path locale
            if screenshot_path and os.path.exists(screenshot_path):
                try:
                    with open(screenshot_path, 'rb') as f:
                        card_data['screenshot_thumbnail_blob'] = f.read()
                except Exception:
                    pass

        # 3. Aggiungi alla tabella
        self.add_card_to_table(card_data, insert_at_top=True)
        
        # 4. Aggiorna conteggio
        self.cards_count_label.setText(t("ui.total_cards_found", count=self.cards_table.rowCount()))
        
        # 5. Mantieni la lista pulita (max 20)
        while self.cards_table.rowCount() > 20:
            self.cards_table.removeRow(self.cards_table.rowCount() - 1)
            
    # --- Logica Interna ---

    def refresh_list_and_cache(self):
        """
        Svuota la cache delle immagini e ricarica la lista.
        """
        try:
            self.image_cache.clear()
            self.main_window.append_bot_log("Cache immagini svuotata...")
            self.load_found_cards_from_database()
            self.load_more_btn.setEnabled(True)
            self.load_more_btn.setText("📥 " + t("ui.load_more", count=20))
            self.main_window.append_bot_log("Lista 'Cards Found' ricaricata.")
        except Exception as e:
            self.main_window.append_bot_log(f"❌ Errore durante il refresh: {e}")

    def on_load_more_cards(self):
        """Callback per caricamento altre cards."""
        if self.load_more_found_cards():
            self.load_more_btn.setText("📥 " + t("ui.load_more", count=20))
        else:
            self.load_more_btn.setEnabled(False)
            self.load_more_btn.setText("✓ " + t("ui.all_cards_loaded"))
            
    def _get_rarity_filter(self) -> list:
        """Carica le rarità selezionate da settings.json."""
        try:
            settings_path = get_app_data_path("settings.json")
        except:
            settings_path = "settings.json"   
        
        saved_rarities = list(RARITY_DATA.keys()) 
        if os.path.exists(settings_path):
            try:
                with open(settings_path, 'r', encoding="utf-8") as f:
                    settings = json.load(f)       
                saved_rarities = settings.get('selected_rarities', list(RARITY_DATA.keys()))
            except Exception:
                pass 
        return saved_rarities

    # ❌ RIMOSSO: Intero blocco di metodi per il precaricamento cover

    def load_found_cards_from_database(self):
        """Carica le prime 20 carte trovate dal database."""
        try:
            self.cards_table.setRowCount(0)
            self.cards_offset = 20 
            
            saved_rarities = self._get_rarity_filter()
            rarity_placeholders = ', '.join('?' for _ in saved_rarities)

            with sqlite3.connect(DB_FILENAME) as conn:
                cursor = conn.cursor()
                
                sql_query = f"""
                    SELECT 
                        c.card_name, c.rarity,
                        COALESCE(a.account_name, 'Unknown') as account_name,
                        c.set_code, c.card_number,
                        c.thumbnail_blob,
                        t.image_url,
                        c.local_image_path,
                        t.screenshot_thumbnail_blob,
                        fc.confidence_score,
                        s.cover_image_blob
                    FROM found_cards fc
                    JOIN cards c ON fc.card_id = c.id
                    JOIN sets s ON c.set_code = s.set_code
                    -- ✅ CORREZIONE CRUCIALE: JOIN su a.device_account (la nuova PK)
                    LEFT JOIN accounts a ON fc.account_id = a.device_account 
                    LEFT JOIN trades t ON fc.message_id = t.message_id
                    WHERE c.rarity IN ({rarity_placeholders})
                    ORDER BY fc.found_at DESC
                    LIMIT 20
                """
                
                cursor.execute(sql_query, saved_rarities)
                cards = cursor.fetchall()

                cursor.execute(f"""
                    SELECT COUNT(fc.id) FROM found_cards fc
                    JOIN cards c ON fc.card_id = c.id
                    WHERE c.rarity IN ({rarity_placeholders})
                """, saved_rarities)
                total_count = cursor.fetchone()[0]
            
            self.cards_count_label.setText(f"{t('ui.cards_found')}: {total_count} ({t('ui.showing')} {len(cards)})") 
            
            # ✅ Converti i dati (il resto è invariato e corretto)
            cards_data = []
            for card_data in cards:
                card_dict = {
                    'card_name': card_data[0],
                    'rarity': card_data[1],
                    'account_name': card_data[2],
                    'set_code': card_data[3],
                    'card_number': str(card_data[4]),
                    'thumbnail_blob': card_data[5],
                    'image_url_screenshot': card_data[6],
                    'image_url': card_data[7],
                    'screenshot_thumbnail_blob': card_data[8],
                    'similarity': (card_data[9]) if card_data[9] else 0,
                    'set_cover_blob': card_data[10]
                }
                cards_data.append(card_dict)
            
            for card_dict in cards_data:
                self.add_card_to_table(card_dict, insert_at_top=False)
                    
        except Exception as e:
            print(f"❌ Errore load_found_cards: {e}")
            import traceback
            traceback.print_exc()

    def load_more_found_cards(self):
        """Carica più carte trovate (paginazione)."""
        try:
            saved_rarities = self._get_rarity_filter()
            rarity_placeholders = ', '.join('?' for _ in saved_rarities)

            with sqlite3.connect(DB_FILENAME) as conn:
                cursor = conn.cursor()
                
                # ✅ MODIFICATO: Seleziona s.cover_image_blob
                sql_query = f"""
                    SELECT 
                        c.card_name, c.rarity,
                        COALESCE(a.account_name, 'Unknown') as account_name,
                        c.set_code, c.card_number,
                        c.thumbnail_blob,
                        t.image_url,
                        c.local_image_path,
                        t.screenshot_thumbnail_blob,
                        fc.confidence_score,
                        s.cover_image_blob -- ✅ MODIFICATO: BLOB
                    FROM found_cards fc
                    JOIN cards c ON fc.card_id = c.id
                    JOIN sets s ON c.set_code = s.set_code
                    -- ✅ CORREZIONE CRUCIALE: JOIN su a.device_account (la nuova PK)
                    LEFT JOIN accounts a ON fc.account_id = a.device_account
                    LEFT JOIN trades t ON fc.message_id = t.message_id
                    WHERE c.rarity IN ({rarity_placeholders})
                    ORDER BY fc.found_at DESC
                    LIMIT 20 OFFSET ?
                """
                
                params = saved_rarities + [self.cards_offset]
                cursor.execute(sql_query, params)
                cards = cursor.fetchall()
                
            if not cards:
                self.main_window.append_bot_log("✅ " + t("ui.no_more_cards_to_load"))
                return False 
            
            self.main_window.append_bot_log(f"📦 {t('ui.loaded_cards', count=len(cards))} (offset: {self.cards_offset})")
            
            # ✅ Converti i dati
            cards_data = []
            for card_data in cards:
                card_dict = {
                    'card_name': card_data[0],
                    'rarity': card_data[1],
                    'account_name': card_data[2],
                    'set_code': card_data[3],
                    'card_number': str(card_data[4]),
                    'thumbnail_blob': card_data[5],
                    'image_url_screenshot': card_data[6],
                    'image_url': card_data[7],
                    'screenshot_thumbnail_blob': card_data[8],
                    'similarity': (card_data[9]) if card_data[9] else 0,
                    'set_cover_blob': card_data[10] # ✅ MODIFICATO: BLOB
                }
                cards_data.append(card_dict)
            
            # ✅ Aggiungi tutte le carte alla tabella
            for card_dict in cards_data:
                self.add_card_to_table(card_dict, insert_at_top=False)
            
            self.cards_offset += 20
            return True 
        
        except Exception as e:
            self.main_window.append_bot_log(f"❌ {t('ui.error_loading_cards')}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def add_card_to_table(self, card_data: dict, insert_at_top: bool = False):
        """Aggiunge una riga alla tabella (in cima o in fondo)."""
        
        row = 0 if insert_at_top else self.cards_table.rowCount()
        self.cards_table.insertRow(row)
        
        # --- COL 0: MINIATURA CARTA (CENTRATA) ---
        card_preview_label = QLabel()
        card_preview_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        card_preview_label.setFixedSize(60, 60)
        image_blob = card_data.get('thumbnail_blob') 
        card_image_url = card_data.get('image_url') or card_data.get('local_image_path')
        
        # Logica mista: Se c'è BLOB usa quello, altrimenti prova URL (con worker)
        if image_blob:
            pixmap = QPixmap()
            pixmap.loadFromData(image_blob) 
            if not pixmap.isNull():
                pixmap = pixmap.scaled(60, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                card_preview_label.setPixmap(pixmap)
            else:
                card_preview_label.setText("❌")
        elif card_image_url:
             # Fallback al download asincrono se non c'è BLOB (raro se scraper funziona)
             self.load_card_image_async(card_image_url, card_preview_label, 60, 60)
        else:
            card_preview_label.setText("🎴")
        
        card_preview_label.setToolTip(self.create_image_tooltip(
            image_blob, card_data.get('card_name', 'Carta')
        ))
        
        # ✅ Wrapper per centramento
        card_wrapper = QWidget()
        card_layout = QHBoxLayout(card_wrapper)
        card_layout.setContentsMargins(5, 5, 5, 5)
        card_layout.addWidget(card_preview_label)
        self.cards_table.setCellWidget(row, 0, card_wrapper)

        # --- COL 1: MINIATURA PACCHETTO (CENTRATA) ---
        pack_preview_label = QLabel()
        pack_preview_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        pack_preview_label.setFixedSize(60, 60)
        pack_blob = card_data.get('screenshot_thumbnail_blob')
        
        if pack_blob:
            pixmap = QPixmap()
            pixmap.loadFromData(pack_blob)
            if not pixmap.isNull():
                pixmap = pixmap.scaled(60, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                pack_preview_label.setPixmap(pixmap)
            else:
                pack_preview_label.setText("❌")
        else:
            pack_preview_label.setText("📦") 
        
        pack_preview_label.setToolTip(self.create_image_tooltip(
            pack_blob, "Screenshot"
        ))
        
        # ✅ Wrapper per centramento
        pack_wrapper = QWidget()
        pack_layout = QHBoxLayout(pack_wrapper)
        pack_layout.setContentsMargins(5, 5, 5, 5)
        pack_layout.addWidget(pack_preview_label)
        self.cards_table.setCellWidget(row, 1, pack_wrapper)
        
        # --- COL 2: NOME ACCOUNT (Testo centrato) ---
        account_item = QTableWidgetItem(card_data.get('account_name', ''))
        account_item.setTextAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        #self.cards_table.setItem(row, 2, account_item)
        
        # ================================================================
        # ✅ COL 3: SET COVER (CENTRATO - DA BLOB)
        # ================================================================
        set_cover_label = QLabel()
        set_cover_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        set_cover_label.setFixedSize(60, 60) 
        
        set_cover_blob = card_data.get('set_cover_blob')

        if set_cover_blob:
            pixmap = QPixmap()
            pixmap.loadFromData(set_cover_blob)
            if not pixmap.isNull():
                set_cover_label.setPixmap(pixmap.scaled(60, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                set_cover_label.setText("❌")
        else:
            set_cover_label.setText("💿") 
            set_cover_label.setStyleSheet("font-size: 24px;")
            set_cover_label.setToolTip(card_data.get('set_code', 'N/A'))

        # ✅ Wrapper per centramento
        cover_wrapper = QWidget()
        cover_layout = QHBoxLayout(cover_wrapper)
        cover_layout.setContentsMargins(5, 5, 5, 5)
        cover_layout.addWidget(set_cover_label)
        self.cards_table.setCellWidget(row, 2, cover_wrapper)
        # ================================================================

        # --- COL 4: NUMERO CARTA (Testo centrato) ---
        card_num_item = QTableWidgetItem(str(card_data.get('card_number', '')))
        card_num_item.setTextAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.cards_table.setItem(row, 3, card_num_item)
        
        # --- COL 5: NOME CARTA (Testo centrato) ---
        card_name_item = QTableWidgetItem(card_data.get('card_name', ''))
        card_name_item.setTextAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.cards_table.setItem(row, 4, card_name_item)
        
        # --- COL 6: RARITÀ (Widget centrato) ---
        rarity_name = card_data.get('rarity', 'NA')
        rarity_widget = QWidget()
        rarity_layout = QHBoxLayout(rarity_widget)
        rarity_layout.setContentsMargins(5, 5, 5, 5)
        rarity_layout.setAlignment(Qt.AlignCenter)
        
        if rarity_name in RARITY_DATA:
            icon_full_path = get_resource_path(RARITY_DATA[rarity_name])
            if os.path.exists(icon_full_path):
                rarity_icon_label = QLabel()
                rarity_icon_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
                pixmap = QPixmap(icon_full_path)
                pixmap = pixmap.scaledToHeight(25, Qt.SmoothTransformation)
                rarity_icon_label.setPixmap(pixmap)
                rarity_icon_label.setToolTip(rarity_name)
                rarity_layout.addWidget(rarity_icon_label)
            else:
                label = QLabel(rarity_name)
                label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
                rarity_layout.addWidget(label)
        else:
            label = QLabel(rarity_name)
            label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
            rarity_layout.addWidget(label)
            
        self.cards_table.setCellWidget(row, 5, rarity_widget)
        
        # ================================================================
        # ✅ COL 7: SIMILARITY (Solo testo colorato, no background)
        # ================================================================
        similarity_value = card_data.get('similarity', 0)
        similarity_item = QTableWidgetItem(f"{similarity_value:.1f}%")
        similarity_item.setTextAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        
        # ✅ Colora SOLO il testo (no background)
        font = QFont()
        font.setBold(True)
        similarity_item.setFont(font)
        
        if similarity_value < 80:
            # Rosso
            similarity_item.setForeground(QColor(200, 0, 0))
        elif similarity_value <= 90:
            # Arancione/Oro
            similarity_item.setForeground(QColor(200, 120, 0))
        else:  # 91-100
            # Verde
            similarity_item.setForeground(QColor(0, 150, 0))
        
        self.cards_table.setItem(row, 6, similarity_item)
        # ================================================================
        
        if insert_at_top:
            self.found_cards_list.insert(0, card_data)
        else:
            self.found_cards_list.append(card_data)

    def clear_cards_list(self):
        """Pulisce la lista delle carte trovate."""
        reply = QMessageBox.question(self, 'Confirm', 
                                     'Are you sure you want to clear the cards list?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.cards_table.setRowCount(0)
            self.found_cards_list.clear()
            self.cards_count_label.setText(t("ui.total_cards_found", count=0))
    
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
                    headers = [self.cards_table.horizontalHeaderItem(c).text() for c in range(self.cards_table.columnCount())]
                    writer.writerow(headers)
                    
                    for row in range(self.cards_table.rowCount()):
                        row_data = []
                        for col in range(self.cards_table.columnCount()):
                            if col == 0 or col == 1 or col == 3 or col == 6:
                                item = self.cards_table.cellWidget(row, col)
                                if isinstance(item, QWidget):
                                    # Estrai il QLabel dal wrapper
                                    layout = item.layout()
                                    if layout and layout.count() > 0:
                                        label = layout.itemAt(0).widget()
                                        if isinstance(label, QLabel):
                                            if not label.pixmap():
                                                row_data.append(label.text())
                                            else:
                                                row_data.append(label.toolTip())
                                        else:
                                            row_data.append("N/A")
                                    else:
                                        row_data.append("N/A")
                                else:
                                    row_data.append("N/A")
                            else:
                                item = self.cards_table.item(row, col)
                                row_data.append(item.text() if item else '')
                        writer.writerow(row_data)
                
                QMessageBox.information(self, "Success", f"Exported {self.cards_table.rowCount()} cards to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export: {str(e)}")

    # --- Funzioni Helper ---

    def create_image_tooltip(self, blob_data, text_fallback=""):
        if not blob_data: return text_fallback
        try:
            b64_data = base64.b64encode(blob_data).decode('utf-8')
            return f'<html><img src="data:image/jpeg;base64,{b64_data}"></html>'
        except Exception as e:
            return text_fallback

    def load_card_image_async(self, image_url: str, target_label: QLabel, scale_w: int, scale_h: int):
        """Carica un'immagine generica in modo asincrono (per carte fallback)."""
        if not image_url or not image_url.startswith('http'):
            target_label.setText("❌") 
            return
        
        pixmap = self.image_cache.get(image_url)
        if pixmap:
            # print(f"CACHE HIT: {image_url}")
            target_label.setPixmap(pixmap.scaled(scale_w, scale_h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            return
        
        # print(f"CACHE MISS: {image_url}")
        target_label.setPixmap(self.placeholder_pixmap.scaled(scale_w, scale_h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        worker = ImageDownloaderWorker(image_url, target_label)
        worker.signals.finished.connect(self.on_image_loaded)
        worker.signals.error.connect(self.on_image_load_error)
        self.image_loader_pool.start(worker)

    @pyqtSlot(bytes, str, QLabel)
    def on_image_loaded(self, image_data: bytes, image_url: str, target_label: QLabel):
        try:
            pixmap = QPixmap()
            pixmap.loadFromData(image_data)
            if pixmap.isNull(): 
                raise Exception("Impossibile caricare QPixmap dai dati")
            
            scaled_pixmap = pixmap.scaled(target_label.width(), target_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # print(f"DOWNLOAD completato. Salvo in cache: {image_url}")
            self.image_cache.put(image_url, scaled_pixmap)
            
            if target_label and target_label.isVisible():
                target_label.setPixmap(scaled_pixmap)
        except Exception as e:
            # print(f"❌ Errore on_image_loaded: {e} | URL: {image_url}")
            if target_label:
                target_label.setText("ERR")
                target_label.setStyleSheet("font-size: 16px; color: red;")

    @pyqtSlot(str, str, QLabel)
    def on_image_load_error(self, error_msg: str, image_url: str, target_label: QLabel):
        """Slot per errore caricamento."""
        # print(f"❌ Fallito caricamento immagine: {error_msg} | URL: {image_url}")
        
        if target_label:
            target_label.setText("💿") 
            target_label.setStyleSheet("font-size: 24px;")