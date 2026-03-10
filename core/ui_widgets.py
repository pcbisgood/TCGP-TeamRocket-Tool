# =========================================================================
# ?? CARD WIDGET - Widget per mostrare una carta nella collezione
# =========================================================================
"""ui_widgets.py - Widget PyQt5 personalizzati"""

# Import PyQt5 - Widgets
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDialog, QScrollArea, QFrame, QMessageBox, QApplication,
    QTableWidget, QTableWidgetItem, QSizePolicy
)

# Import PyQt5 - Core
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QTimer

# Import PyQt5 - GUI
from PyQt5.QtGui import QPixmap, QIcon, QFont, QColor, qGray, qRgb, QPainter, QBrush, QPen

# Import standard library
import os
import sqlite3
import sys
import subprocess
from typing import Optional, Dict
import json               # <-- Aggiungi
from pathlib import Path  # <-- Aggiungi
# Import configurazione
from config import get_resource_path, TCG_IMAGES_DIR, DB_FILENAME, ACCOUNTS_DIR, RARITY_DATA

# Import traduzioni
from .translations import t
import urllib.request


"""ui_widgets.py - Widget PyQt5 personalizzati"""
from PyQt5.QtWidgets import (
    QDialog, QHBoxLayout, QVBoxLayout, QLabel, QFrame, QWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QScrollArea,
    QPushButton, QFileDialog, QMessageBox, QStyle # <-- Aggiungi questi
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QTimer
from PyQt5.QtGui import QPixmap, QIcon, QFont, QColor
import os
import sqlite3
from typing import Optional, Dict

import tempfile
from .ui_dialogs import InjectAccountDialog, SimpleSelectionDialog
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
        self.image_label.setText(t("card.placeholder"))
        
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
        card_num_label = QLabel(t("card.number_label", number=card_data['card_number']))
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
            self.wishlist_btn.setText(t("wishlist.hearted"))
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
            self.wishlist_btn.setText(t("wishlist.unhearted"))
    
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
                    self.image_label.setText(t("card.error_icon"))
            except Exception as e:
                self.image_label.setText(t("card.error_icon"))
                print(t("card.error_loading_image", path=image_path, error=str(e)))
        else:
            self.image_label.setText(t("card.placeholder"))
    
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
            text = str(self.quantity)
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
        
        self.setWindowTitle(t("card_details_ui.title", name=card_data['card_name']))
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
        <b>{t('card_details_ui.card_name')}:</b> {card_data['card_name']}<br>
        <b>{t('card_details_ui.set')}:</b> {card_data.get('set_code', 'N/A')}<br>
        <b>{t('card_details_ui.number')}:</b> #{card_data.get('card_number', 'N/A')}<br>
        <b>{t('card_details_ui.rarity')}:</b> {card_data.get('rarity', 'N/A')}
        """
        info_label = QLabel(info_text)
        info_label.setStyleSheet("QLabel { padding: 10px; background-color: #2a2a2a; border-radius: 5px; }")
        left_layout.addWidget(info_label)
        
        # Pulsante Open Folder
        open_folder_btn = QPushButton(t("card_details_ui.open_card_image_folder"))
        open_folder_btn.clicked.connect(self.open_card_folder)
        open_folder_btn.setStyleSheet("QPushButton { padding: 8px; }")
        left_layout.addWidget(open_folder_btn)
        
        left_layout.addStretch()
        
        layout.addWidget(left_panel)
        
        # ===== PANNELLO DESTRO: Ownership =====
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Titolo
        ownership_title = QLabel(t("card_details_ui.ownership_information"))
        ownership_title.setStyleSheet("QLabel { font-size: 16px; font-weight: bold; padding: 5px; }")
        right_layout.addWidget(ownership_title)
        
        # Tabella degli account
        self.ownership_table = QTableWidget()
        self.ownership_table.setColumnCount(3)
        self.ownership_table.setHorizontalHeaderLabels([
            t("ui.table.account"), 
            t("ui.table.copies"), 
            t("ui.table.actions")
        ])
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
        account_word = t('card_details_ui.account') if total_accounts == 1 else t('card_details_ui.accounts')
        stats_text = f"<b>{t('card_details_ui.total')}:</b> {total_copies} {t('card_details_ui.copies')} {t('card_details_ui.across')} {total_accounts} {account_word}"
        stats_label = QLabel(stats_text)
        stats_label.setStyleSheet("QLabel { padding: 10px; background-color: #2a2a2a; border-radius: 5px; }")
        right_layout.addWidget(stats_label)
        
        # Pulsante Close
        close_btn = QPushButton(t("card_details_ui.close"))
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
                self.image_label.setText(t("card.error_icon") + " " + t("card_details.image_not_found"))
        else:
            self.image_label.setText(t("card.placeholder") + " " + t("card_details.no_image"))
    
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
                    
                    xml_btn = QPushButton(t("card_details_ui.open_xml_folder"))
                    xml_btn.setStyleSheet("QPushButton { padding: 4px 8px; font-size: 10px; }")
                    xml_btn.clicked.connect(lambda checked, acc=account_name: self.open_xml_folder(acc))
                    action_layout.addWidget(xml_btn)
                    
                    self.ownership_table.setCellWidget(row_idx, 2, action_widget)
        
        except Exception as e:
            print(t("card_details.error_loading_ownership", error=str(e)))
            QMessageBox.warning(self, t("ui.error"), t("card_details_ui.failed_load_ownership") + f": {str(e)}")
    
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
            print(t("card_details.error_getting_stats", error=str(e)))
            return 0, 0
    
    def open_card_folder(self):
        """Apre la cartella contenente l'immagine della carta."""
        image_path = self.card_data.get('local_image_path')
        if image_path and os.path.exists(image_path):
            folder = os.path.dirname(os.path.abspath(image_path))
            self.open_folder(folder)
        else:
            QMessageBox.information(self, t("ui.info"), t("card_details_ui.card_image_not_found"))
    
    def open_xml_folder(self, account_name):
        """Apre la cartella XML di un account."""
        try:
            xml_folder = os.path.join(ACCOUNTS_DIR, account_name, "xml")
            if os.path.exists(xml_folder):
                self.open_folder(xml_folder)
            else:
                QMessageBox.information(self, t("ui.info"), t("card_details_ui.xml_folder_not_found", account=account_name))
        except Exception as e:
            QMessageBox.warning(self, t("ui.error"), t("card_details_ui.cannot_open_folder", error=str(e)))
    
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
            QMessageBox.warning(self, t("ui.error"), t("card_details_ui.cannot_open_folder", error=str(e)))


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
            self.image_label.setText(t("card.error_icon") + " " + t("card_details.cannot_load_image"))
        
        layout.addWidget(self.image_label)
        
        # Info label
        info_label = QLabel(t("card_details.info_label", folder=os.path.dirname(image_path), file=os.path.basename(image_path)))
        info_label.setStyleSheet("QLabel { color: #888; font-size: 10px; padding: 5px; }")
        layout.addWidget(info_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        open_folder_btn = QPushButton(t("card_details_ui.open_folder"))
        open_folder_btn.clicked.connect(lambda: self.open_folder(image_path))
        open_folder_btn.setStyleSheet("QPushButton { padding: 5px 15px; }")
        button_layout.addWidget(open_folder_btn)
        
        button_layout.addStretch()
        
        close_btn = QPushButton(t("card_details_ui.close"))
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
            QMessageBox.warning(self, t("ui.error"), t("card_details_ui.cannot_open_folder", error=str(e)))

class CollectionCardDialog(QDialog):
    """
    Finestra di dialogo per mostrare i dettagli e l'immagine grande
    di una carta della collezione.
    Layout: [Dettagli a Sinistra] | [Immagine a Destra]
    """
    def __init__(self, card_id: int, db_manager, parent=None):
        super().__init__(parent)
        
        self.card_id = card_id
        self.db_manager = db_manager
        self.parent_tab = parent
        
        # 1. Recupera i dati completi della carta
        self.card_data = self.fetch_card_data()
        if not self.card_data:
            self.close()
            return

        # 2. Impostazioni della finestra
        self.setWindowTitle(t("collection_card_dialog.title", name=self.card_data['card_name']))
        self.setMinimumSize(600, 450)

        # 3. Layout principale (Orizzontale)
        main_layout = QHBoxLayout(self)
        
        # 4. Crea e aggiungi i pannelli
        details_panel = self.create_details_panel()
        image_panel = self.create_image_panel()
        
        main_layout.addWidget(details_panel, 2) # 2/3 dello spazio (priorità alla tabella)
        main_layout.addWidget(image_panel, 1)   # 1/3 dello spazio (l'immagine si adatterà)

    def _get_mumu_instances(self, mumu_player_path: str) -> Dict[str, str]:
        """
        Legge i file vm_config.json (per la porta) e 
        extra_config.json (per playerName) di MuMu.
        
        Args:
            mumu_player_path: Il percorso DIRETTO alla cartella
                              (es. "C:\\Program Files\\Netease\\MuMuPlayerGlobal-12.0")
        
        Restituisce un dizionario es: {"1": "16416", "2": "16417"}
        """
        instances_map = {}
        vms_path_found = Path(mumu_player_path) / "vms"
        
        if not vms_path_found.exists():
            print(t("mumu.error_vms_not_found", path=str(vms_path_found)))
            return {}

        print(t("mumu.scan_instances", path=str(vms_path_found)))

        for vm_folder in vms_path_found.iterdir():
            if not vm_folder.is_dir() or vm_folder.name.endswith("-base"):
                continue
            
            print(t("mumu.found_instance_folder", name=vm_folder.name))
            
            config_dir = vm_folder / "configs"
            vm_config_path = config_dir / "vm_config.json"
            extra_config_path = config_dir / "extra_config.json"

            if not vm_config_path.exists():
                print(t("mumu.warning_vm_config_not_found", dir=str(config_dir)))
                continue

            port = None
            name = None

            try:
                # 1. Leggi la porta da vm_config.json
                with open(vm_config_path, 'r', encoding='utf-8') as f:
                    vm_config_data = json.load(f)
                
                try:
                    # ================================================================
                    # ✅ CORREZIONE: Aggiunto ['vm'] all'inizio del percorso
                    # ================================================================
                    port = vm_config_data['vm']['nat']['port_forward']['adb']['host_port']
                
                except KeyError:
                    print(t("mumu.error_adb_port_not_found", file=vm_config_path.name))
                    continue # Salta questa istanza

                # 2. Leggi il nome da extra_config.json (se esiste)
                if extra_config_path.exists():
                    with open(extra_config_path, 'r', encoding='utf-8') as f:
                        extra_config_data = json.load(f)
                        name = extra_config_data.get('playerName')
                        print(t("mumu.found_player_name", name=name))
                else:
                    print(t("mumu.warning_extra_config_not_found"))

                # 3. Fallback se 'playerName' non è trovato
                if not name:
                    name = vm_folder.name # Fallback finale al nome cartella
                
                print(t("mumu.instance_uses_port", name=name, port=port))
                instances_map[name] = str(port)
                        
            except Exception as e:
                print(t("mumu.error_reading_json", name=vm_folder.name, error=str(e)))
                continue
                
            if not instances_map:
                print(t("mumu.warning_no_valid_instance_found"))
        
        return instances_map


    def _run_adb_devices(self, adb_path: str) -> set:
        """
        Esegue 'adb devices' e restituisce un set di porte connesse.
        Es: {'127.0.0.1:16384', '127.0.0.1:16385'}
        """
        connected_ports = set()
        try:
            command = [adb_path, "devices"]
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                timeout=5 # 5 sec timeout
            )
            
            lines = result.stdout.strip().splitlines()
            for line in lines:
                if line.strip().endswith("device"):
                    device_id = line.split()[0]
                    if device_id.startswith("127.0.0.1:"):
                        connected_ports.add(device_id)
                        
        except FileNotFoundError:
            QMessageBox.warning(self, t("adb.error_title"), t("adb.command_not_found", path=adb_path))
        except subprocess.TimeoutExpired:
            QMessageBox.warning(self, t("adb.error_title"), t("adb.timeout"))
        except Exception as e:
            QMessageBox.warning(self, t("adb.error_title"), t("adb.scan_error", error=str(e)))
            
        return connected_ports

# Aggiungi questi due metodi a CollectionCardDialog
    
#   def _handle_quantity_change(self, row_index: int, account_id: int, amount_change: int):
#       """Gestisce il click sui pulsanti + e -."""
#       try:
#           # 1. Leggi il valore ATTUALE dalla tabella
#           qty_item = self.owner_table.item(row_index, 1)
#           current_qty = int(qty_item.text())
#           new_qty = current_qty + amount_change
#           
#           # 2. Aggiorna il Database
#           success = self.db_manager.set_inventory_quantity(account_id, self.card_id, new_qty)
#           
#           if not success:
#               QMessageBox.warning(self, "Errore", "Impossibile aggiornare il database.")
#               return
#
#           # 3. Aggiorna la UI
#           if new_qty <= 0:
#               # Rimuovi la riga dalla tabella
#               self.owner_table.removeRow(row_index)
#           else:
#               # Aggiorna il numero nella tabella
#               qty_item.setText(str(new_qty))
#               
#       except Exception as e:
#           print(f"❌ Errore _handle_quantity_change: {e}")
#           QMessageBox.warning(self, "Errore", f"Errore: {e}")

    def _handle_xml_export(self, device_account: str, device_password: str, account_name: str):
        """Genera il file XML e chiede all'utente dove salvarlo."""
        
        if not device_account or not device_password:
            QMessageBox.warning(self, "Dati Mancanti",
                f"Le credenziali 'deviceAccount' o 'devicePassword' non sono impostate per l'account '{account_name}'.\n"
                "Aggiungile nella scheda 'Bot' (gestione account).")
            return

        # 1. Crea il contenuto XML
        xml_content = (
            "<?xml version='1.0' encoding='utf-8' standalone='yes' ?>\n"
            "<map>\n"
            f'    <string name="deviceAccount">{device_account}</string>\n'
            f'    <string name="devicePassword">{device_password}</string>\n'
            "</map>"
        )
        
        # 2. Chiedi all'utente dove salvare
        default_name = f"{account_name}_trade.xml"
        fileName, _ = QFileDialog.getSaveFileName(self, "Salva File XML", default_name, "XML Files (*.xml)")
        
        if fileName:
            try:
                # 3. Scrivi il file
                with open(fileName, 'w', encoding='utf-8') as f:
                    f.write(xml_content)
                
                QMessageBox.information(self, "Successo", f"File XML salvato correttamente in:\n{fileName}")
            except Exception as e:
                QMessageBox.critical(self, "Errore Salvataggio", f"Impossibile salvare il file:\n{e}")


    def _handle_account_inject(self, device_account: str, device_password: str, account_name: str, account_id: int):
        """
        Scansiona istanze, chiede di scegliere, e apre il dialog di iniezione
        passando tutti i dati necessari per il trade.
        """
        if not device_account or not device_password:
            QMessageBox.warning(self, "Dati Mancanti",
                f"Le credenziali 'deviceAccount' o 'devicePassword' non sono impostate per l'account '{account_name}'.")
            return

        # ================================================================
        # 1. RECUPERA INFO (Percorsi locali)
        # ================================================================
        try:
            adb_path_str = "C:\\Program Files\\Netease\\MuMuPlayerGlobal-12.0\\shell\\adb.exe"
            adb_path_obj = Path(adb_path_str)
            if not adb_path_obj.exists():
                QMessageBox.warning(self, "Errore Percorso", f"Percorso ADB non trovato:\n{adb_path_str}")
                return
            
            adb_path = str(adb_path_obj)
            mumu_player_path = str(adb_path_obj.parent.parent) 
            
        except Exception as e:
            QMessageBox.critical(self, "Errore Interno", f"Impossibile definire i percorsi ADB/MuMu: {e}")
            return

        # ================================================================
        # 2. SCANSIONA ISTANZE CONFIGURATE E ATTIVE
        # ================================================================
        QApplication.setOverrideCursor(Qt.WaitCursor)
        configured_instances = self._get_mumu_instances(mumu_player_path) 
        connected_device_ids = self._run_adb_devices(adb_path)
        QApplication.restoreOverrideCursor()
        
        active_ports_set = {dev_id.split(":")[-1] for dev_id in connected_device_ids}

        if not configured_instances:
            QMessageBox.warning(self, "Errore Configurazione",
                f"Nessuna istanza MuMu trovata leggendo i file in:\n{mumu_player_path}\\vms")
            return

        # ================================================================
        # 3. CHIEDI ALL'UTENTE QUALE USARE
        # ================================================================
        selected_port = SimpleSelectionDialog.get_selected_port(
            configured_instances, 
            active_ports_set, 
            self
        )
        
        if not selected_port:
            return # L'utente ha premuto "Annulla"

        # =GA-DA TROVA IL NOME DELL'ISTANZA DALLA PORTA
        selected_name = None
        for name, port in configured_instances.items():
            if port == selected_port:
                selected_name = name
                break
        
        if not selected_name:
             QMessageBox.warning(self, "Errore", "Impossibile trovare il nome per la porta selezionata.")
             return

        # ================================================================
        # 4. CREA XML E AVVIA IL WORKER
        # ================================================================
        xml_content = (
            "<?xml version='1.0' encoding='utf-8' standalone='yes' ?>\n"
            "<map>\n"
            f'    <string name="deviceAccount">{device_account}</string>\n'
            f'    <string name="devicePassword">{device_password}</string>\n'
            "</map>"
        )
        
        temp_xml_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xml', mode='w', encoding='utf-8') as temp_file:
                temp_file.write(xml_content)
                temp_xml_path = temp_file.name
            
            # ✅ PASSA TUTTE LE INFORMAZIONI AL DIALOGO
            inject_dialog = InjectAccountDialog(
                account_name=account_name, 
                temp_xml_path=temp_xml_path,
                adb_path=adb_path,
                selected_port=selected_port, 
                card_id=self.card_id,        # <- Per il log DB
                account_id=account_id,       # <- Per il log DB
                parent=self
            )
            inject_dialog.exec_()
            
        except Exception as e:
            QMessageBox.critical(self, "Errore Creazione File Temporaneo", f"Impossibile creare il file XML temporaneo:\n{e}")
            if temp_xml_path and os.path.exists(temp_xml_path):
                try:
                    os.remove(temp_xml_path)
                except:
                    pass

    def fetch_card_data(self):
        """
        Esegue query al DB per i dettagli completi, l'elenco dei proprietari
        E la copertina del set.
        ✅ CORRETTO: Usa device_account come ID per la SELECT e la JOIN.
        """
        try:
            # Query 1: Dettagli della Carta (invariato)
            query_details = """
                SELECT 
                    c.card_name, s.set_name, c.set_code, 
                    c.card_number, c.rarity, c.thumbnail_blob,
                    s.cover_image_path 
                FROM cards c
                JOIN sets s ON c.set_code = s.set_code
                WHERE c.id = ?
            """
            self.db_manager.cursor.execute(query_details, (self.card_id,))
            result = self.db_manager.cursor.fetchone()
            
            if not result:
                return None
            
            card_data = {
                'card_name': result[0],
                'set_name': result[1],
                'set_code': result[2],
                'card_number': result[3],
                'rarity': result[4],
                'thumbnail_blob': result[5],
                'cover_image_path': result[6]
            }
            
            # Query 2: Elenco Proprietari (CRUCIALE)
            query_owners = """
                SELECT 
                    a.device_account AS account_id,  -- ✅ SELECT: Seleziona la PK device_account e la aliasa
                    a.account_name, 
                    ai.quantity,
                    a.device_account,                -- ✅ device_account (per esportazione)
                    a.device_password
                FROM account_inventory ai
                -- ✅ JOIN: Unisce su account_id (inventario) = device_account (account)
                JOIN accounts a ON ai.account_id = a.device_account 
                WHERE ai.card_id = ? AND ai.quantity > 0
                ORDER BY ai.quantity DESC, a.account_name ASC
            """
            self.db_manager.cursor.execute(query_owners, (self.card_id,))
            owners = self.db_manager.cursor.fetchall()
            
            card_data['owners'] = owners
            
            return card_data
            
        except Exception as e:
            print(f"❌ Errore fetch_card_data (Dialog): {e}")
            return None




    def create_details_panel(self) -> QWidget:
        """Crea il pannello di sinistra con i dettagli testuali."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # --- Dettagli Carta ---
        title_label = QLabel(self.card_data['card_name'])
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #E0E0E0;")
        title_label.setWordWrap(True)
        layout.addWidget(title_label)

        # ================================================================
        # 1. Cover del Set (con download)
        # ================================================================
        set_cover_label = QLabel()
        set_cover_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        cover_path = self.card_data.get('cover_image_path')
        
        pixmap = QPixmap()
        image_loaded = False
        
        if cover_path:
            try:
                if cover_path.startswith('http'):
                    # È un URL, scarichiamolo
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    req = urllib.request.Request(cover_path, headers=headers)
                    with urllib.request.urlopen(req, timeout=3) as response: # Timeout 3 sec
                        image_data = response.read()
                    pixmap.loadFromData(image_data)
                    image_loaded = not pixmap.isNull()
                
                elif os.path.exists(cover_path):
                    # È un percorso locale (fallback)
                    pixmap.load(cover_path)
                    image_loaded = not pixmap.isNull()

            except Exception as e:
                print(f"⚠️ Impossibile caricare la cover del set: {e}")
                image_loaded = False
        
        if image_loaded:
            set_cover_label.setPixmap(pixmap.scaledToHeight(40, Qt.SmoothTransformation))
            set_cover_label.setToolTip(self.card_data['set_name'])
        else:
            # Fallback al testo se la cover fallisce
            set_cover_label.setText(f"{self.card_data['set_name']} ({self.card_data['set_code']})")
            set_cover_label.setStyleSheet("font-size: 14px; color: #AAAAAA;")
        
        layout.addWidget(set_cover_label)

        # --- Numero e Rarità ---
        number_label = QLabel(t("collection_card_dialog.number_label", number=self.card_data['card_number']))
        number_label.setStyleSheet("font-size: 14px; color: #E0E0E0;")
        layout.addWidget(number_label)

        rarity = self.card_data.get('rarity', 'NA')
        rarity_icon_label = QLabel()
        rarity_icon_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        rarity_icon_label.setFixedHeight(25) 
        
        if rarity in RARITY_DATA:
            icon_full_path = get_resource_path(RARITY_DATA[rarity])
            if os.path.exists(icon_full_path):
                pixmap = QPixmap(icon_full_path)
                rarity_icon_label.setPixmap(pixmap.scaledToHeight(25, Qt.SmoothTransformation))
                rarity_icon_label.setToolTip(rarity)
            else:
                rarity_icon_label.setText(rarity)
        else:
            rarity_icon_label.setText(rarity)
        
        layout.addWidget(rarity_icon_label)
        
        # --- Separatore ---
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #555; max-height: 1px; margin-top: 10px; margin-bottom: 5px;")
        layout.addWidget(separator)

        # --- Titolo "Posseduto da:" con contatore ---
        owners = self.card_data.get('owners', [])
        total_copies = sum(owner[2] for owner in owners) if owners else 0
        
        owner_header = QWidget()
        owner_header_layout = QHBoxLayout(owner_header)
        owner_header_layout.setContentsMargins(0, 5, 0, 5)
        
        owner_title = QLabel(t("collection_card_dialog.owned_by"))
        owner_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #E0E0E0;")
        owner_header_layout.addWidget(owner_title)

        if owners:
            copies_badge = QLabel(t("collection_card_dialog.copies_badge", count=total_copies))
            copies_badge.setStyleSheet("""
                background-color: #2E5A44;
                color: #FFFFFF;
                padding: 3px 10px;
                border-radius: 10px;
                font-size: 12px;
                font-weight: bold;
            """)
            owner_header_layout.addWidget(copies_badge)
        
        owner_header_layout.addStretch()
        layout.addWidget(owner_header)

        # --- Lista Account o Messaggio vuoto ---
        if not owners:
            not_owned_container = QWidget()
            not_owned_container.setStyleSheet("""
                background-color: #2A2A2A;
                border: 2px dashed #555;
                border-radius: 8px;
                padding: 20px;
            """)
            not_owned_layout = QVBoxLayout(not_owned_container)
            
            not_owned_label = QLabel(t("collection_card_dialog.not_owned"))
            not_owned_label.setStyleSheet("font-size: 14px; color: #888; font-style: italic;")
            not_owned_label.setAlignment(Qt.AlignCenter)
            not_owned_layout.addWidget(not_owned_label)
            
            layout.addWidget(not_owned_container)
        else:
            # Contenitore con scroll per gli account
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setFrameShape(QFrame.NoFrame)
            scroll_area.setStyleSheet("""
                QScrollArea {
                    border: none;
                    background-color: transparent;
                }
                QScrollBar:vertical {
                    border: none;
                    background: #2A2A2A;
                    width: 10px;
                    border-radius: 5px;
                }
                QScrollBar::handle:vertical {
                    background: #555;
                    border-radius: 5px;
                    min-height: 20px;
                }
                QScrollBar::handle:vertical:hover {
                    background: #666;
                }
            """)
            
            scroll_content = QWidget()
            scroll_layout = QVBoxLayout(scroll_content)
            scroll_layout.setContentsMargins(0, 0, 0, 0)
            scroll_layout.setSpacing(8)
            
            # Ordina gli account per quantità (discendente)
            sorted_owners = sorted(owners, key=lambda x: x[2], reverse=True)
            
            for owner_data in sorted_owners:
                (account_id, account_name, quantity, 
                 device_account, device_password) = owner_data
                
                # Card per ogni account
                account_card = QWidget()
                account_card.setStyleSheet("""
                    QWidget {
                        background-color: #2A2A2A;
                        border: 1px solid #3A3A3A;
                        border-radius: 8px;
                        padding: 10px;
                    }
                    QWidget:hover {
                        border: 1px solid #4A4A4A;
                        background-color: #2F2F2F;
                    }
                """)
                
                card_layout = QHBoxLayout(account_card)
                card_layout.setContentsMargins(10, 8, 10, 8)
                card_layout.setSpacing(10)
                
                # Info account (nome + badge quantità)
                info_layout = QVBoxLayout()
                info_layout.setSpacing(4)
                
                name_label = QLabel(account_name)
                name_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #E0E0E0;")
                info_layout.addWidget(name_label)
                
                qty_container = QWidget()
                qty_layout = QHBoxLayout(qty_container)
                qty_layout.setContentsMargins(0, 0, 0, 0)
                qty_layout.setSpacing(5)
                
                qty_badge = QLabel(t("collection_card_dialog.qty_badge", quantity=quantity))
                qty_badge.setStyleSheet("""
                    background-color: #314C6B;
                    color: #FFFFFF;
                    padding: 2px 8px;
                    border-radius: 8px;
                    font-size: 12px;
                    font-weight: bold;
                """)
                qty_layout.addWidget(qty_badge)
                qty_layout.addStretch()
                
                info_layout.addWidget(qty_container)
                card_layout.addLayout(info_layout, 1)
                
                # Pulsanti azioni (layout verticale compatto)
                actions_layout = QHBoxLayout()
                actions_layout.setSpacing(6)
                
                # Bottone Rimuovi
                minus_btn = QPushButton("−")
                minus_btn.setFixedSize(36, 36)
                minus_btn.setStyleSheet("""
                    QPushButton {
                        font-size: 20px;
                        font-weight: bold;
                        background-color: #5A3A3A;
                        border: 2px solid #8B4444;
                        border-radius: 18px;
                        color: #FFFFFF;
                    }
                    QPushButton:hover {
                        background-color: #6B4545;
                        border: 2px solid #A55555;
                    }
                    QPushButton:pressed {
                        background-color: #4A2A2A;
                    }
                """)
                minus_btn.setToolTip(t("collection_card_dialog.tooltip_remove_copy"))
                minus_btn.clicked.connect(lambda _, a_id=account_id: self._handle_quantity_change(a_id, -1))
                
                # Bottone Aggiungi
                plus_btn = QPushButton("+")
                plus_btn.setFixedSize(36, 36)
                plus_btn.setStyleSheet("""
                    QPushButton {
                        font-size: 20px;
                        font-weight: bold;
                        background-color: #2E5A44;
                        border: 2px solid #449966;
                        border-radius: 18px;
                        color: #FFFFFF;
                    }
                    QPushButton:hover {
                        background-color: #3A6B52;
                        border: 2px solid #55AA77;
                    }
                    QPushButton:pressed {
                        background-color: #244A34;
                    }
                """)
                plus_btn.setToolTip(t("collection_card_dialog.tooltip_add_copy"))
                plus_btn.clicked.connect(lambda _, a_id=account_id: self._handle_quantity_change(a_id, +1))
                
                # Bottone XML
                xml_btn = QPushButton()
                xml_btn.setFixedSize(36, 36) 
                xml_icon = self.style().standardIcon(QStyle.SP_DialogSaveButton) 
                xml_btn.setIcon(xml_icon)
                xml_btn.setIconSize(QSize(20, 20))
                xml_btn.setStyleSheet("""
                        QPushButton {
                            font-size: 16px;
                            background-color: #314C6B;
                            border: 2px solid #4477AA;
                            border-radius: 18px;
                            color: #FFFFFF;
                        }
                        QPushButton:hover {
                            background-color: #3A5A7B;
                            border: 2px solid #5588BB;
                        }
                        QPushButton:pressed {
                            background-color: #243A5A;
                        }
                    """)
                xml_btn.setToolTip(t("collection_card_dialog.tooltip_export_xml"))
                xml_btn.clicked.connect(lambda _, da=device_account, dp=device_password, an=account_name: 
                                        self._handle_xml_export(da, dp, an))                
                
                # Bottone Upload
                upload_btn = QPushButton()
                upload_btn.setFixedSize(36, 36)
                upload_icon = self.style().standardIcon(QStyle.SP_ArrowUp) 
                upload_btn.setIcon(upload_icon)
                upload_btn.setIconSize(QSize(20, 20))
                upload_btn.setStyleSheet("""
                        QPushButton {
                            font-size: 16px;
                            background-color: #5A3A5A; /* Colore viola/magenta */
                            border: 2px solid #AA44AA;
                            border-radius: 18px;
                            color: #FFFFFF;
                        }
                        QPushButton:hover {
                            background-color: #6B456B;
                            border: 2px solid #C055C0;
                        }
                        QPushButton:pressed {
                            background-color: #4A2A4A;
                        }
                    """)
                upload_btn.setToolTip(t("collection_card_dialog.tooltip_inject_account"))
                upload_btn.clicked.connect(lambda _, da=device_account, dp=device_password, an=account_name, acc_id=account_id: 
                                        self._handle_account_inject(da, dp, an, acc_id))

                # Aggiungi i bottoni al layout delle azioni
                actions_layout.addWidget(minus_btn)
                actions_layout.addWidget(plus_btn)
                actions_layout.addWidget(xml_btn)
                actions_layout.addWidget(upload_btn)
                
                # Aggiungi il layout delle azioni al layout della card
                card_layout.addLayout(actions_layout)
                
                scroll_layout.addWidget(account_card)
            
            scroll_layout.addStretch()
            scroll_area.setWidget(scroll_content)
            layout.addWidget(scroll_area)
        
        layout.addStretch()
        return panel

 #   def _handle_quantity_change_by_id(self, account_id: int, delta: int):
 #       """Gestisce il cambio di quantità dato l'account_id."""
 #       # Trova l'indice nella lista owners
 #       owners = self.card_data.get('owners', [])
 #       for idx, owner_data in enumerate(owners):
 #           if owner_data[0] == account_id:
 #               self._handle_quantity_change(idx, account_id, delta)
 #               break


    def clear_layout(self, layout):
        """Helper per pulire un layout prima di ricostruirlo."""
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clear_layout(item.layout())

    def _handle_quantity_change(self, account_id: int, amount_change: int):
        """
        Gestisce il click sui pulsanti + e -.
        Questa versione aggiorna il DB e poi RICARICA l'intero dialog.
        È la soluzione più semplice e robusta per aggiornare la UI.
        """
        try:
            # 1. Trova la quantità corrente dai dati in memoria
            current_qty = 0
            owner_data = None
            for data in self.card_data.get('owners', []):
                if data[0] == account_id: # data[0] è account_id
                    owner_data = data
                    current_qty = data[2] # data[2] è quantity
                    break
            
            if owner_data is None:
                QMessageBox.warning(self, "Errore", "Impossibile trovare l'account nei dati.")
                return

            new_qty = current_qty + amount_change

            # 2. Aggiorna il Database
            # (Assicurati che il tuo db_manager abbia questo metodo)
            try:
                # Assumiamo che db_manager.set_inventory_quantity esista
                # e che il tuo db_manager gestisca commit/connessione.
                # Se il tuo db_manager è quello che ho scritto ieri,
                # questa chiamata è corretta.
                self.db_manager.set_inventory_quantity(account_id, self.card_id, new_qty)
            except Exception as db_e:
                QMessageBox.warning(self, "Errore Database", f"Impossibile aggiornare il database: {db_e}")
                return

            # 3. Aggiorna la UI (La via più semplice: ricarica i dati)
            
            print(f"Aggiornato: Account {account_id}, Carta {self.card_id}, Nuova Qta: {new_qty}")
            
            # Ricarica i dati dal DB
            self.card_data = self.fetch_card_data()
            if not self.card_data:
                self.close() # La carta non è più posseduta? Chiudi
                return

            # Pulisci e ricostruisci i pannelli
            self.clear_layout(self.layout()) # Rimuovi tutto
            
            # Ricostruisci
            details_panel = self.create_details_panel()
            image_panel = self.create_image_panel()
            
            self.layout().addWidget(details_panel, 2)
            self.layout().addWidget(image_panel, 1)

        except Exception as e:
            print(f"❌ Errore _handle_quantity_change: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "Errore", f"Errore: {e}")

#    def create_details_panel(self) -> QWidget:
#        """Crea il pannello di sinistra con i dettagli testuali."""
#        panel = QWidget()
#        layout = QVBoxLayout(panel)
#        layout.setContentsMargins(10, 10, 10, 10)
#        layout.setSpacing(10)
#
#        # --- Dettagli Carta ---
#        title_label = QLabel(self.card_data['card_name'])
#        title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #E0E0E0;")
#        title_label.setWordWrap(True)
#        layout.addWidget(title_label)
#
#        # ================================================================
#        # 1. MODIFICA: Cover del Set (con download)
#        # ================================================================
#        set_cover_label = QLabel()
#        set_cover_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
#        cover_path = self.card_data.get('cover_image_path')
#        
#        pixmap = QPixmap()
#        image_loaded = False
#        
#        if cover_path:
#            try:
#                if cover_path.startswith('http'):
#                    # È un URL, scarichiamolo
#                    headers = {'User-Agent': 'Mozilla/5.0'}
#                    req = urllib.request.Request(cover_path, headers=headers)
#                    with urllib.request.urlopen(req, timeout=3) as response: # Timeout 3 sec
#                        image_data = response.read()
#                    pixmap.loadFromData(image_data)
#                    image_loaded = not pixmap.isNull()
#                
#                elif os.path.exists(cover_path):
#                    # È un percorso locale (fallback)
#                    pixmap.load(cover_path)
#                    image_loaded = not pixmap.isNull()
#
#            except Exception as e:
#                print(f"⚠️ Impossibile caricare la cover del set: {e}")
#                image_loaded = False
#        
#        if image_loaded:
#            set_cover_label.setPixmap(pixmap.scaledToHeight(40, Qt.SmoothTransformation))
#            set_cover_label.setToolTip(self.card_data['set_name'])
#        else:
#            # Fallback al testo se la cover fallisce
#            set_cover_label.setText(f"{self.card_data['set_name']} ({self.card_data['set_code']})")
#            set_cover_label.setStyleSheet("font-size: 14px; color: #AAAAAA;")
#        
#        layout.addWidget(set_cover_label)
#
#        # --- Numero e Rarità ---
#        number_label = QLabel(f"Numero: #{self.card_data['card_number']}")
#        number_label.setStyleSheet("font-size: 14px; color: #E0E0E0;")
#        layout.addWidget(number_label)
#
#        rarity = self.card_data.get('rarity', 'NA')
#        rarity_icon_label = QLabel()
#        rarity_icon_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
#        rarity_icon_label.setFixedHeight(25) 
#        
#        if rarity in RARITY_DATA:
#            icon_full_path = get_resource_path(RARITY_DATA[rarity])
#            if os.path.exists(icon_full_path):
#                pixmap = QPixmap(icon_full_path)
#                rarity_icon_label.setPixmap(pixmap.scaledToHeight(25, Qt.SmoothTransformation))
#                rarity_icon_label.setToolTip(rarity)
#            else:
#                rarity_icon_label.setText(rarity)
#        else:
#            rarity_icon_label.setText(rarity)
#        
#        layout.addWidget(rarity_icon_label)
#        
#        # --- Separatore ---
#        separator = QFrame()
#        separator.setFrameShape(QFrame.HLine)
#        separator.setFrameShadow(QFrame.Sunken)
#        separator.setStyleSheet("background-color: #555; max-height: 1px; margin-top: 10px; margin-bottom: 5px;")
#        layout.addWidget(separator)
#
#        # --- Titolo "Posseduto da:" ---
#        owner_title = QLabel("Posseduto da:")
#        owner_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #E0E0E0;")
#        layout.addWidget(owner_title)
#
#        # --- Tabella Proprietari ---
#        owners = self.card_data.get('owners', [])
#
#        if not owners:
#            not_owned_label = QLabel("Non posseduta da nessun account.")
#            not_owned_label.setStyleSheet("font-size: 13px; color: #888; font-style: italic;")
#            layout.addWidget(not_owned_label)
#        else:
#            self.owner_table = QTableWidget()
#            self.owner_table.setStyleSheet("QTableWidget { border: none; background-color: transparent; }")
#            self.owner_table.setColumnCount(3)
#            self.owner_table.setHorizontalHeaderLabels(["Account", "Qtà", "Azioni"])
#            self.owner_table.setRowCount(len(owners))
#            
#            self.owner_table.verticalHeader().setVisible(False)
#            self.owner_table.setSortingEnabled(True)
#            self.owner_table.verticalHeader().setDefaultSectionSize(40)
#            
#            for row, owner_data in enumerate(owners):
#                (account_id, account_name, quantity, 
#                 device_account, device_password) = owner_data
#                
#                self.owner_table.setItem(row, 0, QTableWidgetItem(account_name))
#                
#                qty_item = QTableWidgetItem()
#                qty_item.setData(Qt.EditRole, quantity) 
#                qty_item.setTextAlignment(Qt.AlignCenter)
#                self.owner_table.setItem(row, 1, qty_item)
#                
#                actions_widget = QWidget()
#                actions_layout = QHBoxLayout(actions_widget)
#                actions_layout.setContentsMargins(5, 5, 5, 5)
#                actions_layout.setSpacing(5)
#                actions_layout.setAlignment(Qt.AlignCenter)
#
#                minus_btn = QPushButton("➖")
#                minus_btn.setFixedSize(30, 30) 
#                minus_btn.setStyleSheet("font-weight: bold; background-color: #643A3A; border: 1px solid #c0392b;")
#                minus_btn.setToolTip("Rimuovi una copia")
#                minus_btn.clicked.connect(lambda _, r=row, a_id=account_id: self._handle_quantity_change(r, a_id, -1))
#                
#                plus_btn = QPushButton("➕")
#                plus_btn.setFixedSize(30, 30) 
#                plus_btn.setStyleSheet("font-weight: bold; background-color: #2E5A44; border: 1px solid #27ae60;")
#                plus_btn.setToolTip("Aggiungi una copia")
#                plus_btn.clicked.connect(lambda _, r=row, a_id=account_id: self._handle_quantity_change(r, a_id, +1))
#
#                xml_btn = QPushButton()
#                xml_btn.setFixedSize(30, 30) 
#                xml_icon = self.style().standardIcon(QStyle.SP_DialogSaveButton) 
#                xml_btn.setIcon(xml_icon)
#                xml_btn.setIconSize(QSize(20, 20))
#                xml_btn.setStyleSheet("background-color: #314C6B; border: 1px solid #3498db;")
#                xml_btn.setToolTip("Esporta credenziali XML")
#                xml_btn.clicked.connect(lambda _, da=device_account, dp=device_password, an=account_name: 
#                                         self._handle_xml_export(da, dp, an))
#
#                actions_layout.addWidget(minus_btn)
#                actions_layout.addWidget(plus_btn)
#                actions_layout.addWidget(xml_btn)
#                
#                self.owner_table.setCellWidget(row, 2, actions_widget)
#            
#            self.owner_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
#            self.owner_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
#            self.owner_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
#            self.owner_table.setColumnWidth(2, 110) 
#            self.owner_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
#            self.owner_table.setAlternatingRowColors(True)
#            
#            self.owner_table.sortByColumn(1, Qt.DescendingOrder)
#            self.owner_table.horizontalHeader().setSortIndicatorShown(True)
#            
#            layout.addWidget(self.owner_table)
#        
#        layout.addStretch()
#        return panel

    def create_image_panel(self) -> QWidget:
        """Crea il pannello di destra con l'immagine della carta."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        
        # ✅ FIX: Carica dal BLOB
        image_blob = self.card_data.get('thumbnail_blob')
        
        if image_blob:
            pixmap = QPixmap()
            pixmap.loadFromData(image_blob) # <-- Carica dati dal BLOB
            
            if not pixmap.isNull():
                # Scala l'immagine
                image_label.setPixmap(pixmap.scaled(
                    350, 500, # Dimensioni massime
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                ))
            else:
                image_label.setText(t("card_image.corrupted_blob"))
                image_label.setStyleSheet("font-size: 16px; color: #888;")

        else:
            # Fallback se il BLOB è nullo
            image_label.setText(t("card_image.not_found"))
            image_label.setStyleSheet("font-size: 16px; color: #888;")
            
        layout.addWidget(image_label)
        return panel