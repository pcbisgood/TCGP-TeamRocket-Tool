# core/scraper_tab.py
"""
Questo modulo contiene il QWidget per la scheda "Database Setup" (Scraper).
"""

# Import PyQt5
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QPushButton, QProgressBar, QTextEdit, QMessageBox, QApplication
)
from PyQt5.QtCore import Qt

# Import standard
import os
import sqlite3
from typing import TYPE_CHECKING

# Import moduli app
from config import DB_FILENAME
from .translations import t
from .threads import ScraperThread # Importa il thread dello scraper

# Type checking
if TYPE_CHECKING:
    from .ui_main_window import MainWindow

class ScraperTab(QWidget):
    
    def __init__(self, main_window: 'MainWindow', parent=None):
        super().__init__(parent)
        
        self.main_window = main_window # Riferimento alla finestra principale
        self.scraper_thread = None     # Riferimento al thread
        
        self.setup_ui()

    def setup_ui(self):
        """Configura l'interfaccia utente di questa scheda."""
        db_layout = QVBoxLayout(self)
        
        # Configuration Group
        config_group = QGroupBox("⚙️ " + t("scraper_ui.scraper_configuration"))
        config_layout = QVBoxLayout(config_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.start_scraper_btn = QPushButton(t("scraper_ui.start_database_setup"))
        self.start_scraper_btn.clicked.connect(self.start_scraper)
        self.start_scraper_btn.setStyleSheet("QPushButton { background-color: #2ecc71; color: white; padding: 8px; font-weight: bold; }")
        
        self.stop_scraper_btn = QPushButton(t("scraper_ui.stop_scraper"))
        self.stop_scraper_btn.clicked.connect(self.stop_scraper)
        self.stop_scraper_btn.setEnabled(False)
        self.stop_scraper_btn.setStyleSheet("QPushButton { background-color: #e74c3c; color: white; padding: 8px; font-weight: bold; }")
        
        self.check_db_btn = QPushButton(t("scraper_ui.check_database"))
        self.check_db_btn.clicked.connect(self.check_database_content)
        
        button_layout.addWidget(self.check_db_btn)
        button_layout.addWidget(self.start_scraper_btn)
        button_layout.addWidget(self.stop_scraper_btn)
        config_layout.addLayout(button_layout)
        db_layout.addWidget(config_group)
        
        # PROGRESS GROUP
        progress_group = QGroupBox(t("scraper_ui.progress"))
        progress_layout = QVBoxLayout(progress_group)
        
        self.scraper_progress_bar = QProgressBar()
        self.scraper_progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 5px;
                color: black;
                background-color: #f0f0f0;
            }
            QProgressBar::chunk {
                background-color: #f39c12;
                border-radius: 3px;
            }
        """)
        progress_layout.addWidget(QLabel("Progresso totale:"))
        progress_layout.addWidget(self.scraper_progress_bar)
        
        self.scraper_info_label = QLabel("Pronto")
        self.scraper_info_label.setStyleSheet("QLabel { font-size: 11px; color: #FFFFFF; margin-top: 10px; }")
        progress_layout.addWidget(self.scraper_info_label)
        
        self.scraper_stats_label = QLabel("")
        self.scraper_stats_label.setStyleSheet("QLabel { font-size: 10px; color: #FFFFFF; }")
        progress_layout.addWidget(self.scraper_stats_label)
        
        db_layout.addWidget(progress_group)
        
        # Log Group
        log_group = QGroupBox(t("scraper_ui.scraper_log"))
        log_layout = QVBoxLayout(log_group)
        self.scraper_log_text = QTextEdit()
        self.scraper_log_text.setReadOnly(True)
        self.scraper_log_text.setStyleSheet("QTextEdit { font-family: 'Courier New'; font-size: 12px; }")
        log_layout.addWidget(self.scraper_log_text)
        db_layout.addWidget(log_group)
        
    def start_scraper(self):
        """Avvia lo scraper."""
        try:
            if self.scraper_thread and self.scraper_thread.isRunning():
                QMessageBox.warning(self, "Avviso", "Scraper già in esecuzione")
                return
            
            self.scraper_thread = ScraperThread()
            
            # Connetti i segnali ai metodi di QUESTA classe
            self.scraper_thread.log_signal.connect(self.append_scraper_log)
            self.scraper_thread.progress_signal.connect(self.on_scraper_progress)
            self.scraper_thread.finished_signal.connect(self.on_scraper_finished)
            
            # Aggiorna UI
            self.start_scraper_btn.setEnabled(False)
            self.stop_scraper_btn.setEnabled(True)
            
            self.scraper_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Errore", f"Impossibile avviare scraper: {e}")
            
    def stop_scraper(self):
        """Ferma lo scraper del database."""
        if self.scraper_thread:
            self.scraper_thread.stop()
            self.scraper_thread = None
        
        self.start_scraper_btn.setEnabled(True)
        self.stop_scraper_btn.setEnabled(False)
        self.append_scraper_log("⏹️ Scraper stopped")

    def append_scraper_log(self, message):
        """Aggiunge un messaggio al log dello scraper."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.scraper_log_text.append(f"[{timestamp}] {message}")
        self.scraper_log_text.verticalScrollBar().setValue(
            self.scraper_log_text.verticalScrollBar().maximum()
        )
    
    def on_scraper_progress(self, progress_info: dict):
        """Aggiorna la progress bar con info dello scraper."""
        try:
            sets_completed = progress_info.get('completed', 0)
            sets_total = progress_info.get('total', 1)
            percent_float = progress_info.get('percentage', 0.0)
            status_string = progress_info.get('status', 'Pronto')
            
            percent_int = int(percent_float)
            
            self.scraper_progress_bar.setValue(percent_int)
            self.scraper_info_label.setText(status_string)
            stats_text = f"Set totali: {sets_completed}/{sets_total}"
            self.scraper_stats_label.setText(stats_text)
            
            QApplication.processEvents()
            
        except Exception as e:
            self.append_scraper_log(f"⚠️ Errore progress: {e}")

    def on_scraper_finished(self, success):
        """Chiamato quando lo scraper finisce."""
        self.start_scraper_btn.setEnabled(True)
        self.stop_scraper_btn.setEnabled(False)
        
        if success:
            QMessageBox.information(self, "✅ Successo", "Scraping completato con successo!")
        else:
            QMessageBox.warning(self, "⚠️ Errore", "Scraping terminato con errori")

    def check_database_content(self):
        """Verifica il contenuto del database (debug)."""
        try:
            with sqlite3.connect(DB_FILENAME) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM sets")
                sets_count = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM cards")
                cards_count = cursor.fetchone()[0]
                
                # ✅ NUOVO: Conta i set che hanno un BLOB
                cursor.execute("SELECT COUNT(*) FROM sets WHERE cover_image_blob IS NOT NULL")
                sets_with_cover_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT set_code, set_name, total_cards FROM sets")
                sets_list = cursor.fetchall()
                
                # ✅ AGGIUNTO: Include l'informazione del BLOB nel messaggio
                msg = f"📊 Database Content:\n\nSets: {sets_count} (Cover BLOB: {sets_with_cover_count})\nCards: {cards_count}\n\n"
                
                if sets_list:
                    msg += "Sets in database:\n"
                    for set_code, set_name, total in sets_list:
                        msg += f"  • {set_code}: {set_name} ({total} cards)\n"
                else:
                    msg += "⚠️ No sets found in database!\n"
                
                QMessageBox.information(self, "Database Check", msg)
                self.append_scraper_log(msg)
        
        except Exception as e:
            QMessageBox.critical(self, t("ui.error"), t("scraper_ui.error_checking_db", error=str(e)))