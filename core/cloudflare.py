"""cloudflare.py - Gestione Cloudflare Tunnel FIXED"""

import subprocess
import os
import sys
from threading import Thread, Event
from typing import Optional

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox, QTextEdit

# Import configurazione
from config import CLOUDFLARED_PATH, get_app_data_path
# ❌ NON SOVRASCRIVERE: from config import CLOUDFLARE_PASSWORD
from PyQt5.QtCore import Qt
# Import traduzioni
from .translations import t


# =========================================================================
# 🌐 CLOUDFLARE TUNNEL THREAD
# =========================================================================

class CloudflareTunnelThread(QThread):
    """Thread per eseguire Cloudflare Tunnel e ottenere URL pubblico."""
    
    log_signal = pyqtSignal(str)
    url_ready_signal = pyqtSignal(str)
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
            # ✅ FIX: Path di cloudflared.exe migliorato
            if getattr(sys, 'frozen', False):
                # Se è in EXE
                cloudflared_path = os.path.join(sys._MEIPASS, 'cloudflared.exe')
            else:
                # Se è in sviluppo
                cloudflared_path = os.path.join(os.getcwd(), 'cloudflared.exe')
            
            if not os.path.exists(cloudflared_path):
                self.error_signal.emit(
                    f"❌ cloudflared.exe not found at: {cloudflared_path}\n"
                    "Download from: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/"
                )
                return
            
            self.log_signal.emit(f"🌐 Starting Cloudflare Tunnel on port {self.local_port}...")
            
            # ✅ Esegui cloudflared
            self.process = subprocess.Popen(
                [cloudflared_path, 'tunnel', '--url', f'http://localhost:{self.local_port}'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # ✅ Leggi output per catturare URL
            for line in self.process.stdout:
                self.log_signal.emit(f"Cloudflare: {line.strip()}")
                
                # Cerca l'URL pubblico
                if 'trycloudflare.com' in line:
                    import re
                    match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
                    if match:
                        self.public_url = match.group(0)
                        self.url_ready_signal.emit(self.public_url)
                        self.log_signal.emit(f"✅ Public URL: {self.public_url}")
            
        except Exception as e:
            import traceback
            self.error_signal.emit(f"❌ Cloudflare Tunnel error: {str(e)}\n{traceback.format_exc()}")
        finally:
            self.stopped_signal.emit()
    
    def stop_tunnel(self):
        """Ferma il tunnel Cloudflare."""
        if self.process:
            self.log_signal.emit("🛑 Stopping Cloudflare Tunnel...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except:
                self.process.kill()
            self.process = None
        self.quit()


# =========================================================================
# 🔐 CLOUDFLARE PASSWORD DIALOG
# =========================================================================

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
        info = QLabel(
            "⚠️ IMPORTANTE:\n\n"
            "• Accesso LOCALE (localhost): NON richiede password\n"
            "• Accesso PUBBLICO (Cloudflare): RICHIEDE password\n\n"
            "La password proteggerà l'accesso quando condividi via Cloudflare."
        )
        info.setStyleSheet("color: #aaa; margin-bottom: 15px;")
        info.setWordWrap(True)
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
        layout.addWidget(QLabel(t("cloudflare.password")))
        layout.addWidget(self.password_input)
        
        # Status
        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: #888; font-size: 11px; margin-top: 10px;")
        self.check_existing_password()
        layout.addWidget(self.status_label)
        
        # Buttons
        button_layout = QVBoxLayout()
        
        save_btn = QPushButton(t("cloudflare.save_password"))
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
        
        cancel_btn = QPushButton(t("cloudflare.cancel"))
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
        
        # Info finale
        info2 = QLabel("💡 Tip: Salva questa password in un posto sicuro!")
        info2.setStyleSheet("color: #7f8c8d; font-size: 12px; margin-top: 10px;")
        info2.setWordWrap(True)
        info2.setAlignment(Qt.AlignCenter)
        layout.addWidget(info2)
        
        self.setLayout(layout)
    
    def check_existing_password(self):
        """Controlla se esiste una password nel .env"""
        from pathlib import Path
        env_file = Path('.env')
        if env_file.exists():
            try:
                with open(env_file, 'r') as f:
                    for line in f:
                        if line.startswith('CLOUDFLARE_PASSWORD='):
                            password_value = line.split('=', 1)[1].strip()
                            if password_value:
                                self.status_label.setText("✅ " + t("cloudflare.password_already_set"))
                                return
            except:
                pass
        self.status_label.setText("⚠️ " + t("cloudflare.no_password_set"))
    
    def save_password(self):
        """Salva la password nel .env"""
        password = self.password_input.toPlainText().strip()
        
        # Validazione
        if not password:
            QMessageBox.warning(self, t("ui.error"), t("cloudflare.password_empty"))
            return
        
        if len(password) < 6:
            QMessageBox.warning(self, t("ui.error"), t("cloudflare.password_min_length"))
            return
        
        try:
            from pathlib import Path
            
            # Scrivi nel .env
            env_file = Path('.env')
            with open(env_file, 'w') as f:
                f.write(f'CLOUDFLARE_PASSWORD={password}\n')
            
            # Aggiorna la variabile d'ambiente
            os.environ['CLOUDFLARE_PASSWORD'] = password
            
            QMessageBox.information(
                self, 
                t("ui.success"), 
                "✅ " + t("cloudflare.password_saved") + "\n\n" +
                "⚠️ Ricorda:\n" +
                "• Accesso locale: NO password\n" +
                "• Accesso Cloudflare: password richiesta"
            )
            
            self.password = password
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, t("ui.error"), t("cloudflare.error_saving", error=str(e)))