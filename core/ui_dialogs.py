# core/ui_dialogs.py
"""
Contiene dialoghi specializzati, come quello per l'iniezione
di account che esegue script esterni.
"""

# ================================================================
# IMPORT LIBRERIA STANDARD
# ================================================================
import os
import sys
import subprocess
import tempfile
import sqlite3             # <-- CORREZIONE
from datetime import datetime  # <-- CORREZIONE
from typing import Optional, Dict

# ================================================================
# IMPORT PyQt5
# ================================================================
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QTextEdit, 
    QPushButton, QDialogButtonBox, QListWidget, QListWidgetItem,
    QInputDialog         # <-- CORREZIONE
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont, QColor


from config import DB_FILENAME, get_app_data_path
from .translations import t 
# ================================================================
# 1. THREAD WORKER PER ESEGUIRE LO SCRIPT
# ================================================================

class InjectWorker(QThread):
    """
    Esegue l'iniezione ADB (SENZA logica AHK).
    """
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool)

    # ✅ __init__ Semplificato
    def __init__(self, xml_file_path: str, adb_path: str, adb_port: str, parent=None):
        super().__init__(parent)
        self.xml_file_path = xml_file_path
        self.adb_path = adb_path
        self.adb_device = f"127.0.0.1:{adb_port}"
        # (variabili AHK rimosse)
        self._is_running = True

    # (_run_command rimane invariato)
    def _run_command(self, command_args: list, log_command=True):
        if not self._is_running:
            raise InterruptedError("Processo interrotto dall'utente.")
            
        if log_command:
            log_cmd = ' '.join(f'"{arg}"' if ' ' in arg else arg for arg in command_args)
            self.log_signal.emit(f"▶️ Esecuzione: {log_cmd}")
        
        try:
            result = subprocess.run(
                command_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                timeout=15,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            output = result.stdout.strip()
            if output:
                self.log_signal.emit(output)
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, command_args, output)
            self.log_signal.emit("✅ OK\n")
            return True
        except subprocess.TimeoutExpired:
            self.log_signal.emit(f"❌ ERRORE: Timeout dopo 15 secondi.")
            return False
        except Exception as e:
            self.log_signal.emit(f"❌ ERRORE: {e}\n")
            return False

    # ✅ run Semplificato
    def run(self):
        """
        Avvia la sequenza di iniezione ADB.
        Flusso: Stop App -> Inject -> Start App
        """
        try:
            # FASE 1: CHIUDI L'APP POKEMON
            self.log_signal.emit("--- Fase 1: Chiusura App Pokémon ---")
            success = self._run_command([
                self.adb_path, "-s", self.adb_device, 
                "shell", "am", "force-stop", "jp.pokemon.pokemontcgp"
            ])
            if not success:
                raise Exception("Impossibile chiudere l'app.")

            # FASE 2: TRASFERISCI FILE XML
            self.log_signal.emit("--- Fase 2: Trasferimento File XML ---")
            target_xml_path = "/sdcard/deviceAccount.xml"
            success = self._run_command([
                self.adb_path, "-s", self.adb_device,
                "push", self.xml_file_path, target_xml_path
            ])
            if not success:
                raise Exception("Impossibile trasferire il file XML.")

            # FASE 3: COPIA NELLA CARTELLA PROTETTA (con root)
            self.log_signal.emit("--- Fase 3: Copia in cartella protetta (root) ---")
            protected_path = "/data/data/jp.pokemon.pokemontcgp/shared_prefs/deviceAccount:.xml"
            su_command = f"\"cp {target_xml_path} {protected_path}\""
            success = self._run_command([
                self.adb_path, "-s", self.adb_device,
                "shell", "su", "-c", su_command
            ])
            if not success:
                raise Exception("Impossibile copiare il file (serve root?).")

            # FASE 4: ELIMINA FILE TEMPORANEO
            self.log_signal.emit("--- Fase 4: Pulizia file temporaneo ---")
            success = self._run_command([
                self.adb_path, "-s", self.adb_device,
                "shell", "rm", target_xml_path
            ])
            if not success:
                self.log_signal.emit("⚠️ Attenzione: impossibile pulire /sdcard/deviceAccount.xml")

            # ================================================================
            # ✅ FASE 5: RIAVVIA L'APP POKÉMON (REINSERITA)
            # ================================================================
            self.log_signal.emit("--- Fase 5: Riavvio App Pokémon ---")
            start_command = "am start -W -n jp.pokemon.pokemontcgp/com.unity3d.player.UnityPlayerActivity -f 0x10018000"
            success = self._run_command([
                self.adb_path, "-s", self.adb_device,
                "shell", start_command
            ])
            if not success:
                raise Exception("Impossibile riavviare l'app.")
            # ================================================================
            
            self.log_signal.emit("\n🎉 " + t("dialogs.injection_completed_message"))
            self.finished_signal.emit(True)

        except InterruptedError:
            self.log_signal.emit("\n⏹️ Processo interrotto.")
            self.finished_signal.emit(False)
        except Exception as e:
            self.log_signal.emit(f"\n❌ Errore Fatale: {e}")
            self.finished_signal.emit(False)
        finally:
            self.cleanup()


    # (cleanup e stop rimangono invariati)
    def cleanup(self):
        self._is_running = False
        if self.xml_file_path and os.path.exists(self.xml_file_path):
            try:
                os.remove(self.xml_file_path)
                self.log_signal.emit(f"🧹 File temporaneo PC rimosso.")
            except Exception as e:
                self.log_signal.emit(f"⚠️ Impossibile rimuovere il file temporaneo PC: {e}")

    def stop(self):
        self.log_signal.emit("\n⏹️ Richiesta interruzione...")
        self._is_running = False
# ================================================================
# 2. FINESTRA DI DIALOGO PER IL LOG
# ================================================================

class InjectAccountDialog(QDialog):
    """
    Mostra il log di iniezione e gestisce il workflow dello scambio manuale.
    """
    # ✅ __init__ Semplificato
    def __init__(self, account_name: str, temp_xml_path: str, 
                 adb_path: str, selected_port: str, 
                 card_id: int, account_id: int, 
                 parent=None):
        super().__init__(parent)
        
        self.account_name = account_name
        self.temp_xml_path = temp_xml_path
        self.adb_path = adb_path
        self.adb_port = selected_port
        
        # Dati per il trade
        self.card_id = card_id
        self.account_id = account_id
        
        self.worker: Optional[InjectWorker] = None
        
        self.setup_ui()
        self.start_injection()

    # ✅ setup_ui Semplificato (rimossa checkbox)
    def setup_ui(self):
        self.setWindowTitle(t("dialogs.injection_account_title", name=self.account_name))
        self.setMinimumSize(600, 450)
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        
        title_label = QLabel(t("dialogs.injection_start_title", name=self.account_name))
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title_label)
        
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        font = QFont("Monospace")
        font.setStyleHint(QFont.TypeWriter)
        self.log_output.setFont(font)
        self.log_output.setStyleSheet("background-color: #1E1E1E; color: #E0E0E0;")
        layout.addWidget(self.log_output)
        
        # (Checkbox rimossa)
        
        self.close_button = QPushButton(t("dialogs.close_button"))
        self.close_button.setEnabled(False)
        
        button_box = QDialogButtonBox()
        button_box.addButton(self.close_button, QDialogButtonBox.AcceptRole)
        layout.addWidget(button_box)
        
    # ✅ start_injection Semplificato
    def start_injection(self):
        """Avvia il thread worker."""
        self.worker = InjectWorker(
            xml_file_path=self.temp_xml_path,
            adb_path=self.adb_path,
            adb_port=self.adb_port
            # (argomenti AHK e restart rimossi)
        )
        
        self.worker.log_signal.connect(self.log_output.append)
        self.worker.finished_signal.connect(self.on_injection_finished)
        
        self.close_button.setText(t("dialogs.injection_running"))
        self.worker.start()

    # (on_injection_finished rimane invariato)
    def on_injection_finished(self, success: bool):
        if success:
            self.log_output.append("\n✅ " + t("dialogs.injection_finished_log"))
            self.close_button.setText(t("dialogs.injection_done"))
            self.close_button.setEnabled(True)
            self.close_button.setStyleSheet("background-color: #2E5A44; color: white;")
            try:
                self.close_button.clicked.disconnect()
            except:
                pass
            self.close_button.clicked.connect(self.on_fatto_clicked)
        else:
            self.close_button.setText(t("dialogs.close_error"))
            self.close_button.setEnabled(True)
            self.close_button.setStyleSheet("background-color: #643A3A; color: white;")
            try:
                self.close_button.clicked.disconnect()
            except:
                pass
            self.close_button.clicked.connect(self.accept) 

    # ✅ on_fatto_clicked Semplificato (rimosso riavvio AHK)
    def on_fatto_clicked(self):
        """
        Chiede la quantità, logga su DB (entrambe le tabelle) e chiude.
        """
        quantity, ok = QInputDialog.getInt(self, t("dialogs.manual_trade_title"), 
                                           t("dialogs.manual_trade_prompt"), 
                                           1, 1, 1000)
        
        if ok and quantity > 0:
            self.log_manual_trade(quantity)
            self._decrease_inventory_quantity(quantity) # <-- QUESTA È LA LOGICA CHE VOLEVI MANTENERE
        else:
            self.log_output.append("ℹ️ " + t("dialogs.trade_not_registered_log"))
        
        # (Riavvio AHK rimosso)
        
        self.accept() # Chiude la finestra

    # (log_manual_trade rimane invariato)
    def log_manual_trade(self, quantity: int):
        try:
            db_path = get_app_data_path(DB_FILENAME)
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO manual_trades (account_id, card_id, trade_date, quantity_traded)
                    VALUES (?, ?, ?, ?)
                """, (self.account_id, self.card_id, datetime.now().isoformat(), quantity))
                conn.commit()
            self.log_output.append(f"✅ Scambio manuale di {quantity} carte registrato.")
        except Exception as e:
            self.log_output.append(f"❌ Errore registrazione scambio: {e}")

    # ✅ QUESTA FUNZIONE È CORRETTA E VIENE MANTENUTA
    def _decrease_inventory_quantity(self, quantity_removed: int):
        """Sottrae la quantità dall'inventario principale (account_inventory)."""
        try:
            db_path = get_app_data_path(DB_FILENAME)
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute(
                    "SELECT quantity FROM account_inventory WHERE account_id = ? AND card_id = ?",
                    (self.account_id, self.card_id)
                )
                result = cursor.fetchone()
                current_qty = result[0] if result else 0
                
                new_qty = current_qty - quantity_removed
                
                if new_qty <= 0:
                    cursor.execute(
                        "DELETE FROM account_inventory WHERE account_id = ? AND card_id = ?",
                        (self.account_id, self.card_id)
                    )
                    self.log_output.append(f"ℹ️ Carta rimossa dall'inventario (Qta: 0).")
                else:
                    cursor.execute(
                        "UPDATE account_inventory SET quantity = ? WHERE account_id = ? AND card_id = ?",
                        (new_qty, self.account_id, self.card_id)
                    )
                    self.log_output.append(f"ℹ️ Quantità inventario aggiornata a {new_qty}.")
                
                conn.commit()
        except Exception as e:
            self.log_output.append(f"❌ Errore aggiornamento inventario: {e}")

    # (Metodo _restart_ahk ELIMINATO)
            
    # ✅ closeEvent Semplificato
    def closeEvent(self, event):
        """Gestisce la chiusura (es. 'X') prima che il worker finisca."""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1000)
        
        # (Riavvio AHK rimosso)
             
        event.accept()

# ================================================================
# 3. DIALOGO DI SELEZIONE ISTANZA (NUOVO)
# ================================================================

class SimpleSelectionDialog(QDialog):
    """
    Un dialogo che chiede all'utente di scegliere un'istanza MuMu.
    Mostra in verde le istanze attive e in rosso quelle spente.
    """
    def __init__(self, configured_instances: Dict[str, str], active_ports: set, parent=None):
        """
        Args:
            configured_instances: Dizionario {nome_istanza: porta_adb}
            active_ports: Set di porte attive (es. {"16416", "16417"})
        """
        super().__init__(parent)
        self.setWindowTitle(t("dialogs.select_instance_title"))
        self.setMinimumWidth(400) # Leggermente più largo
        self.setModal(True)
        
        self.selected_port = None # Conterrà la porta scelta (es. "16416")
        
        layout = QVBoxLayout(self)
        
        label = QLabel(t("dialogs.select_instance_label"))
        label.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
        layout.addWidget(label)
        
        self.list_widget = QListWidget()
        
        first_active_item = None
        
        # Ordina per nome istanza
        sorted_instances = sorted(configured_instances.items(), key=lambda item: item[0])

        for name, port in sorted_instances:
            item_text = f"{name} (127.0.0.1:{port})"
            item = QListWidgetItem()
            item.setData(Qt.UserRole, port) # Salva la porta come dato
            
            if port in active_ports:
                # 🟢 Attiva (Selezionabile)
                item.setText(f"🟢 {item_text} [{t('dialogs.instance_running')}]")
                item.setForeground(QColor("#2ecc71"))
                if first_active_item is None:
                    first_active_item = item # Salva per la selezione default
            else:
                # 🔴 Spenta (Non selezionabile)
                item.setText(f"🔴 {item_text} [{t('dialogs.instance_stopped')}]")
                item.setForeground(QColor("#e74c3c"))
                # Disabilita l'item
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            
            self.list_widget.addItem(item)
            
        # Seleziona di default il primo elemento ATTIVO
        if first_active_item:
            self.list_widget.setCurrentItem(first_active_item)
            
        self.list_widget.itemDoubleClicked.connect(self.accept)
        layout.addWidget(self.list_widget)
        
        # Pulsanti OK / Annulla
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        
        # Disabilita OK se non c'è nulla di attivo
        if first_active_item is None:
            self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)
            label.setText(t("dialogs.no_instance_active"))

    def accept(self):
        """Salva la porta selezionata prima di chiudere."""
        selected_item = self.list_widget.currentItem()
        # Controlla anche che l'item sia abilitato
        if selected_item and (selected_item.flags() & Qt.ItemIsEnabled):
            self.selected_port = selected_item.data(Qt.UserRole)
            super().accept()
        else:
            # Non chiudere se l'item non è valido
            pass 

    @staticmethod
    def get_selected_port(configured_map: Dict[str, str], active_ports: set, parent=None) -> Optional[str]:
        """
        Metodo helper statico per mostrare il dialogo e restituire la porta.
        """
        # Passa entrambi gli argomenti al costruttore
        dialog = SimpleSelectionDialog(configured_map, active_ports, parent)
        if dialog.exec_() == QDialog.Accepted:
            return dialog.selected_port
        return None