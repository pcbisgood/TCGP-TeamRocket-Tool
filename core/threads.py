# Import standard library
from threading import Thread, Event
import asyncio
import signal
import sys
import os
import sqlite3
from typing import Optional, Callable, List # <-- AGGIUNTO List
# Import PyQt5
from PyQt5.QtCore import QThread, pyqtSignal, QTimer

# Import Discord
import discord

# Import configurazione
from config import MAX_WORKERS, CARD_RECOGNITION_WORKERS, DB_FILENAME, ACCOUNTS_DIR

# Import moduli
from .database import DatabaseManager
from .scraper import TCGPocketScraper  # Importa lo scraper aggiornato
from .discord_client import TradeMonitorClient

# Import traduzioni
from .translations import t


# =========================================================================
# 🧵 THREAD PER LO SCRAPER DEL DATABASE (Aggiornato per Asyncio)
# =========================================================================

class ScraperThread(QThread):
    """Thread per eseguire lo scraper asincrono."""
    
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(dict)  # ✅ Dict con info completa
    clear_log_signal = pyqtSignal()
    finished_signal = pyqtSignal(bool)
    
    def __init__(self, download_images=True):
        super().__init__()
        #self.download_images = download_images
        self.scraper: Optional[TCGPocketScraper] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
    
    def run(self):
        """Esegue lo scraper con validazione e event loop asincrono."""
        try:
            # ✅ Pulisci il log
            self.clear_log_signal.emit()
            
            # ✅ PASSO 1: Valida il database
            self.log_signal.emit("🔧 Validazione database...")
            db_manager = DatabaseManager(log_callback=self.log_signal.emit)
            
            if not db_manager.connect():
                self.log_signal.emit("❌ Impossibile connettersi al database")
                self.finished_signal.emit(False)
                return
            
            if not db_manager.validate_and_repair_database():
                self.log_signal.emit("❌ Database non valido dopo validazione")
                db_manager.close()
                self.finished_signal.emit(False)
                return
            
            db_manager.close()
            
            # ✅ PASSO 2: Crea lo scraper CON CALLBACK PROGRESS
            self.log_signal.emit("✅ Retrieving sets...")
            self.scraper = TCGPocketScraper(
                log_callback=self.log_signal.emit,
                progress_callback=self._on_progress
            )
            
            # ✅ PASSO 3: Esegui lo scraping
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.scraper.scrape_all_parallel())
            
            self.log_signal.emit("✅ Scraping completato!")
            self.finished_signal.emit(True)
            
        except asyncio.CancelledError:
            self.log_signal.emit("⏹️ Scraping cancellato.")
            self.finished_signal.emit(False)
        except Exception as e:
            self.log_signal.emit(f"❌ Scraper error: {str(e)}")
            import traceback
            traceback.print_exc()
            self.finished_signal.emit(False)
        finally:
            if self.scraper:
                try:
                    self.scraper.close()
                except Exception as e:
                    self.log_signal.emit(f"⚠️ Close error: {e}")
            if self.loop and self.loop.is_running():
                self.loop.stop()
            if self.loop:
                self.loop.close()
    
    def _on_progress(self, completed: int, total: int):
        """Riceve aggiornamenti di progresso dallo scraper.
        
        Args:
            completed: Numero di set completati
            total: Numero totale di set
        """
        if total > 0:
            percentage = (completed / total) * 100
            progress_info = {
                'completed': completed,
                'total': total,
                'percentage': percentage,
                'status': f"Scraping set {completed}/{total} ({percentage:.1f}%)"
            }
            self.progress_signal.emit(progress_info)

    
    def stop(self):
        """Richiede di fermare lo scraper."""
        self.log_signal.emit("⏹️ Richiesta di arresto scraper...")
        if self.loop and self.loop.is_running():
            for task in asyncio.all_tasks(self.loop):
                task.cancel()
# =========================================================================
# 🧵 THREAD PER IL DISCORD BOT (Invariato)
# =========================================================================

class DiscordBotThread(QThread):
    """Thread per eseguire il Discord bot - CON RESET COMPLETO."""

    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(dict)
    trade_signal = pyqtSignal(dict)
    status_signal = pyqtSignal(str)
    card_found_signal = pyqtSignal(dict)
    recover_history_signal = pyqtSignal()
    channels_ready_signal = pyqtSignal(dict)

    # 💥 MODIFICATO: Accetta una lista di ID interi
    def __init__(self, token, channel_ids: List[int]): 
        super().__init__()
        self.token = token
        self.channel_ids = channel_ids # <-- Salva la lista
        self.client = None
        self.loop = None
        self._stop_requested = False
        self._shutdown_complete = False
        
    def run(self):
        """Avvia il bot Discord."""
        try:
            # ✅ CONNETTI IL SEGNALE ALL'INIZIO DEL THREAD
            self.recover_history_signal.connect(self._recover_history_handler)
            
            # ✅ RESET: Assicurati che non ci sia un loop precedente
            try:
                current_loop = asyncio.get_event_loop()
                if current_loop.is_running():
                    current_loop.stop()
                current_loop.close()
            except:
                pass

            # ❌ RIMOSSA: os.environ['CHANNEL_ID'] non più necessario

            # ✅ Crea intents
            intents = discord.Intents.default()
            intents.message_content = True
            intents.messages = True
            intents.guilds = True

            # ✅ Crea il client
            self.client = TradeMonitorClient(
                intents=intents,
                log_callback=self.log_signal.emit,
                progress_callback=self.progress_signal.emit,
                trade_callback=self.trade_signal.emit,
                status_callback=self.status_signal.emit,
                card_found_callback=self.card_found_signal.emit,
                channel_ids=self.channel_ids, # <-- PASSATA LA LISTA
                channels_ready_callback=self.channels_ready_signal.emit
            )

            # ✅ NUOVO EVENT LOOP (mai riusato)
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            self.log_signal.emit("🚀 Inizio bot Discord...")
            self.loop.run_until_complete(self.client.start(self.token))

        except asyncio.CancelledError:
            self.log_signal.emit("✅ Bot cancellato")

        except Exception as e:
            self.log_signal.emit(f"❌ Bot error: {str(e)}")
            import traceback
            traceback.print_exc()

        finally:
            # ✅ CLEANUP TOTALE
            self._cleanup()
            self._shutdown_complete = True

    def _cleanup(self):
        """Cleanup TOTALE - Senza task pending."""
        try:
            if self.loop:
                # ✅ STEP 1: Ferma TUTTI i task in sospeso
                pending = asyncio.all_tasks(self.loop)
                for task in pending:
                    task.cancel()

                # ✅ STEP 2: Esegui i cancellamenti
                if pending:
                    try:
                        self.loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                    except:
                        pass

                # ✅ STEP 3: Ferma tutti i timer/callbacks
                for handle in self.loop._ready:
                    handle.cancel()
                self.loop._ready.clear()

                # ✅ STEP 4: Ferma il loop
                if self.loop.is_running():
                    self.loop.stop()

                # ✅ STEP 5: Chiudi il loop
                self.loop.close()
                self.log_signal.emit("✅ Event loop chiuso completamente")

        except Exception as e:
            self.log_signal.emit(f"⚠️ Errore cleanup: {e}")

        finally:
            self.loop = None
            self.client = None
            self.log_signal.emit("✅ Thread fermato")

    def stop(self):
        """Ferma il bot Discord - METODO SICURO."""
        self._stop_requested = True

        if self.client and self.loop and self.loop.is_running():
            try:
                # ✅ Chiudi il client in modo ASINCRONO
                self.loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(self._close_client_safely())
                )

                self.log_signal.emit("⏹️ Richiesta arresto bot...")

                # Aspetta che lo shutdown sia completo (max 3 secondi)
                for _ in range(30):  # 30 * 100ms = 3 secondi
                    if self._shutdown_complete:
                        break
                    QThread.msleep(100)

            except Exception as e:
                self.log_signal.emit(f"⚠️ Errore stop: {e}")

    async def _close_client_safely(self):
        """Chiude il client in modo sicuro."""
        try:
            if self.client and not self.client.is_closed():
                await self.client.close()
                self.log_signal.emit("✅ Client chiuso")
        except Exception as e:
            self.log_signal.emit(f"⚠️ Errore chiusura client: {e}")

    def _recover_history_handler(self):
        """Handler per il recupero storico (usa scan incrementale)."""
        if self.client and self.loop and self.loop.is_running():
            try:
                # Usa call_soon_threadsafe per safety
                asyncio.run_coroutine_threadsafe(
                    self.client.perform_incremental_scan_fast(),
                    self.loop
                )

                self.log_signal.emit("🔄 Inizio recupero messaggi...")

            except Exception as e:
                self.log_signal.emit(f"❌ Errore recupero: {e}")
        else:
            self.log_signal.emit("⚠️ Bot non pronto")

    def request_historical_scan(self):
        """Richiede una scansione storica."""
        self.recover_history_signal.emit()

    def is_running_safe(self):
        """Verifica se il bot sta girando SAFELY."""
        return self.isRunning() and self.loop is not None and self.loop.is_running()

# =========================================================================
# 🧵 THREAD PER IL CARICAMENTO DELLA COLLEZIONE (Invariato)
# =========================================================================

class CollectionLoaderThread(QThread):
    """Thread per caricare la collezione in background."""
    
    progress_signal = pyqtSignal(str)  # Messaggi di stato
    set_ready_signal = pyqtSignal(object, str, str, int, str, str, dict)  # (frame, set_code, set_name, total, cover, account, inventory)
    finished_signal = pyqtSignal(int, int)  # (total_owned, total_cards)
    error_signal = pyqtSignal(str)
    progress = pyqtSignal(str)  # Invia messaggio progress
    set_ready = pyqtSignal(str, str, int, str, str, dict)  # set_code, name, total, cover, account, inventory
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, db_filename, selected_account):
        super().__init__()
        self.db_filename = db_filename
        self.selected_account = selected_account
        self.sets_data = []
    
# In threads.py

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
                    if self.selected_account == t("ui.all_accounts"):
                        cursor.execute("""
                            SELECT COUNT(DISTINCT c.id)
                            FROM cards c
                            JOIN account_inventory ai ON c.id = ai.card_id
                            WHERE c.set_code = ? AND ai.quantity > 0
                        """, (set_code,))
                    else:
                        # LA PARTE PROBLEMATICA E' QUI
                        # Assicurati che la JOIN usi 'a.account_id' e non 'a.id'
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
                    if self.selected_account == t("ui.all_accounts"):
                        cursor.execute("""
                            SELECT c.id, SUM(ai.quantity)
                            FROM cards c
                            JOIN account_inventory ai ON c.id = ai.card_id
                            WHERE c.set_code = ?
                            GROUP BY c.id
                        """, (set_code,))
                    else:
                        # E ANCHE QUI
                        # Assicurati che la JOIN usi 'a.account_id' e non 'a.id'
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
            # L'errore "no such column: a.id" viene catturato qui
            self.error_signal.emit(f"Error: {str(e)}\n{traceback.format_exc()}")

# threads.py (Aggiungi la nuova classe prima di DiscordBotThread)

class DiscordChannelLoaderClient(discord.Client):
    """Client leggero per la sola connessione iniziale e recupero canali."""
    def __init__(self, *, intents, callback):
        super().__init__(intents=intents)
        self.callback = callback
        
    async def on_ready(self):
        channels_data = {}
        for guild in self.guilds:
            for channel in guild.text_channels:
                # Controlla se il bot può leggere la cronologia per quel canale
                if channel.permissions_for(guild.me).read_message_history:
                    channels_data[channel.id] = channel.name
        
        # Invia i dati alla UI
        self.callback(channels_data)
        
        # Chiudi immediatamente la connessione dopo aver inviato i dati
        await self.close()
        
    async def on_disconnect(self):
        # Assicura che il loop si chiuda dopo la disconnessione
        if self.loop and self.loop.is_running():
             self.loop.stop()


class DiscordChannelLoaderThread(QThread):
    """Thread leggero per connettersi e recuperare la lista dei canali."""

    channels_ready_signal = pyqtSignal(dict)
    log_signal = pyqtSignal(str)

    def __init__(self, token, parent=None):
        super().__init__(parent)
        self.token = token
        self.client = None
        self.loop = None
        self._stop_requested = False

    def run(self):
        try:
            self.log_signal.emit("🚀 " + t("discord_bot.background_connection_start"))

            intents = discord.Intents.default()
            intents.message_content = False
            intents.messages = False
            intents.guilds = True

            self.client = DiscordChannelLoaderClient(
                intents=intents,
                callback=self.channels_ready_signal.emit
            )

            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.client.start(self.token))

        except Exception as e:
            self.log_signal.emit("❌ " + t("discord_bot.background_connection_error", e=str(e)))
        finally:
            if self.loop and self.loop.is_running():
                self.loop.stop()
            if self.client and not self.client.is_closed():
                self.loop.run_until_complete(self.client.close())
            if self.loop:
                 self.loop.close()
            self.log_signal.emit("✅ " + t("discord_bot.thread_completed"))