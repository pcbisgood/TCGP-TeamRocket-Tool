"""discord_client.py - Client Discord per monitoraggio trade"""

# Import standard library
import discord
from discord.ext import commands
import aiohttp
import asyncio
import json
import os
import sqlite3
import time
import re
from datetime import datetime
from threading import Lock, Semaphore
from typing import Optional, List, Dict, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import xml.etree.ElementTree as ET
import io
import cv2
import numpy as np
from PIL import Image
# Import configurazione
from config import (
    ACCOUNTS_DIR, 
    LOG_FILENAME, 
    SEARCH_STRING,
    MAX_CONCURRENT_DOWNLOADS, 
    TCG_IMAGES_DIR, 
    SAVE_INTERVAL,
    ACCOUNT_NAME_PATTERN, 
    REQUEST_TIMEOUT, 
    DOWNLOAD_TIMEOUT,
    DB_FILENAME,
    BATCH_SIZE,
    MAX_RETRIES,
    RETRY_DELAY,
    CHUNK_SIZE,
    SELECTED_RARITIES
)

# Import traduzioni
from .translations import t


# =========================================================================
# 🤖 DISCORD BOT CLIENT
# =========================================================================

class TradeMonitorClient(discord.Client):
    """Client Discord per monitorare i trade e scansionare le carte."""
    
    def __init__(self, *, intents: discord.Intents, log_callback, progress_callback, 
                 trade_callback, status_callback, card_found_callback, 
                 channel_ids: List[int], channels_ready_callback: Callable):
        super().__init__(intents=intents)
        self.log_callback = log_callback
        self.progress_callback = progress_callback
        self.trade_callback = trade_callback
        self.status_callback = status_callback
        self.card_found_callback = card_found_callback
        self.channel_ids = channel_ids # <-- SALVATA LA LISTA
        self.channels_ready_callback = channels_ready_callback
        
        self.initial_scan_done = False
        self.session = None
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
        self.pending_trades = []
        
        # ✅ CACHE
        self.last_message_id_cache = 0
        self.last_cache_update = 0
        
        # ✅ THREAD POOL
        self.card_recognition_executor = ThreadPoolExecutor(
            max_workers=8,
            thread_name_prefix="CardRecognizer"
        )
        
        # ✅ CardRecognizer
        from .card_recognizer import CardRecognizer
        self.card_recognizer = CardRecognizer(
            db_path=DB_FILENAME,
            similarity_threshold=75.0
        )
        self.base_url = "http://www.pkmn-pocket-api.it/api"
        # ================================================================
        # ✅ AGGIUNTO: Connessione DB per questo thread
        # ================================================================
        try:
            self.db_conn = sqlite3.connect(DB_FILENAME, check_same_thread=False, timeout=10.0)
            self.db_conn.execute("PRAGMA journal_mode = WAL")
            self.db_conn.row_factory = sqlite3.Row # Per accedere ai dati come dict
        except Exception as e:
            self.log_callback(f"❌ Errore connessione DB nel Client: {e}")
            raise

        self.db_lock = Lock()


    def _get_channels(self) -> List[discord.TextChannel]:
            """Recupera tutti gli oggetti canale validi DALLA LISTA channel_ids."""
            channels = []
            for channel_id in self.channel_ids:
                # Assicurati che l'ID sia un intero prima di passarlo al bot
                if isinstance(channel_id, str):
                    try: channel_id = int(channel_id)
                    except ValueError: continue
                    
                channel = self.get_channel(channel_id)
                if channel and isinstance(channel, discord.TextChannel):
                    channels.append(channel)
                elif channel_id != 0: 
                    self.log_callback(f"❌ Canale ID '{channel_id}' non trovato o non è un canale di testo.")
            return channels


    def _insert_or_update_inventory(self, account_id, card_id, quantity=1):
        """
        Inserisce o aggiorna il conteggio delle carte nell'inventario.
        CORRETTO: Rimossa la colonna 'last_updated' da account_inventory.
        """
        if not account_id or not card_id:
            return

        try:
            with self.db_lock:
                cursor = self.db_conn.cursor()
                
                # 1. Tenta di inserire (nuovo record)
                # account_id qui è il Device ID (PK)
                cursor.execute("""
                    INSERT OR IGNORE INTO account_inventory (account_id, card_id, quantity) 
                    VALUES (?, ?, ?)
                """, (account_id, card_id, quantity))
                
                # 2. Aggiorna se già esistente (aumenta la quantità)
                cursor.execute("""
                    UPDATE account_inventory 
                    SET quantity = quantity + ?
                    WHERE account_id = ? AND card_id = ?
                """, (quantity, account_id, card_id))
                
                self.db_conn.commit()
                
        except Exception as e:
            self.log_callback(f"❌ Errore aggiornamento inventario per {account_id}: {e}")



    def _create_screenshot_thumbnail(self, image_bytes):
        """Crea un thumbnail 60x60 in-memory dallo screenshot."""
        if not image_bytes:
            return None
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.thumbnail((300, 300), Image.Resampling.LANCZOS)
            
            # Se è WebP/PNG, converti per JPEG (più piccolo)
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=80)
            return output.getvalue()
        except Exception as e:
            self.log_callback(f"⚠️ Errore creazione thumbnail screenshot: {e}")
            return None


    # discord_client.py (dentro TradeMonitorClient)
    async def recover_missing_cards(self, trade_list: List[Dict]):
        """
        Tenta di recuperare le schede non trovate o non scansionate
        per i trade storici riscaricando l'immagine dall'URL.
        """
        processed_count = 0
        error_count = 0

        for i, trade_data in enumerate(trade_list, 1):
            
            # Assicurati che trade_data sia un dict e non una Row SQLite
            if not isinstance(trade_data, dict):
                trade_data = dict(trade_data)
                
            image_url = trade_data.get('image_url')
            message_id = trade_data.get('message_id')
            
            # 💥 CORREZIONE CRUCIALE: Assicurati che l'ID PK sia disponibile
            # Quando recuperiamo dal DB, 'account_id' è già la PK salvata in precedenza.
            account_id_pk = trade_data.get('account_id') 
            if not account_id_pk:
                self.log_callback(f"⚠️ [{i}/{len(trade_list)}] ID account (PK) mancante per {message_id}.")
                error_count += 1
                continue

            if not image_url:
                self.log_callback(f"⚠️ [{i}/{len(trade_list)}] URL immagine non valido per {message_id}.")
                error_count += 1
                continue

            self.log_callback(f"⬇️ [{i}/{len(trade_list)}] Riscarico immagine per {message_id}")

            # Passo 1: Download dell'immagine completa in memoria (bytes)
            image_bytes = await self._download_attachment_to_bytes(image_url)
            
            if not image_bytes:
                # Se il download fallisce, non possiamo scansionare
                self.log_callback(f"⚠️ [{i}/{len(trade_list)}] Immagine non scaricata da URL: {image_url}")
                error_count += 1
                
                # OPTIONAL: Aggiorna lo stato su "Errore Download"
                try:
                    cursor = self.db_conn.cursor()
                    cursor.execute("UPDATE trades SET scan_status = 2 WHERE message_id = ?", (message_id,))
                    self.db_conn.commit()
                except Exception as e:
                    self.log_callback(f"❌ Errore aggiornamento stato DB per {message_id}: {e}")
                
                continue

            # Passo 2: Scansione utilizzando i byte scaricati
            try:
                # 💥 CORREZIONE: Aggiungiamo l'ID PK al trade_data, come se venisse da process_message_batch_fast.
                trade_data['account_id'] = account_id_pk
                trade_data['account_name'] = trade_data.get('account_name', account_id_pk)
                
                # Passa i byte scaricati direttamente alla funzione di scansione
                await self.scan_image_for_cards(trade_data, image_bytes)
                processed_count += 1
            except Exception as e:
                self.log_callback(f"❌ Errore scansione recupero [{i}/{len(trade_list)}] per {message_id}: {e}")
                error_count += 1

        self.log_callback(f"✅ Recupero completato: {processed_count} elaborati, {error_count} errori")


    async def setup_hook(self):
        """Setup del client."""
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
        self.session = aiohttp.ClientSession(connector=connector)
    
    async def close(self):
        """Chiude il client con shutdown SICURO (senza recursion error)."""
        self.log_callback("⏹️ Arresto bot...")
        
        try:
            # ❌ NON fare: asyncio.all_tasks() + cancel (causa RecursionError)
            # ✅ Invece: chiudi direttamente websocket
            
            if self.session:
                await self.session.close()
            
            # Chiudi il thread pool
            if hasattr(self, 'card_recognition_executor'):
                self.card_recognition_executor.shutdown(wait=False)
            if hasattr(self, 'db_conn'):
                self.db_conn.close()            
            # Chiudi la connessione Discord (no tasks.cancel())
            await super().close()
            
            self.log_callback("✅ Bot arrestato")
        
        except asyncio.CancelledError:
            self.log_callback("✅ Bot interrotto")
        except Exception as e:
            self.log_callback(f"⚠️ Errore chiusura: {e}")
        
    async def on_ready(self):
        """
        Esecuzione all'avvio del bot.
        MODIFICATO: Gestisce 4 casi:
        1. DB Vuoto -> Scansione Storica Completa
        2. DB Esistente -> Carica UI
        3. DB Esistente -> Recupera falliti
        4. DB Esistente -> Scansione Incrementale (Nuovi) + Scansione Storica (Vecchi)
        """
        
        self.log_callback("✅ " + t("discord_bot.connected_as", name=self.user.name))
        self.status_callback(t("discord_bot.status_connected"))
        channels_data = {}
        for guild in self.guilds:
            for channel in guild.text_channels:
                # Controlla se il bot può leggere la cronologia per quel canale
                if channel.permissions_for(guild.me).read_message_history:
                    channels_data[channel.id] = channel.name
        
        # Invia la lista alla UI (QThread) per popolare la selezione
        self.channels_ready_callback(channels_data)        
        cursor = self.db_conn.cursor()
        
        try:
            cursor.execute("SELECT COUNT(message_id) FROM trades")
            total_trades_in_db = cursor.fetchone()[0]

            if total_trades_in_db == 0:
                # ================================================================
                # CASO A: Database Vuoto -> Avvia Scansione Storica Completa
                # ================================================================
                self.log_callback("🚀 Database vuoto. Inizio scansione storica completa...")
                await self.perform_historical_scan_streaming()
                
            else:
                # ================================================================
                # CASO B: Database Esistente
                # ================================================================
                self.log_callback(f"Database esistente con {total_trades_in_db} trade.")
                
                # 1. CARICA LA UI
                try:
                    cursor.execute("SELECT * FROM trades ORDER BY processed_at DESC LIMIT 100")
                    recent_trades = cursor.fetchall()
                    for trade_row in recent_trades:
                        self.trade_callback(dict(trade_row))
                except Exception as e:
                    self.log_callback(f"⚠️ Errore caricamento trade recenti: {e}")

                try:
                    self.log_callback("🚀 Inizio scansione storica completa...")
                    cursor.execute("""
                        SELECT * FROM trades 
                        WHERE (scan_status = 0 OR scan_status = 2) 
                        AND image_url IS NOT NULL     -- ✅ Recupera solo quelli con URL
                        AND image_url != ''           -- ✅ Assicurati che l'URL non sia vuoto
                    """)
                    trades_to_reprocess = cursor.fetchall()
                    
                    self.log_callback(f"🔍 Recupero carte per {len(trades_to_reprocess)} messaggi...")
                    if trades_to_reprocess:
                        # Converti da tuple SQLite a dicts per un uso più semplice
                        trade_dicts = [dict(row) for row in trades_to_reprocess]
                        await self.recover_missing_cards(trade_dicts)
                        
                except Exception as e:
                    self.log_callback(f"⚠️ Errore recupero trade: {e}")

                # ================================================================
                # ✅ CORREZIONE: ESEGUI ENTRAMBE LE SCANSIONI
                # ================================================================
                
                # 3. SCANSIONE INCREMENTALE (per messaggi NUOVI, arrivati offline)
                await self.perform_incremental_scan_fast()
                
                # 4. SCANSIONE STORICA (per messaggi VECCHI, se la scansione era incompleta)
                await self.perform_historical_scan_streaming()
                

        except Exception as e:
            self.log_callback(f"❌ Errore critico durante on_ready: {e}")
            import traceback
            traceback.print_exc()
        
        # ================================================================
        # PASSO FINALE: INIZIO MONITORAGGIO IN TEMPO REALE
        # ================================================================
        
        self.initial_scan_done = True
        self.log_callback("👂 Inizio monitoraggio messaggi in tempo reale...")


    def _get_or_create_account(self, display_name: str, pk_id: Optional[str] = None, device_password: Optional[str] = None):
        """
        Ottiene o crea un account nel database (Thread-safe).
        pk_id: Il Device ID univoco (la chiave primaria da salvare in device_account) estratto da XML.
        display_name: Il nome file di fallback (timestamp).
        
        Se pk_id non è fornito, usa display_name come chiave primaria (FALLBACK).
        Ritorna la chiave primaria (PK) utilizzata.
        """
        # 1. Determina la CHIAVE PRIMARIA (PK)
        final_pk = pk_id if pk_id else display_name
        
        if not final_pk:
            return None
            
        try:
            with self.db_lock:
                cursor = self.db_conn.cursor()
                
                # STEP 1: Tenta di Inserire o Ignora.
                # final_pk viene usato come PK (device_account).
                # display_name viene usato come account_name (nome di visualizzazione iniziale).
                cursor.execute("""
                    INSERT OR IGNORE INTO accounts 
                    (device_account, account_name, device_password) 
                    VALUES (?, ?, ?)
                """, (final_pk, display_name, device_password))
                
                # STEP 2: Aggiorna i campi (password e nome display) se il record esiste.
                cursor.execute("""
                    UPDATE accounts 
                    SET device_password = ?, 
                        account_name = ?,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE device_account = ?
                """, (device_password, display_name, final_pk))
                
                self.db_conn.commit()

                return final_pk
                
        except Exception as e:
            self.log_callback(f"⚠️ Errore DB in _get_or_create_account per '{final_pk}': {e}")
            return final_pk
        

    # discord_client.py (dentro TradeMonitorClient)
    async def perform_historical_scan_streaming(self):
        """
        Scansiona i messaggi storici in streaming, ciclando su tutti i canali configurati.
        """
        
        channels = self._get_channels()
        if not channels:
            self.log_callback("❌ Nessun canale valido configurato per la scansione storica.")
            self.initial_scan_done = True
            return
        
        # Prepara la cache MAX_ID per tutti i canali (per evitare di scansionare l'intero storico ogni volta)
        try:
            cursor = self.db_conn.cursor()
            # Ottieni l'ID del messaggio più vecchio da 'trades'
            cursor.execute("SELECT MIN(CAST(message_id AS INTEGER)) FROM trades")
            result = cursor.fetchone()
            oldest_message_id = int(result[0]) if result and result[0] else None
        except Exception as e:
            self.log_callback(f"⚠️ Errore DB (getting MIN_msg_id): {e}")
            oldest_message_id = None
        
        
        total_messages = 0
        processed_messages = 0
        
        # 💥 CICLO SU TUTTI I CANALI
        for i, channel in enumerate(channels, 1):
            self.log_callback(f"🚀 Inizio scansione storica su Canale '{channel.name}' ({i}/{len(channels)})...")

            # Verifica permessi per ogni canale
            if hasattr(channel, 'guild') and channel.guild:
                bot_member = channel.guild.get_member(self.user.id)
                if bot_member:
                    permissions = channel.permissions_for(bot_member)
                    if not permissions.read_message_history:
                        self.log_callback(f"❌ Permesso negato: lettura cronologia messaggi su {channel.name}")
                        continue
            
            try:
                history_iter = channel.history(
                    limit=None,
                    before=discord.Object(id=oldest_message_id) if oldest_message_id else None,
                    oldest_first=False 
                )
                
                async for message in history_iter:
                    total_messages += 1
                    
                    if SEARCH_STRING not in message.content:
                        continue
                    
                    # ✅ PASSO 1: ESTRAI I DATI DAL MESSAGGIO
                    try:
                        trade_data, xml_att, img_att = extract_trade_data_fast(message)
                        trade_data['message_id'] = message.id
                        trade_data['channel_id'] = str(message.channel.id) # <-- AGGIUNTO
                        
                        # Inizializza con il fallback (timestamp) come PK temporanea
                        fallback_pk = trade_data['fallback_account_name']
                        trade_data['account_id'] = fallback_pk 

                    except Exception as e:
                        self.log_callback(f"⚠️ Errore estrazione dati msg {message.id}: {e}")
                        continue
                    
                    # ================================================================
                    # ✅ PASSO 2: LEGGI ALLEGATI IN MEMORIA
                    # ================================================================
                    account_name = trade_data.get('account_name')
                    trade_data['xml_path'] = None
                    trade_data['image_url'] = None
                    xml_content_bytes = None
                    image_content_bytes = None

                    try:
                        if xml_att:
                            xml_content_bytes = await xml_att.read()
                        if img_att:
                            image_content_bytes = await img_att.read()
                            trade_data['image_url'] = img_att.url
                    except Exception as e:
                        self.log_callback(f"⚠️ Errore lettura allegati in RAM (Storico): {e}")
                        continue

                    # ================================================================
                    # ✅ GESTIONE XML IN-MEMORIA E SYNC INVENTARIO
                    # ================================================================
                    device_account_pk = None # ID univoco estratto
                    device_password = None
                    
                    if xml_content_bytes:
                        try:
                            root = ET.fromstring(xml_content_bytes)
                            data = {child.get('name'): child.text for child in root.findall('string')}
                            device_account_pk, device_password = data.get('deviceAccount'), data.get('devicePassword')
                            
                            if device_account_pk:
                                trade_data['xml_path'] = "DB_STORED"
                                # 💥 AGGIORNA TRADE_DATA: Usa l'ID univoco come PK
                                trade_data['account_id'] = device_account_pk 
                                
                        except Exception as e:
                            self.log_callback(f"⚠️ Errore parsing XML in-memory (Storico): {e}")

                    # 💥 CREA/RECUPERA ACCOUNT: Usa l'ID definitivo determinato sopra
                    final_pk = self._get_or_create_account(fallback_pk, device_account_pk, device_password)
                    
                    # Assicura che trade_data['account_id'] abbia il final_pk restituito (per coerenza)
                    if final_pk:
                        trade_data['account_id'] = final_pk

                    # ================================================================
                    # ✅ GESTIONE IMMAGINE (Thumbnail BLOB)
                    # ================================================================
                    screenshot_thumb_blob = None
                    if image_content_bytes:
                        screenshot_thumb_blob = self._create_screenshot_thumbnail(image_content_bytes)
                    
                    # ================================================================
                    # ✅ PASSO 3: INSERISCI IL TRADE NEL DATABASE
                    # ================================================================
                    try:
                        # Usiamo trade_data['account_id'] che contiene la PK definitiva (Device ID o Fallback)
                        lookup_account = trade_data['account_id'] 
                        
                        cursor = self.db_conn.cursor()
                        cursor.execute("""
                            INSERT OR IGNORE INTO trades 
                            (message_id, account_id, account_name, xml_path, image_url, 
                            screenshot_thumbnail_blob, message_link, cards_found_text, scan_status)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            str(trade_data['message_id']),
                            lookup_account,  
                            trade_data['fallback_account_name'], # Account name display
                            trade_data.get('xml_path'),
                            trade_data.get('image_url'),
                            screenshot_thumb_blob,
                            trade_data.get('message_link'),
                            trade_data.get('cards_found_text'),
                            0  # scan_status = 0
                        ))
                        self.db_conn.commit()
                        
                        trade_data['screenshot_thumbnail_blob'] = screenshot_thumb_blob
                        self.trade_callback(trade_data) 
                        
                    except Exception as e:
                        self.log_callback(f"⚠️ Errore INSERT trade msg {message.id}: {e}")
                        continue
                    
                    # ✅ PASSO 4: SCANSIONA IMMAGINE
                    if image_content_bytes:
                        try:
                            await self.scan_image_for_cards(trade_data, image_content_bytes) 
                        except Exception as e:
                            self.log_callback(f"⚠️ Errore scansione immagine: {e}")
                    
                    # Aggiorna progress
                    processed_messages += 1
                    status = f"Scansione storica: {processed_messages} processati ({channel.name})"
                    self.progress_callback({'percent': -1, 'status': status})
                
            except Exception as e:
                self.log_callback(f"❌ Errore scansione storica in {channel.name}: {e}")
                import traceback
                self.log_callback(traceback.format_exc())
            
            finally:
                self.log_callback(f"✅ Scansione storica completata su Canale '{channel.name}'.")

        self.log_callback(f"✅ Scansione storica completata su tutti i canali: {processed_messages} messaggi elaborati")
        self.progress_callback({'percent': 100, 'status': 'Scansione storica completata'})
        self.initial_scan_done = True
        
    async def perform_incremental_scan_fast(self):
        """Scansione incrementale con cache ottimizzato, ciclando su tutti i canali."""
        start_time = time.time()
        
        channels = self._get_channels()
        if not channels:
            self.log_callback("❌ Nessun canale valido configurato per la scansione incrementale.")
            return
            
        # ✅ CACHE OTTIMIZZATO - Aggiorna ogni 60s
        if self.last_cache_update == 0 or (time.time() - self.last_cache_update > 60):
            try:
                cursor = self.db_conn.cursor()
                # Interroga la nuova tabella 'trades' per l'ID del messaggio più recente in tutti i canali
                cursor.execute("SELECT MAX(CAST(message_id AS INTEGER)) FROM trades")
                result = cursor.fetchone()
                if result and result[0]:
                    self.last_message_id_cache = int(result[0])
                self.last_cache_update = time.time()
            except Exception as e:
                self.log_callback(f"⚠️ Errore cache: {e}")
        
        max_msg_id = self.last_message_id_cache
        
        if max_msg_id > 0:
            total_new_messages = 0
            all_new_messages = []
            
            # 💥 CICLA SU TUTTI I CANALI
            for channel in channels:
                try:
                    # Fetch messaggi DOPO l'ID più alto trovato nel DB, indipendentemente dal canale
                    async for message in channel.history(limit=None, after=discord.Object(id=max_msg_id), oldest_first=False):
                        all_new_messages.append(message)
                        
                except Exception as e:
                    self.log_callback(f"❌ Errore fetch in {channel.name}: {e}")
                    continue

            if all_new_messages:
                total_new_messages = len(all_new_messages)
                self.log_callback(f"📨 Trovati {total_new_messages} messaggi in totale nei canali.")
                batch = []
                
                # Ordina i messaggi per ID per elaborazione sequenziale (opzionale ma consigliato)
                all_new_messages.sort(key=lambda m: m.id)

                for i, message in enumerate(all_new_messages):
                    batch.append(message)
                    percent = int(((i + 1) / total_new_messages) * 100) if total_new_messages > 0 else 0
                    status = f"Nuovi messaggi {i+1}/{total_new_messages} ({message.channel.name})"
                    self.progress_callback({'percent': percent, 'status': status})
                    
                    if len(batch) >= BATCH_SIZE:
                        await self.process_message_batch_fast(batch)
                        batch = []
                
                if batch:
                    await self.process_message_batch_fast(batch)
                
                elapsed = time.time() - start_time
                self.log_callback(f"✅ Elaborati {total_new_messages} messaggi in {elapsed:.2f}s")
            else:
                self.log_callback("✅ Nessun nuovo messaggio")    




    async def process_message_batch_fast(self, messages: List) -> int:
        """Processa batch in PARALLELO."""
        if not messages: 
            return 0
        
        trades_to_insert = [] 
        tasks_to_run = []
        
        cursor = self.db_conn.cursor()
        
        for message in messages:
            if SEARCH_STRING not in message.content: 
                continue
            
            try:
                # Controllo duplicato del messaggio
                cursor.execute("SELECT 1 FROM trades WHERE message_id = ?", (str(message.id),))
                if cursor.fetchone(): 
                    continue 
            except Exception as e:
                print(f"Errore controllo duplicato: {e}")
                continue

            trade_data, xml_att, img_att = extract_trade_data_fast(message)
            trade_data['xml_path'] = None
            trade_data['image_url'] = None
            
            xml_content_bytes = None
            image_content_bytes = None
            
            try:
                if xml_att:
                    xml_content_bytes = await xml_att.read()
                if img_att:
                    image_content_bytes = await img_att.read()
                    trade_data['image_url'] = img_att.url
            except Exception as e:
                self.log_callback(f"⚠️ Errore lettura allegati in RAM: {e}")
                continue
            
            # ✅ GESTIONE ACCOUNT CORRETTA - Priorità a deviceAccount
            fallback_account_name = trade_data['fallback_account_name']
            device_password = None
            xml_pk_id = None # ID univoco da XML (se presente)

            if xml_content_bytes:
                try:
                    root = ET.fromstring(xml_content_bytes)
                    data = {child.get('name'): child.text for child in root.findall('string')}
                    
                    d_acc = data.get('deviceAccount')
                    d_pass = data.get('devicePassword')
                    
                    if d_acc:
                        # ✅ Caso 1: XML ha deviceAccount - Usa l'ID univoco come PK
                        device_password = d_pass
                        trade_data['xml_path'] = "DB_STORED"
                        xml_pk_id = d_acc # Usato per chiamare _get_or_create_account
                    else:
                        self.log_callback(f"⚠️ XML senza deviceAccount msg {message.id}, uso fallback: {fallback_account_name}")

                except Exception as e:
                    self.log_callback(f"⚠️ Errore parsing XML in-memory msg {message.id}: {e}")
            
            # 💥 CHIAMA _get_or_create_account: 
            final_pk = self._get_or_create_account(fallback_account_name, xml_pk_id, device_password)
            
            # Aggiorna il dizionario con l'ID definitivo che è stato usato come PK
            trade_data['account_name'] = fallback_account_name # Nome display (originale)
            trade_data['account_id'] = final_pk               # ⬅️ ID univoco (PK)
            
            # ================================================================
            # ✅ GESTIONE IMMAGINE IN-MEMORIA E SCANSIONE
            # ================================================================
            screenshot_thumb_blob = None
            if image_content_bytes:
                screenshot_thumb_blob = self._create_screenshot_thumbnail(image_content_bytes)
                tasks_to_run.append(self.scan_image_for_cards(trade_data, image_content_bytes))
            
            # ================================================================
            # ✅ PREPARA I TRADE NEL DB
            # ================================================================
            trade_data['screenshot_thumbnail_blob'] = screenshot_thumb_blob
            self.trade_callback(trade_data)
            
            # ✅ Usa final_pk come account_id nella tabella trades
            trades_to_insert.append((
                str(trade_data['message_id']),
                final_pk,                      # ← PK definitiva/fallback
                fallback_account_name,         # ← Nome file originale
                trade_data.get('xml_path'),
                trade_data.get('image_url'), 
                screenshot_thumb_blob,       
                trade_data.get('message_link'),
                trade_data.get('cards_found_text'),
                0  # scan_status = 0
            ))

        # Salva tutti i trade nel DB
        if trades_to_insert:
            try:
                cursor.executemany("""
                    INSERT OR IGNORE INTO trades 
                    (message_id, account_id, account_name, xml_path, image_url, 
                    screenshot_thumbnail_blob, message_link, cards_found_text, scan_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, trades_to_insert)
                self.db_conn.commit()
            except Exception as e:
                self.log_callback(f"❌ Errore INSERT batch trades: {e}")

        # Esegui scan tasks
        if tasks_to_run:
            await asyncio.gather(*tasks_to_run, return_exceptions=True)
        
        return len(trades_to_insert)

    async def _download_attachment_to_bytes(self, image_url: str) -> Optional[bytes]:
        """Scarica l'allegato su byte dalla URL, gestendo i tentativi."""
        # Assumiamo che self.session sia una aiohttp.ClientSession
        if not hasattr(self, 'session') or self.session.closed:
            # Se la sessione non è pronta (non dovrebbe accadere in on_ready)
            return None 

        for attempt in range(MAX_RETRIES):
            try:
                # Usa self.session
                async with self.session.get(image_url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                    if resp.status == 200:
                        return await resp.read()
                    elif resp.status in (404, 403):
                        # URL non più valido, fermiamo i tentativi
                        self.log_callback(f"❌ Download fallito (Status {resp.status}) per URL: {image_url}")
                        return None 
                    elif attempt == MAX_RETRIES - 1:
                        return None
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    self.log_callback(f"❌ Errore download finale per {image_url}: {e}")
                    return None
            await asyncio.sleep(RETRY_DELAY)
        return None

    # discord_client.py (dentro la classe TradeMonitorClient)

    async def scan_image_for_cards(self, trade_data, image_bytes):
        """
        Scansiona immagine (dai bytes), aggiorna Trade e popola found_cards/account_inventory.
        CORRETTO: Usa trade_data.get('account_id') come chiave primaria (PK) se disponibile.
        """
        if self.card_recognition_executor._shutdown or not image_bytes:
            return
        
        # ✅ CORREZIONE INPUT: Cerca prima 'account_id' (la PK nella tabella trades)
        # altrimenti usa 'account_name' (la PK settata in 'process_message_batch_fast')
        account_id_pk = trade_data.get('account_id') or trade_data.get('account_name')
        message_id = str(trade_data.get('message_id'))
        
        if not account_id_pk:
            self.log_callback(f"❌ Impossibile determinare ID PK per trade: {message_id}")
            return
        
        # Protezione: Verifica connessione DB
        if self.db_conn is None:
            try:
                self.log_callback("⚠️ Riconnessione al database...")
                self.db_conn = sqlite3.connect(DB_FILENAME, check_same_thread=False, timeout=10.0)
                self.db_conn.execute("PRAGMA journal_mode = WAL")
                self.db_conn.row_factory = sqlite3.Row
                self.log_callback("✅ Database riconnesso")
            except Exception as e:
                self.log_callback(f"❌ Impossibile riconnettersi al DB: {e}")
                return
        
        # --- RECUPERO O CREAZIONE ACCOUNT (CRUCIALE) ---
        device_account = None
        try:
            # ✅ _get_or_create_account() crea l'account (o lo recupera) e ritorna la PK (device_account)
            device_account = self._get_or_create_account(account_id_pk)
            
            if not device_account:
                self.log_callback(f"❌ Impossibile determinare Device ID per account: {account_id_pk}")
                return
            
        except Exception as e:
            self.log_callback(f"❌ Errore durante la verifica/creazione Account: {e}")
            return 

        # --- VARIABILI INIZIALI ---
        scan_status = 2  # Default = Errore
        results_json = "[]"
        results = []

        # --- Esecuzione scansione immagine ---
        try:
            source_img = Image.open(io.BytesIO(image_bytes))

            loop = asyncio.get_event_loop()
            results = await asyncio.wait_for(
                loop.run_in_executor(
                    self.card_recognition_executor,
                    self.card_recognizer.recognize_from_image, 
                    source_img,
                    False, 
                    None, 
                    None   
                ),
                timeout=30.0
            )
            scan_status = 1 
            if results:
                results_json = json.dumps(results)

        except asyncio.TimeoutError:
            self.log_callback(f"⏱️ Timeout: Scansione fallita per {message_id}")
        except RuntimeError as e:
            if "cannot schedule new futures after shutdown" in str(e): 
                return
            raise
        except Exception as e:
            self.log_callback(f"❌ Errore scansione immagine {message_id}: {e}")

        # ================================================================
        # PASSO 1: Aggiorna la tabella 'trades' con i risultati
        # ================================================================
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                UPDATE trades 
                SET scan_status = ?, scan_results_json = ?
                WHERE message_id = ?
            """, (scan_status, results_json, message_id))
            self.db_conn.commit()
        except Exception as e:
            self.log_callback(f"❌ Errore UPDATE trade {message_id}: {e}")

        if not results:
            return 
        
        # ================================================================
        # PASSO 2: Inserimento in found_cards e account_inventory
        # ================================================================
        
        filtered_cards = [c for c in results if c.get('rarity') in SELECTED_RARITIES]
        
        if not filtered_cards:
            return
        
        found_cards_batch = []
        
        # Recupera card_id in batch
        card_identifiers = [(c['set_code'], c['card_number']) for c in filtered_cards]
        
        placeholders = ', '.join(['(?, ?)' for _ in card_identifiers])
        flat_params = [item for pair in card_identifiers for item in pair]
        
        card_id_map = {}
        if placeholders:
            try:
                cursor = self.db_conn.cursor()
                cursor.execute(f"""
                    SELECT id, set_code, card_number FROM cards 
                    WHERE (set_code, card_number) IN ({placeholders})
                """, flat_params)
                card_id_map = {(row[1], row[2]): row[0] for row in cursor.fetchall()}
            except Exception as e:
                self.log_callback(f"❌ Errore batch fetch card_id: {e}. Riprovo singolarmente.")
        
        screenshot_blob = trade_data.get('screenshot_thumbnail_blob')

        for card_data in filtered_cards:
            set_code = card_data.get('set_code')
            card_number = card_data.get('card_number')
            
            card_id = card_id_map.get((set_code, card_number))
            if not card_id:
                # Fallback di recupero se non trovato in batch
                try:
                    cursor = self.db_conn.cursor()
                    cursor.execute("""
                        SELECT id FROM cards 
                        WHERE set_code = ? AND card_number = ?
                    """, (set_code, card_number))
                    row = cursor.fetchone()
                    if row:
                        card_id = row[0]
                        card_id_map[(set_code, card_number)] = card_id
                except Exception as e:
                    self.log_callback(f"❌ Impossibile trovare card_id per {set_code}-{card_number}: {e}")
                    continue

            # --- Aggiorna Found Cards (usa message_id) ---
            found_cards_batch.append((
                card_id,
                message_id,  
                device_account,  # ✅ device_account (PK)
                card_data.get('similarity', 0),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))

            # --- Aggiorna Account Inventory ---
            self._insert_or_update_inventory(device_account, card_id, 1)

            # --- Callback alla UI ---
            callback_data = {
                "account_name": trade_data.get('fallback_account_name', trade_data.get('account_name', 'Unknown')), 
                "card_name": card_data.get('card_name', 'Unknown'),
                "card_number": card_data.get('card_number', '?'),
                "set_code": set_code,
                "rarity": card_data.get('rarity', 'NA'),
                "similarity": card_data.get('similarity', 0),
                "image_url_screenshot": trade_data.get('image_url', ''), 
                "screenshot_thumbnail_blob": screenshot_blob,
                "message_id": message_id,
            }
            self.card_found_callback(callback_data) 
        
        # --- Esecuzione Batch Found Cards ---
        if found_cards_batch:
            try:
                cursor = self.db_conn.cursor()
                cursor.executemany("""
                    INSERT INTO found_cards 
                    (card_id, message_id, account_id, confidence_score, found_at)
                    VALUES (?, ?, ?, ?, ?)
                """, found_cards_batch)
                self.db_conn.commit()
            except Exception as e:
                self.log_callback(f"❌ Errore INSERT batch found_cards: {e}")


    async def on_message(self, message):
        """
        Chiamato quando viene ricevuto un nuovo messaggio.
        MODIFICATO: Rimosso il controllo 'processed_message_ids' e il salvataggio JSON.
        """
        if not self.initial_scan_done:
            return
        
        # Il controllo duplicati è ora gestito da 'process_message_batch_fast'
        if SEARCH_STRING in message.content:
            self.log_callback(t("misc.new_trade_detected", id=message.id))
            await self.process_message_batch_fast([message])
            
            # ❌ RIMOSSO: save_trade_log_fast(self.trade_log)


    def _clean_filename_for_account(filename_raw):
        """
        Pulisce il nome del file per ottenere un identificativo account di fallback.
        Esempio: '23P_20251024212128_3(B).xml' -> '20251024212128'
        """
        if not filename_raw:
            return "unknown_account"
            
        # Rimuovi estensione
        name = filename_raw.rsplit('.', 1)[0]
        
        # Logica specifica: Prendi la parte centrale tra i primi due underscore
        # Es: 23P_20251024212128_3(B) -> split('_') -> ['23P', '20251024212128', '3(B)']
        parts = name.split('_')
        if len(parts) >= 2:
            # Restituisci la seconda parte (spesso è l'ID o Timestamp dell'utente)
            return parts[1]
        
        # Fallback: restituisci il nome intero pulito
        return name




# =========================================================================
# 🔧 FUNZIONI HELPER GLOBALI (Fuori dalla classe TradeMonitorClient)
# =========================================================================

def _clean_filename_for_account(filename_raw):
    """
    Pulisce il nome del file per ottenere un identificativo account di fallback.
    Esempio: '23P_20251024212128_3(B).xml' -> '20251024212128'
    """
    if not filename_raw:
        return "unknown_account"
        
    # Rimuovi estensione
    name = filename_raw.rsplit('.', 1)[0]
    
    # Logica specifica: Prendi la parte centrale tra i primi due underscore
    # Es: 23P_20251024212128_3(B) -> split('_') -> ['23P', '20251024212128', '3(B)']
    parts = name.split('_')
    if len(parts) >= 2:
        # Restituisci la seconda parte (spesso è l'ID o Timestamp dell'utente)
        return parts[1]
    
    # Fallback: restituisci il nome intero pulito
    return name

def extract_trade_data_fast(message):
    """
    Estrae i dati del trade da un messaggio Discord.
    MODIFICATO: Usa _clean_filename_for_account per il fallback.
    """
    content = message.content
    
    # 1️⃣ Estrai il nome file XML (Grezzo) dagli allegati
    xml_filename_raw = None
    for att in message.attachments:
        if att.filename.endswith(".xml"):
            xml_filename_raw = att.filename
            break
    
    # Se non c'è allegato, cerca nel testo "File name: ..."
    if not xml_filename_raw:
        file_pattern = r'File name: ([\w\-\(\)\.]+\.xml)'
        match = re.search(file_pattern, content)
        if match:
            xml_filename_raw = match.group(1)
            
    # 2️⃣ Determina un nome account "Fallback" (dal nome file pulito)
    # ✅ ORA QUESTA CHIAMATA FUNZIONERÀ
    fallback_account_name = _clean_filename_for_account(xml_filename_raw) if xml_filename_raw else "unknown_account"
    
    # 3️⃣ Estrai il testo "Found:"
    cards_found_text = ""
    cards_pattern = r'Found: ([\w\s]+(?:\s*\(x\d+\))?(?:,\s*[\w\s]+\s*\(x\d+\))*)'
    cards_match = re.search(cards_pattern, content)
    if cards_match:
        cards_found_text = cards_match.group(1).strip()
    
    # 4️⃣ Estrai gli allegati oggetti
    xml_att = None
    image_att = None
    
    for att in message.attachments:
        if not xml_att and att.filename.endswith('.xml'):
            xml_att = att
        elif not image_att and att.filename.endswith(('.png', '.jpg', '.jpeg', '.webp')):
            image_att = att
        
        if xml_att and image_att:
            break
    
    return {
        "message_id": message.id,
        "fallback_account_name": fallback_account_name, # ✅ Usato se XML non parsabile
        "account_name": fallback_account_name,          # ✅ Default (verrà sovrascritto)
        "xml_filename_text": xml_filename_raw or "N/A",
        "cards_found_text": cards_found_text,
        "cards_found": cards_found_text,
        "message_link": message.jump_url,
        "elaborato": False,  
        "cards": []  
    }, xml_att, image_att

async def download_attachment_fast(session: aiohttp.ClientSession, attachment,
                                   sub_folder: str, filename: str) -> Tuple[Optional[str], str]:
    """Downloads an attachment from Discord."""
    os.makedirs(sub_folder, exist_ok=True)
    file_path = os.path.join(sub_folder, filename)
    
    if os.path.exists(file_path):
        return file_path, 'skipped'
    
    for attempt in range(MAX_RETRIES):
        try:
            async with session.get(attachment.url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                if resp.status == 200:
                    with open(file_path, 'wb') as f:
                        async for chunk in resp.content.iter_chunked(CHUNK_SIZE):
                            f.write(chunk)
                    return file_path, 'downloaded'
                elif attempt == MAX_RETRIES - 1:
                    return None, 'failed'
        except:
            if attempt == MAX_RETRIES - 1:
                return None, 'failed'
        await asyncio.sleep(RETRY_DELAY)
    
    return None, 'failed'
