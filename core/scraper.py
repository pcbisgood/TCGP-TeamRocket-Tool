"""scraper.py - Web scraping TCG Pocket (Versione asincrona ad alte prestazioni)"""

# Import standard library
import requests  # Mantenuto per il download iniziale dei proxy
from bs4 import BeautifulSoup
import sqlite3
import aiohttp
import asyncio
import os
import time
import io
import random
import re
from datetime import datetime
from threading import Lock
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Tuple
import json
import cv2
from PIL import Image

# Import configurazione
from config import (
    PROXIES_FILE, 
    PROXY_API_URL, 
    BATCH_SIZE, 
    MAX_RETRIES,
    RETRY_DELAY, 
    CHUNK_SIZE, 
    SIMILARITY_THRESHOLD,
    TEMPLATE_CROP_BOX, 
    SOURCE_ROI_BOXES, 
    HASH_SIZE,
    TEMPLATE_DOWNSCALE_FACTOR, 
    REQUEST_TIMEOUT,
    TCG_IMAGES_DIR
)
print(TCG_IMAGES_DIR)
# Import database
from .database import DatabaseManager


# Numero di set da processare in parallelo
# Mantenuto basso (come l'originale max_workers=3) per non sovraccaricare
SET_CONCURRENCY = 7

# Numero di carte da processare in parallelo (per set)
# Può essere molto più alto con asyncio
CARD_CONCURRENCY = 50 

class TCGPocketScraper:
    """
    Scraper asincrono ad alte prestazioni per TCG Pocket.
    """
    
    # In scraper.py

    def __init__(self, base_url="http://www.pkmn-pocket-api.it/api",
                 log_callback=None, progress_callback=None):
        
        self.base_url = base_url
        self.log_callback = log_callback or print
        self.progress_callback = progress_callback
        self.db_manager = DatabaseManager(log_callback=log_callback)
        self.db_lock = Lock()
        
        # Gestione sessione asincrona
        self.session: aiohttp.ClientSession = None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # ❌ RIMOSSO: Tutta la logica dei proxy
        
        # Setup database
        try:
            self.db_manager.connect()
            self.db_manager.setup_database()
            os.makedirs(TCG_IMAGES_DIR, exist_ok=True) 
            self.log_callback("✅ Scraper inizializzato (Connessione Diretta).")
        except Exception as e:
            self.log_callback(f"❌ Errore setup scraper: {e}")
            raise

    async def setup_session(self):
        """Inizializza la sessione aiohttp."""
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession(headers=self.headers)
            self.log_callback("🚀 Sessione asincrona avviata")

    async def close_session(self):
        """Chiude la sessione aiohttp."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.log_callback("🛑 Sessione asincrona chiusa")
    
#    def fetch_and_save_proxies(self, api_url):
#        """Scarica i proxy dall'API (Sincrono)."""
#        self.log_callback(f"📥 Scaricamento proxy da API...")
#        try:
#            # Usiamo requests qui perché è un'operazione singola all'avvio
#            response = requests.get(api_url, timeout=10)
#            response.raise_for_status()
#            data = response.json()
#            
#            proxy_details_list = data.get('proxies', [])
#            ip_port_list = [
#                f"{p['ip']}:{p['port']}" 
#                for p in proxy_details_list if 'ip' in p and 'port' in p
#            ]
#            
#            if ip_port_list:
#                with open(self.proxy_file, 'w') as f:
#                    f.write("\n".join(ip_port_list))
#                self.log_callback(f"✅ {len(ip_port_list)} proxy salvati in {self.proxy_file}")
#            else:
#                self.log_callback("⚠️ Nessun proxy trovato nella risposta API")
#                
#        except Exception as e:
#            self.log_callback(f"⚠️ Errore download proxy: {e}. Continuo con file esistente...")
#    
#    def load_proxies(self, proxy_file):
#        """Carica i proxy dal file (Sincrono)."""
#        proxies = []
#        try:
#            with open(proxy_file, 'r') as f:
#                for line in f:
#                    proxy_ip_port = line.strip()
#                    if proxy_ip_port:
#                        # Formato per aiohttp
#                        proxies.append(f"http://{proxy_ip_port}")
#            return proxies if proxies else [None]
#        except FileNotFoundError:
#            self.log_callback(f"⚠️ File proxy {proxy_file} non trovato")
#            return [None]
#        except Exception as e:
#            self.log_callback(f"⚠️ Errore caricamento proxy: {e}")
#            return [None]
#    
#    def get_random_proxy(self):
#        """Ritorna un proxy casuale."""
#        return random.choice(self.proxies)
    
# In scraper.py

    async def get_json(self, url, max_retries=MAX_RETRIES):
        """Esegue richiesta HTTP asincrona per JSON."""
        
        # ‼️ MODIFICA: Rimuoviamo il proxy per accedere alla TUA API.
        # Il tuo server (pkmn-pocket-api.it) sta bloccando i tuoi stessi proxy (Errore 403).
        
        for attempt in range(max_retries):
            try:
                # proxy_to_use = self.get_random_proxy() # <-- RIMOSSO
                
                async with self.session.get(url, timeout=REQUEST_TIMEOUT) as response: # <-- proxy= RIMOSSO
                    response.raise_for_status()
                    return await response.json()
            except Exception as e:
                if attempt == max_retries - 1:
                    self.log_callback(f"❌ Fallito get_json: {url} dopo {max_retries} tentativi ({e})")
                    raise
                await asyncio.sleep(RETRY_DELAY)
        return None

    # In scraper.py

    # In scraper.py

    # In scraper.py

    async def get_bytes(self, url, max_retries=MAX_RETRIES):
        """
        Esegue richiesta HTTP asincrona per BYTES (immagini).
        MODIFICATO: Rimosso il proxy per evitare l'errore 403 dal nostro stesso server.
        """
        for attempt in range(max_retries):
            try:
                # ❌ RIMOSSO: proxy_to_use = self.get_random_proxy()

                # Usiamo l'header User-Agent definito in __init__
                async with self.session.get(url, timeout=REQUEST_TIMEOUT) as response: # <-- proxy= RIMOSSO
                    response.raise_for_status()
                    return await response.read()
            except Exception as e:
                if attempt == max_retries - 1:
                    err_msg = f"❌❌❌ FALLITO DOWNLOAD IMMAGINE: {url} dopo {max_retries} tentativi ({e})"
                    print(err_msg)
                    self.log_callback(err_msg)
                    return None
                await asyncio.sleep(RETRY_DELAY)
        return None
    # ================================================================
    # SCRAPING SET (Convertito ad async)
    # ================================================================
    
    # In scraper.py

    # In scraper.py

    async def get_sets(self):
        """Recupera tutti i set dalla TUA API JSON."""
        url = f"{self.base_url}/sets" # <-- USA IL NUOVO BASE URL
        self.log_callback(f"📋 Recupero lista set da {url}...")
        
        try:
            sets_data = await self.get_json(url) 
            
            if not sets_data:
                self.log_callback("⚠️ API non ha ritornato set.")
                return []

            sets = []
            for set_item in sets_data:
                sets.append({
                    'code': set_item.get('set_code'),
                    'name': set_item.get('set_name'),
                    'release_date': set_item.get('release_date', ''),
                    'total_cards': set_item.get('total_cards', 0),
                    'url': set_item.get('set_url'), 
                    'cover_image_url': set_item.get('cover_image_url')
                })
            
            self.log_callback(f"✅ Trovati {len(sets)} set dall'API")
            return sets
            
        except Exception as e:
            self.log_callback(f"❌ Errore recupero set API: {e}")
            return []

    # In scraper.py

    async def get_cards_from_set(self, set_code):
        """
        Recupera tutte le carte da un set dalla TUA API JSON.
        MODIFICATO: Gestisce la struttura [ { "cards": [...] } ]
        """
        url = f"{self.base_url}/cards?set_code={set_code}"
        self.log_callback(f"🃏 Scaricamento carte JSON da {set_code}...")
        
        try:
            # 1. L'API ritorna una lista: [ { "cards": [...] } ]
            json_response = await self.get_json(url)
            
            # 2. Controlla che sia una lista e che non sia vuota
            if not isinstance(json_response, list) or not json_response:
                self.log_callback(f"❌ Errore API: La risposta per {set_code} non era una lista valida o era vuota.")
                return []
            
            # 3. Prendi il primo oggetto della lista
            first_object = json_response[0]
            
            # 4. Estrai la lista di carte dalla chiave "cards"
            cards = first_object.get('cards')
            
            if not isinstance(cards, list):
                self.log_callback(f"❌ Errore API: Chiave 'cards' non trovata o non è una lista in {set_code}.")
                return []
                
            self.log_callback(f"✅ Trovate {len(cards)} carte nel set {set_code} (JSON)")
            return cards
            
        except Exception as e:
            self.log_callback(f"❌ Errore recupero carte API {set_code}: {e}")
            return []
    
    # ================================================================
    # DETTAGLI CARTA (Convertito ad async)
    # ================================================================
    
    async def get_card_details(self, card_url):
        """Recupera dettagli carta (nome e rarità) - (Async)"""
        try:
            html = await self.get_html(card_url)
            soup = BeautifulSoup(html, 'html.parser')
            
            card_name = ""
            name_elem = soup.select_one('span.card-text-name a, p.card-text-title span.card-text-name a')
            if name_elem:
                card_name = name_elem.get_text(strip=True)
            
            rarity = ""
            details_div = soup.find('div', class_='prints-current-details')
            if details_div:
                text = details_div.get_text(strip=True)
                match = re.search(r'#\d+\s*·\s*([^·]+)', text)
                if match:
                    rarity = match.group(1).strip()
            
            return {
                'card_name': card_name,
                'rarity': rarity
            }
            
        except Exception as e:
            self.log_callback(f"⚠️ Errore dettagli carta: {e}")
            return {}
    
    # ================================================================
    # DOWNLOAD IMMAGINI (Convertito ad async)
    # ================================================================
    
    async def download_image(self, image_url, local_path):
        """Scarica un'immagine (Async)."""
        try:
            if os.path.exists(local_path):
                return local_path
            
            image_bytes = await self.get_bytes(image_url)
            
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'wb') as f:
                f.write(image_bytes)
            
            return local_path
            
        except Exception as e:
            self.log_callback(f"⚠️ Errore download immagine: {e}")
            return None
    
    async def download_set_cover_bytes(self, cover_url):
        """Scarica i bytes della copertina del set (Async)."""
        if not cover_url:
            return None
        
        try:
            # Usiamo get_bytes che gestisce i retry
            image_bytes = await self.get_bytes(cover_url)
            
            if image_bytes:
                self.log_callback(f"🖼️ Scaricati {len(image_bytes)} bytes per la cover.")
            
            return image_bytes
            
        except Exception as e:
            self.log_callback(f"⚠️ Errore download cover: {e}")
            return None
    
    # ================================================================
    # SALVATAGGIO DATABASE (Ottimizzato per Batch)
    # ================================================================
    
    def save_set_to_db(self, set_data):
        """Salva un set nel database (Sincrono)."""
        try:
            with self.db_lock:
                # ✅ MODIFICATO: Inserita cover_image_blob
                self.db_manager.cursor.execute("""
                    INSERT OR REPLACE INTO sets 
                    (set_code, set_name, url, release_date, total_cards, cover_image_path, cover_image_blob)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    set_data['code'],
                    set_data['name'],
                    set_data['url'],
                    set_data.get('release_date', ''),
                    set_data.get('total_cards', 0),
                    set_data.get('cover_path', None),
                    set_data.get('cover_image_blob', None) # <- NUOVO BLOB
                ))
                self.db_manager.conn.commit()
        except Exception as e:
            self.log_callback(f"❌ Errore salvataggio set: {e}")
    
    


    # In scraper.py

    # In scraper.py

    def _prepare_card_data_for_db(self, card_data: Dict) -> Optional[Tuple]:
        """
        Funzione Sincrona (in ThreadPool) per preparare il tuple.
        MODIFICATO: Aggiunto logging rumoroso per debug BLOB.
        """
        try:
            image_bytes = card_data.get('image_bytes')
            thumbnail_blob = None
            
            # ================================================================
            # ✅ ROBUSTEZZA PHASE 2 (Thumbnail 150x150 JPEG)
            # ================================================================
            
            # Controlla se i bytes sono validi (almeno 1KB)
            if image_bytes and len(image_bytes) > 1024:
                try:
                    img = Image.open(io.BytesIO(image_bytes))
                    
                    # Ridimensiona (150px mantenendo proporzioni)
                    img.thumbnail((250, 350), Image.Resampling.LANCZOS) 
                    
                    output = io.BytesIO()
                    
                    # Converti in RGB (JPEG non supporta trasparenza o P/L)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                        
                    img.save(output, format='JPEG', quality=85) # Salva come JPEG
                    thumbnail_blob = output.getvalue()
                
                except Exception as e:
                    # =======================================================
                    # ✅ LOGGING RUMOROSO
                    # =======================================================
                    err_msg = f"❌❌❌ FALLIMENTO CREAZIONE BLOB ❌❌❌: {e} | URL: {card_data.get('image_url')}"
                    print(err_msg)
                    self.log_callback(err_msg)
                    # =======================================================
            
            elif image_bytes:
                # Log se l'immagine è troppo piccola (probabilmente un errore)
                self.log_callback(f"⚠️ Immagine scartata (troppo piccola): {card_data.get('image_url')}")
            # ================================================================

            image_url_from_api = card_data.get('image_url')
            
            # Fallback per campi NOT NULL
            card_num = card_data.get('card_number', 'N/A') 
            set_code = card_data.get('set_code', 'N/A')
            card_name = card_data.get('card_name', 'Unknown Card')
            color_hash_from_json = card_data.get('color_hash')

            return (
                set_code,
                card_num,
                card_name,
                card_data.get('rarity', ''),
                image_url_from_api,
                image_url_from_api,
                card_data.get('card_url'),
                color_hash_from_json,
                thumbnail_blob
            )
        except Exception as e:
            self.log_callback(f"❌ Errore preparazione dati carta: {e}")
            return None

    def save_cards_to_db_batch(self, card_data_list: List[Dict]):
        """
        Salva una lista di carte nel database (batch).
        Include la creazione del BLOB (CPU-bound) in un ThreadPool.
        """
        if not card_data_list:
            return
            
        self.log_callback(f"⚙️ Preparazione batch di {len(card_data_list)} carte (resizing)...")
        
        tuples_to_insert = []
        
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [
                executor.submit(self._prepare_card_data_for_db, card) 
                for card in card_data_list
            ]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    tuples_to_insert.append(result)
        
        if not tuples_to_insert:
            self.log_callback("⚠️ Nessun dato valido da inserire nel batch.")
            return

        self.log_callback(f"💾 Salvataggio batch di {len(tuples_to_insert)} carte nel DB...")
        
        try:
            with self.db_lock:
                # ✅ MODIFICATO: Query SQL aggiornata
                self.db_manager.cursor.executemany("""
                    INSERT OR REPLACE INTO cards
                    (set_code, card_number, card_name, rarity, image_url, 
                    local_image_path, card_url, color_hash, thumbnail_blob)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, tuples_to_insert)
                self.db_manager.conn.commit()
            self.log_callback(f"✅ Batch di {len(tuples_to_insert)} carte salvato.")
        except Exception as e:
            self.log_callback(f"❌ Errore salvataggio batch carte: {e}")


    def _get_cropped_pil_image_for_hash(self, image_path):
        """
        Scala, ritaglia e ritorna una PIL image. (Sincrono, CPU-bound)
        IDENTICO a make_db.py per garantire compatibilità hash.
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            scaled_img = cv2.resize(
                img, (0, 0), 
                fx=TEMPLATE_DOWNSCALE_FACTOR, 
                fy=TEMPLATE_DOWNSCALE_FACTOR
            )
            
            x1, y1, x2, y2 = TEMPLATE_CROP_BOX
            scaled_crop_box = (
                int(x1 * TEMPLATE_DOWNSCALE_FACTOR),
                int(y1 * TEMPLATE_DOWNSCALE_FACTOR),
                int(x2 * TEMPLATE_DOWNSCALE_FACTOR),
                int(y2 * TEMPLATE_DOWNSCALE_FACTOR)
            )
            sc_x1, sc_y1, sc_x2, sc_y2 = scaled_crop_box
            
            cropped_img = scaled_img[sc_y1:sc_y2, sc_x1:sc_x2]
            
            return Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            
        except Exception as e:
            self.log_callback(f"⚠️ Errore processamento immagine hash: {e}")
            return None

    
    # ================================================================
    # ELABORAZIONE (Convertita ad async e batch)
    # ================================================================
    
    async def process_card(self, card_data: Dict) -> Optional[Dict]:
        """
        Elabora una singola carta (Async).
        MODIFICATO: Scarica i bytes dell'immagine per il BLOB.
        """
        try:
            # ❌ RIMOSSO: get_card_details (dati già presenti)
            
            # ✅ AGGIUNTO: Scarica i bytes dell'immagine
            if card_data.get('image_url'):
                try:
                    image_bytes = await self.get_bytes(card_data['image_url'])
                    card_data['image_bytes'] = image_bytes
                except Exception as e:
                    self.log_callback(f"⚠️ Errore download immagine {card_data['image_url']}: {e}")
                    card_data['image_bytes'] = None

            return card_data
            
        except Exception as e:
            self.log_callback(f"⚠️ Errore processing carta {card_data.get('card_number')}: {e}")
            return None

    def _log(self, message, level='info'):
        """Log selettivo - riduce spam console."""
        if level == 'error':
            self.log_callback(f"❌ {message}")
        elif level == 'warn':
            self.log_callback(f"⚠️ {message}")
        elif 'progress' in message.lower() or 'completat' in message.lower():
            self.log_callback(f"📊 {message}")


    async def process_set(self, set_data, download_images=True): # download_images ora è ignorato
        """
        Elabora un intero set (Async).
        MODIFICATO: Aggiunta logica di download del BLOB e controllo hash.
        """
        loop = asyncio.get_running_loop()
        set_code = set_data['code']
        
        try:
            # 1. Prendi la HASH/URL della cover ESISTENTE dal DB
            db_cover_url = None
            try:
                with self.db_lock:
                    cursor = self.db_manager.conn.cursor()
                    # Recuperiamo l'URL precedente per il check
                    cursor.execute("SELECT url FROM sets WHERE set_code = ?", (set_code,))
                    result = cursor.fetchone()
                    if result:
                        db_cover_url = result[0]
            except Exception as e:
                self.log_callback(f"⚠️ Errore lettura DB per cover {set_code}: {e}")

            cover_image_blob = None
            api_cover_url = set_data.get('cover_image_url')
            
            # 2. Controlla se la cover è cambiata (URL diverso) o mancante
            if api_cover_url:
                if api_cover_url != db_cover_url:
                    self.log_callback(f"🖼️ Cover set {set_code} cambiata/nuova. Download...")
                    cover_image_blob = await self.download_set_cover_bytes(api_cover_url)
                else:
                    # Se l'URL è lo stesso, proviamo a recuperare il BLOB esistente
                    # per assicurarci che non venga sovrascritto con NULL se non lo scarichiamo
                    # (Anche se INSERT OR REPLACE dovrebbe mantenere il BLOB esistente se non fornito)
                    try:
                        with self.db_lock:
                            cursor = self.db_manager.conn.cursor()
                            cursor.execute("SELECT cover_image_blob FROM sets WHERE set_code = ?", (set_code,))
                            result = cursor.fetchone()
                            if result and result[0]:
                                cover_image_blob = result[0]
                    except Exception as e:
                        self.log_callback(f"⚠️ Errore recupero BLOB esistente: {e}")


            # 3. Prepara i dati del SET per il salvataggio
            set_data_to_save = {
                'code': set_data.get('code'),
                'name': set_data.get('name'),
                'url': api_cover_url, # Usiamo l'URL della cover come 'hash' implicito
                'release_date': set_data.get('release_date'),
                'total_cards': set_data.get('total_cards'),
                'cover_path': set_data.get('cover_image_url'), 
                'cover_image_blob': cover_image_blob # <- PASSAGGIO DEL BLOB
            }
            # Salvataggio del SET
            await loop.run_in_executor(None, self.save_set_to_db, set_data_to_save)
            
            # ================================================================
            # INIZIO LOGICA CARTE
            # ================================================================

            # 4. Prendi le carte dall'API JSON
            cards_json = await self.get_cards_from_set(set_code)
            if not cards_json:
                self.log_callback(f"Nessuna carta trovata per set {set_code}")
                return 0

            # 5. Prendi gli hash delle carte ESISTENTI dal DB
            db_cards_map = {}
            try:
                with self.db_lock:
                    cursor = self.db_manager.conn.cursor()
                    cursor.execute("SELECT card_number, color_hash FROM cards WHERE set_code = ?", (set_code,))
                    for row in cursor.fetchall():
                        db_cards_map[row[0]] = row[1]
            except Exception as e:
                self.log_callback(f"⚠️ Errore lettura hash DB per {set_code}: {e}")

            # 6. Dividi le carte in "da processare" e "da saltare" (Smart Update)
            cards_to_process = [] # Nuove o modificate (hash diverso)
            cards_to_skip = []    # Identiche (hash uguale)

            for api_card in cards_json:
                api_card_num = api_card.get('card_number')
                api_hash = api_card.get('color_hash')
                
                db_hash = db_cards_map.get(api_card_num)
                
                if db_hash and db_hash == api_hash:
                    # Hash identico, salta questa carta
                    cards_to_skip.append(api_card)
                else:
                    # Carta nuova o hash diverso, processala
                    cards_to_process.append(api_card)

            if cards_to_skip:
                self.log_callback(f"ℹ️ Set {set_code}: Saltate {len(cards_to_skip)} carte (già aggiornate).")
            
            if not cards_to_process:
                self.log_callback(f"✅ Set {set_code} già sincronizzato.")
                return 0 # Nessuna carta da processare

            # 7. Scarica i bytes SOLO per le carte nuove/modificate
            self.log_callback(f"🖼️ Download di {len(cards_to_process)} immagini per {set_code}...")
            
            card_semaphore = asyncio.Semaphore(CARD_CONCURRENCY)
            
            async def process_card_wrapper(card_json):
                async with card_semaphore:
                    # ✅ AGGIUNGI set_code prima di processare
                    card_json['set_code'] = set_code 
                    return await self.process_card(card_json)

            tasks = [process_card_wrapper(card) for card in cards_to_process]
            processed_cards_list = await asyncio.gather(*tasks)
            
            cards_with_bytes = [card for card in processed_cards_list if card is not None]

            # 8. Salva SOLO le carte nuove/modificate nel DB
            if cards_with_bytes:
                await loop.run_in_executor(None, self.save_cards_to_db_batch, cards_with_bytes)
            
            return len(cards_with_bytes) # Ritorna solo il numero di carte *processate*
            
        except Exception as e:
            self.log_callback(f"Errore processing set {set_code}: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0    

    async def scrape_all_parallel(self):
        """Scarica tutti i set e le carte in parallelo (Async)."""
        try:
            await self.setup_session()
            sets = await self.get_sets()
            if not sets:
                self.log_callback("Nessun set trovato")
                await self.close_session()
                return

            total_sets = len(sets)
            completed_sets = 0
            total_cards = sum(set_data.get('totalcards', 0) for set_data in sets)
            completed_cards = 0
            
            # Progresso iniziale: 0%
            if self.progress_callback:
                self.progress_callback(0, total_sets)
            
            semaphore = asyncio.Semaphore(SET_CONCURRENCY)
            
            async def process_setwrapper(set_data):
                async with semaphore:
                    try:
                        result = await self.process_set(set_data, True)
                        return result
                    except Exception as e:
                        return e
            
            # Avvia task per tutti i set
            tasks = [process_setwrapper(set_data) for set_data in sets]
            completed_task_count = 0
            
            # Elaborazione parallela con aggiornamenti progress
            for future in asyncio.as_completed(tasks):
                result = await future
                completed_sets += 1
                completed_task_count += 1
                
                if isinstance(result, int) and result > 0:
                    completed_cards += result
                elif isinstance(result, Exception):
                    self.log_callback(f"Errore set: {str(result)}")  # Solo errori gravi
                
                # Aggiornamento progress: set + stima carte
                set_progress = completed_sets / total_sets * 70  # 70% per set
                card_progress = (completed_cards / total_cards * 30) if total_cards > 0 else 0
                overall_progress = int(set_progress + card_progress)
                
                status = f"Set {completed_sets}/{total_sets} | Carte {completed_cards}/{total_cards}"
                
                if self.progress_callback:
                    self.progress_callback(completed_sets, total_sets)

                
                # Log ridotti - solo milestone
                if completed_sets % 3 == 0 or completed_sets == total_sets:
                    self.log_callback(f"Progresso: {overall_progress}% - {status}")
            
            await self.close_session()
            
            # Progresso finale: 100%
            if self.progress_callback:
                self.progress_callback(total_sets, total_sets)
            
            self.log_callback(f"Scraping completato: {total_sets} set, {total_cards} carte elaborate")
            
        except Exception as e:
            self.log_callback(f"Errore scraping: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            if self.session and not self.session.closed:
                await self.close_session()
                
            
#    def get_stats(self):
#        """Ritorna statistiche del database (Sincrono)."""
#        try:
#            with self.db_lock:
#                self.db_manager.cursor.execute("SELECT COUNT(*) FROM sets")
#                num_sets = self.db_manager.cursor.fetchone()[0]
#                
#                self.db_manager.cursor.execute("SELECT COUNT(*) FROM cards")
#                num_cards = self.db_manager.cursor.fetchone()[0]
#                
#                return {
#                    'sets': num_sets,
#                    'cards': num_cards,
#                    'proxies': len(self.proxies) if self.use_proxy else 0
#                }
#        except Exception as e:
#            self.log_callback(f"❌ Errore statistiche: {e}")
#            return {}
    
    def close(self):
        """Chiude la connessione al database (Sincrono)."""
        self.log_callback("Database connection closed.")
        self.db_manager.close()

# =========================================================================