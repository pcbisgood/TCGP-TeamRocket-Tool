# ============================================================================
# ??? DATABASE VALIDATION & MIGRATION SYSTEM
# ============================================================================
"""database.py - Gestione database SQLite e validazione"""
import sqlite3
import os
from datetime import datetime
from threading import Lock

from config import DB_FILENAME, TABLES_SCHEMA, get_app_data_path




# ============================================================================
# 🛡️ DATABASE VALIDATION & MIGRATION SYSTEM
# ============================================================================

class DatabaseValidator:
    """Valida e ripara il database all'avvio."""
    
    # Definizione schema corretto
    TABLES_SCHEMA = {
        'sets': [
            ('set_code', 'TEXT PRIMARY KEY'),
            ('set_name', 'TEXT NOT NULL'),
            ('cover_image_path', 'TEXT'),
            ('cover_image_blob', 'BLOB'),
            ('release_date', 'TEXT'),
            ('total_cards', 'INTEGER DEFAULT 0'),
            ('url', 'TEXT'),
            ('created_at', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),
        ],
        'cards': [
            ('id', 'INTEGER PRIMARY KEY AUTOINCREMENT'),
            ('set_code', 'TEXT NOT NULL'),
            ('card_number', 'TEXT NOT NULL'),
            ('card_name', 'TEXT NOT NULL'),
            ('rarity', 'TEXT'),
            ('image_url', 'TEXT'),
            ('local_image_path', 'TEXT'),
            ('card_url', 'TEXT'),
            ('color_hash', 'TEXT'),
            ('thumbnail_blob', 'BLOB'),
            ('created_at', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),
        ],
        # database.py (dentro TABLES_SCHEMA)

        'accounts': [
            ('device_account', 'TEXT PRIMARY KEY'),        # <-- ✅ NUOVA CHIAVE PRIMARIA
            ('account_name', 'TEXT NOT NULL'),             # <-- Non più UNIQUE. Nome di visualizzazione/Fallback
            ('alias', 'TEXT'),
            ('shiny_dust', 'INTEGER DEFAULT 0'),
            ('hourglasses', 'INTEGER DEFAULT 0'),
            ('discord_user_id', 'TEXT'),
            ('device_password', 'TEXT'),
            ('created_at', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),
            ('last_updated', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),
        ],
        'account_inventory': [
            ('id', 'INTEGER PRIMARY KEY AUTOINCREMENT'),
            ('account_id', 'TEXT NOT NULL'),   # <-- ✅ MODIFICATO: TEXT
            ('card_id', 'INTEGER NOT NULL'),
            ('quantity', 'INTEGER DEFAULT 1'),
            ('acquisition_date', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),
            ('UNIQUE (account_id, card_id)', '')
        ],
        'found_cards': [
            ('id', 'INTEGER PRIMARY KEY AUTOINCREMENT'),
            ('card_id', 'INTEGER NOT NULL'),
            ('account_id', 'TEXT'),            # <-- ✅ CORRETTO: Deve essere TEXT
            ('message_id', 'TEXT'),
            ('channel_id', 'TEXT'),
            ('user_id', 'TEXT'),
            ('found_at', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),
            ('confidence_score', 'REAL DEFAULT 1.0'),
            ('source_image_path', 'TEXT'),
        ],
        'wishlist': [
            ('id', 'INTEGER PRIMARY KEY AUTOINCREMENT'),
            ('card_id', 'INTEGER NOT NULL UNIQUE'), # <-- MODIFICATO
            ('added_at', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),
            ('added_date', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),
            ('priority', 'INTEGER DEFAULT 0'),
        ],
        'trades': [
            ('message_id', 'TEXT PRIMARY KEY'),
            ('account_id', 'TEXT'),
            ('account_name', 'TEXT'),
            ('xml_path', 'TEXT'),
            ('image_url', 'TEXT'),
            ('screenshot_thumbnail_blob', 'BLOB'),
            ('message_link', 'TEXT'),
            ('cards_found_text', 'TEXT'),
            ('processed_at', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),
            ('scan_status', 'INTEGER DEFAULT 0'), 
            ('scan_results_json', 'TEXT')
        ],
        'manual_trades': [
            ('id', 'INTEGER PRIMARY KEY AUTOINCREMENT'),
            ('account_id', 'TEXT NOT NULL'), # <-- ✅ CORRETTO: Deve essere TEXT
            ('card_id', 'INTEGER NOT NULL'),
            ('trade_date', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),
            ('quantity_traded', 'INTEGER NOT NULL DEFAULT 1'),
            ('FOREIGN KEY(account_id) REFERENCES accounts(device_account)', ''), # <-- ✅ CRUCIALE: Riferisce device_account
            ('FOREIGN KEY(card_id) REFERENCES cards(id)', '')
        ],
    }
    
    def __init__(self, db_filename, log_callback=None):
        self.db_filename = db_filename
        self.log_callback = log_callback or print
        self.conn = None
    
    # In database.py (dentro la classe DatabaseManager)

    
    def validate_and_repair(self):
        """Valida TUTTE le tabelle e colonne, ripara se necessario."""
        if not self.conn:
            self.connect()
        
        try:
            cursor = self.conn.cursor()
            
            missing_tables = []
            
            for table_name in self.TABLES_SCHEMA.keys():
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table_name,)
                )
                
            
            # STEP 2: Se tabelle mancanti, ricreale TUTTE
            if missing_tables:
                self._create_all_tables(cursor)
            
            # STEP 3: Controlla colonne di ogni tabella
            columns_added = 0
            
            for table_name, expected_cols in self.TABLES_SCHEMA.items():
                # Ottieni colonne attuali
                cursor.execute(f"PRAGMA table_info({table_name})")
                existing_cols = {row[1]: row[2] for row in cursor.fetchall()}
                
                # Controlla colonne mancanti
                for col_name, col_type in expected_cols:
                    # ✅ CORREZIONE: Ignora i vincoli (come UNIQUE) che non hanno un tipo
                    if col_type and col_name not in existing_cols:
                        self._add_column(cursor, table_name, col_name, col_type)
                        columns_added += 1
            
            cursor.execute("PRAGMA foreign_key_check")
            orphans = cursor.fetchall()
            
            
            # STEP 5: Verifica indici
            required_indexes = [
                ('idx_cards_set', 'cards', 'set_code', ''),
                ('idx_cards_rarity', 'cards', 'rarity', ''),
                ('idx_inventory_account', 'account_inventory', 'account_id', ''),
                ('idx_found_cards_card', 'found_cards', 'card_id', ''),
                # ✅ AGGIUNTO: Vincolo UNIQUE per l'inventario
                ('idx_inventory_account_card_unique', 'account_inventory', '(account_id, card_id)', 'UNIQUE')
            ]
            
            for idx_name, table, column, *extra in required_indexes:
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND name=?",
                    (idx_name,)
                )
                if not cursor.fetchone():
                    try:
                        unique_str = "UNIQUE" if extra and extra[0] == 'UNIQUE' else ""
                        columns_str = column if column.startswith('(') else f"({column})"
                        cursor.execute(f"CREATE {unique_str} INDEX IF NOT EXISTS {idx_name} ON {table}{columns_str}")
                    except Exception as e:
                        self.log_callback(f"   ⚠️ Errore creazione indice {idx_name}: {e}")
            
            for idx_name, table, column in required_indexes:
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND name=?",
                    (idx_name,)
                )
                if not cursor.fetchone():
                    cursor.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table}({column})")
            
            # Salva modifiche
            self.conn.commit()
            
            
            return True
        
        except Exception as e:
            self.log_callback(f"❌ Errore validazione: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_all_tables(self, cursor):
        """Crea TUTTE le tabelle da zero."""
        for table_name, columns in self.TABLES_SCHEMA.items():
            # Filtra le definizioni di colonna (hanno un tipo)
            col_list = [f"{col} {dtype}" for col, dtype in columns if dtype]
            # Filtra i vincoli di tabella (non hanno un tipo)
            constraint_list = [col for col, dtype in columns if not dtype]
            
            # Unisci le due liste
            all_defs = col_list + constraint_list
            col_defs_str = ', '.join(all_defs)
            
            create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({col_defs_str})"
            
            try:
                cursor.execute(create_sql)
            except Exception as e:
                self.log_callback(f"   ❌ Errore creazione '{table_name}': {e}")
    
    def _add_column(self, cursor, table, column, col_type):
        """Aggiungi una colonna mancante."""
        try:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        except Exception as e:
            self.log_callback(f"   ❌ Errore: {e}")
    
    def close(self):
        """Chiudi la connessione."""
        if self.conn:
            self.conn.close()




# =========================================================================
# 🗄️ DATABASE MANAGEMENT CON MIGRAZIONE
# =========================================================================

class DatabaseManager:
    """Gestisce il database SQLite con tutte le tabelle necessarie."""
    
    def __init__(self, db_filename=DB_FILENAME, log_callback=None):
        self.db_filename = db_filename
        self.log_callback = log_callback or print
        self.conn = None
        self.cursor = None
        self.db_lock = Lock()

    # ✅ Schema corretto
    # ✅ Schema corretto
    TABLES_SCHEMA = {
        'sets': [
            ('set_code', 'TEXT PRIMARY KEY'),
            ('set_name', 'TEXT NOT NULL'),
            ('cover_image_path', 'TEXT'),
            ('cover_image_blob', 'BLOB'),
            ('release_date', 'TEXT'),
            ('total_cards', 'INTEGER DEFAULT 0'),
            ('url', 'TEXT'),
            ('created_at', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),
        ],
        'cards': [
            ('id', 'INTEGER PRIMARY KEY AUTOINCREMENT'),
            ('set_code', 'TEXT NOT NULL'),
            ('card_number', 'TEXT NOT NULL'),
            ('card_name', 'TEXT NOT NULL'),
            ('rarity', 'TEXT'),
            ('image_url', 'TEXT'),
            ('local_image_path', 'TEXT'),
            ('card_url', 'TEXT'),
            ('color_hash', 'TEXT'),
            ('thumbnail_blob', 'BLOB'),
            ('created_at', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),
        ],
        # database.py (dentro TABLES_SCHEMA)

        'accounts': [
            ('device_account', 'TEXT PRIMARY KEY'),        # <-- ✅ NUOVA CHIAVE PRIMARIA
            ('account_name', 'TEXT NOT NULL'),             # <-- Non più UNIQUE. Nome di visualizzazione/Fallback
            ('alias', 'TEXT'),
            ('shiny_dust', 'INTEGER DEFAULT 0'),
            ('hourglasses', 'INTEGER DEFAULT 0'),
            ('discord_user_id', 'TEXT'),
            ('device_password', 'TEXT'),
            ('created_at', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),
            ('last_updated', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),
        ],
        'account_inventory': [
            ('id', 'INTEGER PRIMARY KEY AUTOINCREMENT'),
            ('account_id', 'TEXT NOT NULL'),   # <-- ✅ MODIFICATO: TEXT
            ('card_id', 'INTEGER NOT NULL'),
            ('quantity', 'INTEGER DEFAULT 1'),
            ('acquisition_date', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),
            ('UNIQUE (account_id, card_id)', '')
        ],
        'found_cards': [
            ('id', 'INTEGER PRIMARY KEY AUTOINCREMENT'),
            ('card_id', 'INTEGER NOT NULL'),
            ('account_id', 'TEXT'),            # <-- ✅ CORRETTO: Deve essere TEXT
            ('message_id', 'TEXT'),
            ('channel_id', 'TEXT'),
            ('user_id', 'TEXT'),
            ('found_at', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),
            ('confidence_score', 'REAL DEFAULT 1.0'),
            ('source_image_path', 'TEXT'),
        ],
        'wishlist': [
            ('id', 'INTEGER PRIMARY KEY AUTOINCREMENT'),
            ('card_id', 'INTEGER NOT NULL UNIQUE'), # <-- MODIFICATO
            ('added_at', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),
            ('added_date', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),
            ('priority', 'INTEGER DEFAULT 0'),
        ],
        'trades': [ # <-- TABELLA MANCANTE AGGIUNTA
            ('message_id', 'TEXT PRIMARY KEY'),
            ('account_id', 'TEXT'),
            ('account_name', 'TEXT'),
            ('xml_path', 'TEXT'),
            ('image_url', 'TEXT'),
            ('screenshot_thumbnail_blob', 'BLOB'),
            ('message_link', 'TEXT'),
            ('cards_found_text', 'TEXT'),
            ('processed_at', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),
            ('scan_status', 'INTEGER DEFAULT 0'), 
            ('scan_results_json', 'TEXT')
        ],
        'manual_trades': [
            ('id', 'INTEGER PRIMARY KEY AUTOINCREMENT'),
            ('account_id', 'TEXT NOT NULL'), # <-- ✅ CORRETTO: Deve essere TEXT
            ('card_id', 'INTEGER NOT NULL'),
            ('trade_date', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),
            ('quantity_traded', 'INTEGER NOT NULL DEFAULT 1'),
            ('FOREIGN KEY(account_id) REFERENCES accounts(device_account)', ''), # <-- ✅ CRUCIALE: Riferisce device_account
            ('FOREIGN KEY(card_id) REFERENCES cards(id)', '')
        ],
    }
    


    def get_wishlist(self) -> set:
            """
            Recupera tutti i card_id dalla wishlist.
            Restituisce un set per un controllo rapido (es. `card_id in set`).
            """
            try:
                self.cursor.execute("SELECT card_id FROM wishlist")
                # Usa una set comprehension per efficienza
                return {row[0] for row in self.cursor.fetchall()}
            except Exception as e:
                self.log_callback(f"❌ Errore get_wishlist: {e}")
                return set()

# database.py (Aggiungi questo metodo alla tua classe DatabaseManager)

    def toggle_wishlist_status(self, card_id: int) -> bool:
        """
        Aggiunge o rimuove una carta dalla wishlist universale. 
        Restituisce True se la carta è ora nella wishlist, False altrimenti.
        """
        if card_id is None:
            return False

        card_id = int(card_id)

        try:
            # 1. Controlla se la carta è già nella wishlist
            self.cursor.execute(
                "SELECT id FROM wishlist WHERE card_id = ?", 
                (card_id,)
            )
            exists = self.cursor.fetchone()
            
            if exists:
                # 2. Se esiste, rimuovi (TOGGLE OFF)
                self.cursor.execute(
                    "DELETE FROM wishlist WHERE card_id = ?", 
                    (card_id,)
                )
                self.conn.commit()
                return False
            else:
                # 3. Se non esiste, aggiungi (TOGGLE ON)
                self.cursor.execute(
                    "INSERT INTO wishlist (card_id) VALUES (?)", 
                    (card_id,) 
                )
                self.conn.commit()
                return True
                
        except Exception as e:
            self.log_callback(f"Errore toggle wishlist: {e}") 
            self.conn.rollback()
            # Se 'exists' è None, c'era un errore, ma lo stato non è cambiato
            # Se 'exists' non è None, c'era un errore, ma lo stato non è cambiato
            return exists is not None # Ritorna lo stato pre-operazione in caso di errore

    def connect(self):
        """
        Connetti al database.
        MODIFICATO: Aggiunte ottimizzazioni PRAGMA (Phase 1).
        """
        try:
            self.conn = sqlite3.connect(self.db_filename, check_same_thread=False)
            
            # ✅ OTTIMIZZAZIONI PRAGMA (Phase 1)
            self.conn.execute("PRAGMA journal_mode = WAL")
            self.conn.execute("PRAGMA synchronous = NORMAL")
            self.conn.execute("PRAGMA cache_size = -64000")    # 64MB
            self.conn.execute("PRAGMA mmap_size = 30000000")   # 30MB
            self.conn.execute("PRAGMA temp_store = MEMORY")
            
            self.conn.execute("PRAGMA foreign_keys = ON")
            self.cursor = self.conn.cursor()
            return True
        except Exception as e:
            self.log_callback(f"❌ Errore connessione DB: {e}")
            return False

    def set_inventory_quantity(self, account_id: int, card_id: int, new_quantity: int) -> bool:
            """
            Imposta la quantità esatta di una carta per un account.
            Se la quantità è <= 0, la voce viene eliminata.
            Altrimenti, esegue un UPSERT (Insert/Update).
            """
            if account_id is None or card_id is None:
                return False
                
            try:
                if new_quantity <= 0:
                    # Quantità 0 o negativa: Rimuovi la voce
                    self.cursor.execute("""
                        DELETE FROM account_inventory
                        WHERE account_id = ? AND card_id = ?
                    """, (account_id, card_id))
                else:
                    # Quantità positiva: Inserisci o Aggiorna (UPSERT)
                    self.cursor.execute("""
                        INSERT INTO account_inventory (account_id, card_id, quantity)
                        VALUES (?, ?, ?)
                        ON CONFLICT(account_id, card_id) 
                        DO UPDATE SET quantity = excluded.quantity
                    """, (account_id, card_id, new_quantity))
                
                self.conn.commit()
                return True
                
            except Exception as e:
                self.log_callback(f"❌ Errore set_inventory_quantity: {e}")
                self.conn.rollback()
                return False
    
    def validate_and_repair_database(self):
        """✅ NUOVO: Valida TUTTE le tabelle e colonne."""
        if not self.conn:
            self.connect()
        
        try:
            cursor = self.conn.cursor()
            
            missing_tables = []
            
            for table_name in self.TABLES_SCHEMA.keys():
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table_name,)
                )
                
                if not cursor.fetchone():
                    missing_tables.append(table_name)
            
            # Se tabelle mancanti, ricreale TUTTE
            if missing_tables:
                self.setup_database()
            
            # STEP 2: Controlla colonne di ogni tabella
            columns_added = 0
            
            for table_name, expected_cols in self.TABLES_SCHEMA.items():
                cursor.execute(f"PRAGMA table_info({table_name})")
                existing_cols = {row[1]: row[2] for row in cursor.fetchall()}
                
                for col_name, col_type in expected_cols:
                    # Ignora i vincoli (come UNIQUE) che non hanno un tipo
                    for col_name, col_type in expected_cols:
                    # ✅ CORREZIONE: Ignora i vincoli (come UNIQUE) che non hanno un tipo
                        if col_type and col_name not in existing_cols:
                            try:
                                cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}")
                                columns_added += 1
                            except Exception as e:
                                self.log_callback(f"   ⚠️ Errore: {e}")

            
            # STEP 3: Crea indici
            required_indexes = [
                ('idx_cards_set', 'cards', 'set_code', ''),
                ('idx_cards_rarity', 'cards', 'rarity', ''),
                ('idx_inventory_account', 'account_inventory', 'account_id', ''),
                ('idx_found_cards_card', 'found_cards', 'card_id', ''),
                # ✅ AGGIUNTO: Vincolo UNIQUE per l'inventario
                ('idx_inventory_account_card_unique', 'account_inventory', '(account_id, card_id)', 'UNIQUE')
            ]
            
            for idx_name, table, column, *extra in required_indexes:
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND name=?",
                    (idx_name,)
                )
                if not cursor.fetchone():
                    try:
                        unique_str = "UNIQUE" if extra and extra[0] == 'UNIQUE' else ""
                        columns_str = column if column.startswith('(') else f"({column})"
                        cursor.execute(f"CREATE {unique_str} INDEX IF NOT EXISTS {idx_name} ON {table}{columns_str}")
                        self.log_callback(f"   ✅ Creato indice {idx_name} su {table}.")
                    except Exception as e:
                        if "already exists" not in str(e):
                            self.log_callback(f"   ⚠️ Errore creazione indice {idx_name}: {e}")
            
            self.conn.commit()
            
            
            return True
        
        except Exception as e:
            return False
    
    def setup_database(self):
        """Crea tutte le tabelle da zero."""
        if not self.conn:
            self.connect()
        
        try:
            for table_name, columns in self.TABLES_SCHEMA.items():
                # Filtra le definizioni di colonna (hanno un tipo)
                col_list = [f"{col} {dtype}" for col, dtype in columns if dtype]
                # Filtra i vincoli di tabella (non hanno un tipo)
                constraint_list = [col for col, dtype in columns if not dtype]
                
                # Unisci le due liste
                all_defs = col_list + constraint_list
                col_defs_str = ', '.join(all_defs)

                create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({col_defs_str})"
                
                self.cursor.execute(create_sql)
            
            self.conn.commit()
        
        except Exception as e:
            self.log_callback(f"❌ Errore setup_database: {e}") # Log esteso
            self.conn.rollback()
    
    
    def close(self):
        """Chiude la connessione al database."""
        if self.conn:
            self.conn.close()
