
# ================================================================
# FILE: sync_shiny_dust.py (COMPLETO)
# ================================================================

import sqlite3
import csv
import os
import json
from pathlib import Path


from config import DB_FILENAME, SETTINGS_FILE




def get_bot_path_from_settings():
    """Legge il percorso del bot da settings.json"""
    if not os.path.exists(SETTINGS_FILE):
        print(f"❌ Errore: {SETTINGS_FILE} non trovato")
        return None
    
    try:
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            settings = json.load(f)
            bot_path = settings.get('bot_folder') or settings.get('botFolder') or settings.get('BOT_FOLDER')
            if bot_path and os.path.isdir(bot_path):
                return bot_path
            else:
                print(f"❌ Percorso bot non valido: {bot_path}")
                return None
    except Exception as e:
        print(f"❌ Errore lettura settings.json: {e}")
        return None


def find_csv_recursive(bot_path: str):
    """
    ✅ Cerca il file Trades_Database.csv RICORSIVAMENTE
    in tutte le cartelle e sottocartelle
    """
    if not bot_path or not os.path.isdir(bot_path):
        print(f"❌ Percorso bot non valido: {bot_path}")
        return None
    
    # Cerca ricorsivamente con Path.rglob()
    for csv_file in Path(bot_path).rglob("Trades_Database.csv"):
        print(f"✅ CSV trovato: {csv_file}")
        return str(csv_file)
    
    # Se non trovato, stampa le cartelle cercate
    print(f"❌ CSV non trovato in: {bot_path}")
    print(f"   (cercato ricorsivamente in tutte le sottocartelle)")
    return None


def sync_shiny_dust_from_csv(csv_path: str, db_path: str):
    """
    ✅ Sincronizza shiny_dust e hourglasses dal CSV al database
    Aggiorna il database con i valori dal CSV
    """
    if not os.path.exists(csv_path):
        return {
            'updated': 0, 
            'total': 0, 
            'not_found': 0, 
            'errors': [f"CSV non trovato: {csv_path}"]
        }
    
    if not os.path.exists(db_path):
        return {
            'updated': 0, 
            'total': 0, 
            'not_found': 0, 
            'errors': [f"Database non trovato: {db_path}"]
        }
    
    result = {
        'updated': 0,
        'total': 0,
        'not_found': 0,
        'errors': []
    }
    
    try:
        # ================================================================
        # STEP 1: Leggi il CSV in memoria
        # ================================================================
        csv_data = {}  # device_account → {shiny_dust, hourglasses}
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                device_account = row.get('DeviceAccount', '').strip()
                shiny_dust = row.get('Shinedust', '').strip()
                hourglasses = row.get('Hourglasses', '').strip()
                
                if device_account:
                    try:
                        dust_value = int(shiny_dust) if shiny_dust else 0
                        hourglasses_value = int(hourglasses) if hourglasses else 0
                        
                        csv_data[device_account] = {
                            'shiny_dust': dust_value,
                            'hourglasses': hourglasses_value
                        }
                    except ValueError:
                        pass
        
        result['total'] = len(csv_data)
        
        # ================================================================
        # STEP 2: Aggiorna il database
        # ================================================================
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        for device_account, values in csv_data.items():
            try:
                shiny_dust = values['shiny_dust']
                hourglasses = values['hourglasses']
                
                # ✅ IMPORTANTE: UPDATE con WHERE device_account
                cursor.execute("""
                    UPDATE accounts 
                    SET shiny_dust = ?, hourglasses = ?
                    WHERE device_account = ?
                """, (shiny_dust, hourglasses, device_account))
                
                if cursor.rowcount > 0:
                    result['updated'] += 1
                else:
                    # ✅ Account non trovato nel DB - ignora
                    result['not_found'] += 1
            except Exception as e:
                result['errors'].append(f"Errore UPDATE {device_account}: {e}")
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        result['errors'].append(f"Errore generale: {e}")
    
    return result


def sync_shiny_dust():
    """
    ✅ Funzione principale - Sincronizza shiny_dust dal CSV al database
    SEMPRE ritorna dict
    """
    print("\n" + "="*80)
    print("🔄 SINCRONIZZAZIONE SHINY DUST")
    print("="*80)
    
    result = {
        'updated': 0,
        'total': 0,
        'not_found': 0,
        'errors': []
    }
    
    # Step 1: Leggi percorso bot
    print("\n1️⃣ Lettura percorso bot da settings.json...")
    bot_path = get_bot_path_from_settings()
    if not bot_path:
        print("❌ Impossibile leggere il percorso del bot")
        return result
    
    print(f"✅ Percorso bot: {bot_path}")
    
    # Step 2: Trova CSV ricorsivamente
    print("\n2️⃣ Ricerca CSV in tutte le cartelle...")
    csv_path = find_csv_recursive(bot_path)
    if not csv_path:
        print("❌ CSV non trovato")
        return result
    
    # Step 3: Sincronizza
    print("\n3️⃣ Sincronizzazione shiny_dust...")
    sync_result = sync_shiny_dust_from_csv(csv_path, DB_FILENAME)
    
    # Unisci i risultati
    result['updated'] = sync_result.get('updated', 0)
    result['total'] = sync_result.get('total', 0)
    result['not_found'] = sync_result.get('not_found', 0)
    result['errors'].extend(sync_result.get('errors', []))
    
    # Stampa risultati
    print(f"\n📊 RISULTATI:")
    print(f"  Totale nel CSV: {result['total']}")
    print(f"  Aggiornati nel DB: {result['updated']}")
    if result['not_found'] > 0:
        print(f"  ⚠️ Non trovati nel DB: {result['not_found']}")
    
    if result['errors']:
        print(f"\n❌ ERRORI ({len(result['errors'])}):")
        for error in result['errors'][:5]:
            print(f"  - {error}")
        if len(result['errors']) > 5:
            print(f"  ... e altri {len(result['errors']) - 5} errori")
    else:
        print(f"\n✅ Sincronizzazione completata con successo!")
    
    print("\n" + "="*80)
    
    # ✅ SEMPRE ritorna dict
    return result


if __name__ == "__main__":
    result = sync_shiny_dust()