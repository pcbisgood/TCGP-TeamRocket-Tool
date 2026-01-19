"""flask_server.py - Server Flask per web interface"""

# Import standard library
import secrets
import mimetypes
from flask import Flask, render_template, send_file, jsonify, request, send_from_directory, session

import sqlite3
import os
from datetime import datetime
from threading import Thread, Event
from typing import Optional, Dict, List
import json
from functools import wraps
from dotenv import load_dotenv
from PyQt5.QtCore import QThread, pyqtSignal
from flask import Blueprint, render_template, jsonify
from .database import DatabaseManager
from collections import defaultdict
from .stats_endpoints import stats_bp
# Crea un Blueprint per le statistiche
stats_bp = Blueprint('stats', __name__, url_prefix='/stats')

# Import configurazione
from config import (
    TCG_IMAGES_DIR, 
    ICON_PATH, 
    DB_FILENAME, 
    ACCOUNTS_DIR,
    FLASK_HOST, 
    FLASK_PORT, 
    FLASK_DEBUG, 
    CLOUDFLARE_PASSWORD,
    get_resource_path,
    SELECTED_RARITIES
)

# Import traduzioni
from .translations import t, set_language, get_language
class FlaskServerThread(QThread):
    """Thread per eseguire il server Flask in background senza bloccare la GUI."""
    
    log_signal = pyqtSignal(str)
    started_signal = pyqtSignal()
    stopped_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.flask_app = None
        self.server = None
        self.should_stop = False
        
    def run(self):
        """Avvia il server Flask in un thread separato."""
        try:
            from flask import Flask, render_template, jsonify, send_from_directory
            from werkzeug.serving import make_server
            # stats_bp is already created at module level above, do not re-import flask_stats_routes

            # Crea Flask app
            app = Flask(__name__, 
                       template_folder='templates',
                       static_folder='static')
            app.register_blueprint(stats_bp)            

            app.static_folder = '.'
            app.static_url_path = '/static'
            # ✅ IMPORTANTE: Configura la chiave segreta per le sessioni
            app.config['SECRET_KEY'] = secrets.token_hex(32)

            # Configurazioni aggiuntive
            app.config['SESSION_COOKIE_SECURE'] = False  # True se usi HTTPS
            app.config['SESSION_COOKIE_HTTPONLY'] = True
            app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
            app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 ora
            
            # ✅ Configura traduzioni per Flask
            # Context processor per passare t() ai template
            @app.context_processor
            def inject_translations():
                # Carica la lingua depuis settings
                try:
                    if os.path.exists('settings.json'):
                        with open('settings.json', 'r', encoding='utf-8') as f:
                            settings = json.load(f)
                            saved_language = settings.get('language', 'fr')
                            set_language(saved_language)
                    else:
                        set_language('fr')
                except:
                    set_language('fr')
                
                # Passe la fonction t() aux templates
                return dict(t=t, current_language=get_language())



         
            self.flask_app = app
            
            # ===== ROUTES  ====
            @app.route('/tcg_images/<path:image_path>')
            def serve_tcg_images(image_path):
                """Serve immagini TCG - funziona con EXE."""
                try:
                    import urllib.parse
                    
                    # Decodifica il path
                    decoded_path = urllib.parse.unquote(image_path)
                    
                    # Costruisci path completo
                    full_path = os.path.join(TCG_IMAGES_DIR, decoded_path)
                    
                    # Normalizza il path (rimuovi .. e simili)
                    full_path = os.path.abspath(full_path)
                    base_dir = os.path.abspath(TCG_IMAGES_DIR)
                    
                    # Verifica che il file sia dentro TCG_IMAGES_DIR (security check)
                    if not full_path.startswith(base_dir):
                        return "Access denied", 403
                    
                    print(f"📁 Requested: {decoded_path}")
                    print(f"📁 Full path: {full_path}")
                    print(f"✅ Exists: {os.path.exists(full_path)}")
                    
                    if not os.path.exists(full_path):
                        return "Not found", 404
                    
                    # Determina il tipo MIME
                    mimetype, _ = mimetypes.guess_type(full_path)
                    if mimetype is None:
                        mimetype = 'image/webp'
                    
                    print(f"✅ Sending: {mimetype}")
                    return send_file(full_path, mimetype=mimetype)
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                    return f"Error: {str(e)}", 500
                
            @app.route('/debug/images')
            def debug_images():
                """Debug: mostra il percorso delle immagini."""
                import os
                
                html = f"""
                <h1>Debug Images</h1>
                <p><strong>TCG_IMAGES_DIR:</strong> {TCG_IMAGES_DIR}</p>
                <p><strong>Exists:</strong> {os.path.exists(TCG_IMAGES_DIR)}</p>
                <hr>
                <h2>Available images:</h2>
                <ul>
                """
                
                if os.path.exists(TCG_IMAGES_DIR):
                    for root, dirs, files in os.walk(TCG_IMAGES_DIR):
                        for file in files:
                            if file.endswith(('.webp', '.png', '.jpg')):
                                full_path = os.path.join(root, file)
                                rel_path = os.path.relpath(full_path, TCG_IMAGES_DIR)
                                
                                # Converti backslash a forward slash
                                url_path = rel_path.replace('\\', '/')
                                
                                html += f"""
                                <li>
                                    <strong>{file}</strong><br>
                                    Full: {full_path}<br>
                                    Rel: {rel_path}<br>
                                    URL: <a href="/tcg_images/{url_path}">Test</a>
                                </li>
                                """
                
                html += "</ul>"
                return html




            @app.route('/account/<account_name>', methods=['GET', 'POST'])
            @require_password
            def account_collection(account_name):
                """Visualizza la collezione completa di un account specifico."""
                try:
                    import base64
                    conn = sqlite3.connect(DB_FILENAME)
                    cursor = conn.cursor()
                    
                    # Verifica che l'account esista + get shiny_dust e hourglasses
                    cursor.execute("""
                        SELECT device_account, account_name, alias, shiny_dust, hourglasses
                        FROM accounts 
                        WHERE account_name = ?
                    """, (account_name,))

                    account = cursor.fetchone()
                    
                    if not account:
                        conn.close()
                        return f"<h1>Account '{account_name}' not found</h1><a href='/'>Back to home</a>", 404
                    
                    account_id, account_name, alias, shiny_dust, hourglasses = account
                    display_name = alias if alias else account_name
                    
                    # Get statistiche account
                    cursor.execute("""
                        SELECT 
                            COUNT(DISTINCT c.id) as unique_cards,
                            SUM(ai.quantity) as total_copies,
                            COUNT(DISTINCT c.set_code) as sets_owned
                        FROM cards c
                        JOIN account_inventory ai ON c.id = ai.card_id
                        WHERE ai.account_id = ? AND ai.quantity > 0
                    """, (account_id,))
                    
                    stats = cursor.fetchone()
                    
                    # Get collezione per set con cover_blob
                    cursor.execute("""
                        SELECT DISTINCT c.set_code, s.set_name, s.cover_image_blob
                        FROM cards c
                        JOIN sets s ON c.set_code = s.set_code
                        JOIN account_inventory ai ON c.id = ai.card_id
                        WHERE ai.account_id = ? AND ai.quantity > 0
                        ORDER BY s.set_code
                    """, (account_id,))
                    
                    sets = cursor.fetchall()
                    
                    # Get carte per ogni set con thumbnail_blob
                    collection_by_set = {}
                    for set_code, set_name, cover_blob in sets:
                        # Converti cover blob in base64
                        cover_blob_b64 = None
                        if cover_blob:
                            cover_blob_b64 = base64.b64encode(cover_blob).decode('utf-8')
                        
                        cursor.execute("""
                            SELECT c.id, c.card_number, c.card_name, c.rarity, c.thumbnail_blob, ai.quantity
                            FROM cards c
                            JOIN account_inventory ai ON c.id = ai.card_id
                            WHERE c.set_code = ? AND ai.account_id = ? AND ai.quantity > 0
                            ORDER BY CAST(c.card_number AS INTEGER)
                        """, (set_code, account_id))
                        
                        cards = []
                        for row in cursor.fetchall():
                            thumb_blob_b64 = None
                            if row[4]:  # thumbnail_blob
                                thumb_blob_b64 = base64.b64encode(row[4]).decode('utf-8')
                            
                            cards.append({
                                'id': row[0],
                                'card_number': row[1],
                                'card_name': row[2],
                                'rarity': row[3],
                                'thumbnail_blob': thumb_blob_b64,
                                'quantity': row[5]
                            })
                        
                        collection_by_set[set_code] = {
                            'set_name': set_name,
                            'cover_blob': cover_blob_b64,
                            'cards': cards
                        }
                    
                    conn.close()
                    
                    return render_template('account_collection.html',
                                        account_name=display_name,
                                        account_id=account_id,
                                        shiny_dust=shiny_dust or 0,
                                        hourglasses=hourglasses or 0,
                                        stats={
                                            'unique_cards': stats[0] or 0,
                                            'total_copies': stats[1] or 0,
                                            'sets_owned': stats[2] or 0
                                        },
                                        collection_by_set=collection_by_set)
                
                except Exception as e:
                    import traceback
                    return f"<pre>Error\n{traceback.format_exc()}</pre><a href='/'>Back</a>", 500




            @app.route('/tcg_images/<path:filename>', methods=['GET', 'POST'])
            @require_password
            def serve_card_image(filename):
                return send_from_directory('tcg_images', filename)
            
            @app.route('/', methods=['GET', 'POST'])
            @require_password
            def index():
                """Pagina principale con lista di tutti i set."""
                try:
                    import base64
                    conn = sqlite3.connect(DB_FILENAME)
                    cursor = conn.cursor()
                    
                    placeholders = ','.join('?' * len(SELECTED_RARITIES))
                    
                    # ✅ Aggiunto cover_image_blob alla query
                    cursor.execute(f"""
                    SELECT 
                        s.set_code,
                        s.set_name,
                        s.release_date,
                        s.cover_image_blob,
                        COUNT(DISTINCT CASE WHEN c.rarity IN ({placeholders}) THEN c.id END) as target_total,
                        COUNT(DISTINCT CASE WHEN c.rarity IN ({placeholders}) AND ai.quantity > 0 THEN c.id END) as owned_cards,
                        COALESCE(SUM(CASE WHEN c.rarity IN ({placeholders}) THEN ai.quantity END), 0) as total_copies
                    FROM sets s
                    LEFT JOIN cards c ON s.set_code = c.set_code
                    LEFT JOIN account_inventory ai ON c.id = ai.card_id
                    GROUP BY s.set_code
                    ORDER BY s.set_code DESC
                    """, SELECTED_RARITIES + SELECTED_RARITIES + SELECTED_RARITIES)
                    
                    sets_data = cursor.fetchall()
                    
                    sets = []
                    for row in sets_data:
                        set_code, set_name, release_date, cover_blob, target_total, owned_cards, total_copies = row
                        
                        completion = int((owned_cards / target_total) * 100) if target_total > 0 else 0
                        
                        # ✅ Converti BLOB in base64
                        cover_blob_b64 = None
                        if cover_blob:
                            cover_blob_b64 = base64.b64encode(cover_blob).decode('utf-8')
                        
                        sets.append({
                            'code': set_code,
                            'name': set_name,
                            'release_date': release_date or 'N/A',
                            'cover_blob': cover_blob_b64,
                            'total_cards': target_total,
                            'owned': owned_cards,
                            'completion': completion,
                            'copies': total_copies or 0
                        })
                    
                    conn.close()
                    return render_template('index.html', sets=sets)
                    
                except Exception as e:
                    import traceback
                    print(f"Error in index: {traceback.format_exc()}")
                    return f"<pre>Error\n{traceback.format_exc()}</pre>", 500

            @app.route('/rarity_icon/<filename>')
            def serve_rarity_icon(filename):
                """Serve rarity icons from gui folder."""
                icon_path = get_resource_path(os.path.join("gui", "rarity_icon", filename))
                if os.path.exists(icon_path):
                    return send_file(icon_path, mimetype='image/png')
                return "Icon not found", 404



            @app.route('/api/wishlist/toggle', methods=['POST'])
            @require_password
            def toggle_wishlist():
                """Toggle wishlist status for a card (global wishlist)."""
                try:
                    data = request.get_json()
                    card_id = data.get('card_id')
                    
                    if not card_id:
                        return jsonify({'success': False, 'error': 'Missing card_id'}), 400
                    
                    conn = sqlite3.connect(DB_FILENAME)
                    cursor = conn.cursor()
                    
                    # Check if card is in wishlist
                    cursor.execute("""
                    SELECT id FROM wishlist WHERE card_id = ?
                    """, (card_id,))
                    
                    existing = cursor.fetchone()
                    
                    if existing:
                        # Remove from wishlist
                        cursor.execute("DELETE FROM wishlist WHERE card_id = ?", (card_id,))
                        in_wishlist = False
                    else:
                        # Add to wishlist with priority 0
                        cursor.execute("""
                        INSERT INTO wishlist (card_id, added_at, added_date, priority)
                        VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 0)
                        """, (card_id,))
                        in_wishlist = True
                    
                    conn.commit()
                    conn.close()
                    
                    return jsonify({
                        'success': True,
                        'in_wishlist': in_wishlist,
                        'card_id': card_id
                    })
                    
                except Exception as e:
                    import traceback
                    print(f"Error toggling wishlist: {traceback.format_exc()}")
                    return jsonify({
                        'success': False,
                        'error': str(e)
                    }), 500


            @app.route('/api/wishlist/status/<int:card_id>', methods=['GET'])
            @require_password
            def get_wishlist_status(card_id):
                """Get wishlist status for a specific card."""
                try:
                    conn = sqlite3.connect(DB_FILENAME)
                    cursor = conn.cursor()
                    
                    cursor.execute("""
                    SELECT id FROM wishlist WHERE card_id = ?
                    """, (card_id,))
                    
                    in_wishlist = cursor.fetchone() is not None
                    conn.close()
                    
                    return jsonify({
                        'success': True,
                        'in_wishlist': in_wishlist,
                        'card_id': card_id
                    })
                    
                except Exception as e:
                    import traceback
                    print(f"Error getting wishlist status: {traceback.format_exc()}")
                    return jsonify({
                        'success': False,
                        'error': str(e)
                    }), 500


            @app.route('/api/wishlist/all', methods=['GET'])
            @require_password
            def get_all_wishlist():
                """Get all cards in wishlist with details."""
                try:
                    import base64
                    conn = sqlite3.connect(DB_FILENAME)
                    cursor = conn.cursor()
                    
                    cursor.execute("""
                    SELECT 
                        w.card_id,
                        c.card_name,
                        c.card_number,
                        c.rarity,
                        c.set_code,
                        c.thumbnail_blob,
                        w.added_at,
                        w.priority
                    FROM wishlist w
                    JOIN cards c ON w.card_id = c.id
                    ORDER BY w.priority DESC, w.added_at DESC
                    """)
                    
                    wishlist_items = []
                    for row in cursor.fetchall():
                        thumb_blob_b64 = None
                        if row[5]:  # thumbnail_blob
                            thumb_blob_b64 = base64.b64encode(row[5]).decode('utf-8')
                        
                        wishlist_items.append({
                            'card_id': row[0],
                            'card_name': row[1],
                            'card_number': row[2],
                            'rarity': row[3],
                            'set_code': row[4],
                            'thumbnail_blob': thumb_blob_b64,
                            'added_at': row[6],
                            'priority': row[7]
                        })
                    
                    conn.close()
                    
                    return jsonify({
                        'success': True,
                        'wishlist': wishlist_items,
                        'count': len(wishlist_items)
                    })
                    
                except Exception as e:
                    import traceback
                    print(f"Error getting wishlist: {traceback.format_exc()}")
                    return jsonify({
                        'success': False,
                        'error': str(e)
                    }), 500




            @app.route('/set/<set_code>', methods=['GET', 'POST'])
            @require_password
            def set_view(set_code):
                """Visualizza tutte le carte di un set specifico con copie."""
                try:
                    import base64
                    conn = sqlite3.connect(DB_FILENAME)
                    cursor = conn.cursor()
                    
                    filter_type = request.args.get('filter', 'all')
                    
                    # Get info del set + cover_image_blob
                    cursor.execute("""
                    SELECT set_name, release_date, total_cards, cover_image_blob
                    FROM sets
                    WHERE set_code = ?
                    """, (set_code,))
                    
                    set_info = cursor.fetchone()
                    
                    if not set_info:
                        conn.close()
                        return f"<h1>Set '{set_code}' not found</h1><a href='/'>Back</a>", 404
                    
                    set_name, release_date, total_cards, cover_blob = set_info
                    
                    cover_blob_b64 = None
                    if cover_blob:
                        cover_blob_b64 = base64.b64encode(cover_blob).decode('utf-8')
                    
                    selected_rarities_tuple = tuple(SELECTED_RARITIES)
                    placeholders = ','.join('?' * len(selected_rarities_tuple))
                    
                    # Build query based on filter + include thumbnail_blob + wishlist status
                    if filter_type == 'owned':
                        query = f"""
                        SELECT 
                            c.id,
                            c.card_number,
                            c.card_name,
                            c.rarity,
                            c.thumbnail_blob,
                            COALESCE(SUM(ai.quantity), 0) as total_copies,
                            EXISTS(SELECT 1 FROM wishlist w WHERE w.card_id = c.id) as in_wishlist
                        FROM cards c
                        LEFT JOIN account_inventory ai ON c.id = ai.card_id
                        WHERE c.set_code = ? AND c.rarity IN ({placeholders})
                        GROUP BY c.id
                        HAVING COALESCE(SUM(ai.quantity), 0) > 0
                        ORDER BY CAST(c.card_number AS INTEGER)
                        """
                    elif filter_type == 'missing':
                        query = f"""
                        SELECT 
                            c.id,
                            c.card_number,
                            c.card_name,
                            c.rarity,
                            c.thumbnail_blob,
                            COALESCE(SUM(ai.quantity), 0) as total_copies,
                            EXISTS(SELECT 1 FROM wishlist w WHERE w.card_id = c.id) as in_wishlist
                        FROM cards c
                        LEFT JOIN account_inventory ai ON c.id = ai.card_id
                        WHERE c.set_code = ? AND c.rarity IN ({placeholders})
                        GROUP BY c.id
                        HAVING COALESCE(SUM(ai.quantity), 0) = 0
                        ORDER BY CAST(c.card_number AS INTEGER)
                        """
                    else:  # 'all'
                        query = f"""
                        SELECT 
                            c.id,
                            c.card_number,
                            c.card_name,
                            c.rarity,
                            c.thumbnail_blob,
                            COALESCE(SUM(ai.quantity), 0) as total_copies,
                            EXISTS(SELECT 1 FROM wishlist w WHERE w.card_id = c.id) as in_wishlist
                        FROM cards c
                        LEFT JOIN account_inventory ai ON c.id = ai.card_id
                        WHERE c.set_code = ? AND c.rarity IN ({placeholders})
                        GROUP BY c.id
                        ORDER BY CAST(c.card_number AS INTEGER)
                        """
                    
                    cursor.execute(query, (set_code,) + selected_rarities_tuple)
                    cards = cursor.fetchall()
                    
                    # Format cards data with base64 thumbnails + wishlist status
                    cards_data = []
                    for row in cards:
                        thumb_blob_b64 = None
                        if row[4]:  # thumbnail_blob
                            thumb_blob_b64 = base64.b64encode(row[4]).decode('utf-8')
                        
                        cards_data.append({
                            'id': row[0],
                            'card_number': row[1],
                            'card_name': row[2],
                            'rarity': row[3],
                            'thumbnail_blob': thumb_blob_b64,
                            'quantity': row[5],
                            'in_wishlist': bool(row[6])  # ✅ AGGIUNTO
                        })
                    
                    conn.close()
                    
                    return render_template('set_view.html',
                                        set_code=set_code,
                                        set_name=set_name,
                                        release_date=release_date or 'N/A',
                                        total_cards=len(cards_data),
                                        cards=cards_data,
                                        filter_type=filter_type,
                                        cover_blob=cover_blob_b64)
                    
                except Exception as e:
                    import traceback
                    return f"<pre>Error\n{traceback.format_exc()}</pre><a href='/'>Back</a>", 500

            
            @app.route('/card/<int:card_id>', methods=['GET', 'POST'])
            @require_password
            def card_details(card_id):
                """Visualizza i dettagli di una carta specifica."""
                try:
                    import base64
                    conn = sqlite3.connect(DB_FILENAME)
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    
                    # Get card info + thumbnail_blob
                    cursor.execute("""
                    SELECT c.*, s.set_name
                    FROM cards c
                    JOIN sets s ON c.set_code = s.set_code
                    WHERE c.id = ?
                    """, (card_id,))
                    
                    card = cursor.fetchone()
                    
                    if not card:
                        conn.close()
                        return "Card not found", 404
                    
                    # Converti thumbnail in base64
                    thumb_blob_b64 = None
                    if card['thumbnail_blob']:
                        thumb_blob_b64 = base64.b64encode(card['thumbnail_blob']).decode('utf-8')
                    
                    # Get owners (accounts che possiedono questa carta)
                    cursor.execute("""
                    SELECT a.account_name, ai.quantity
                    FROM account_inventory ai
                    JOIN accounts a ON ai.account_id = a.device_account
                    WHERE ai.card_id = ? AND ai.quantity > 0
                    ORDER BY ai.quantity DESC
                    """, (card_id,))
                    
                    owners = cursor.fetchall()
                    
                    # Get total copies
                    cursor.execute("""
                    SELECT COALESCE(SUM(quantity), 0)
                    FROM account_inventory WHERE card_id = ?
                    """, (card_id,))
                    
                    total_copies = cursor.fetchone()[0]
                    
                    # Check if in wishlist
                    cursor.execute("""
                    SELECT id FROM wishlist WHERE card_id = ?
                    """, (card_id,))
                    
                    in_wishlist = cursor.fetchone() is not None
                    
                    conn.close()
                    
                    # Convert card to dict and add thumbnail
                    card_dict = dict(card)
                    card_dict['thumbnail_blob'] = thumb_blob_b64
                    
                    return render_template('card_details.html',
                                        card=card_dict,
                                        owners=owners,
                                        total_copies=total_copies,
                                        in_wishlist=in_wishlist)
                    
                except Exception as e:
                    import traceback
                    return f"<pre>Error\n{traceback.format_exc()}</pre><a href='/'>Back</a>", 500




            @app.route('/stats/data/summary', methods=['GET'])
            @require_password
            def stats_summary():
                """Restituisce statistiche generali"""
                db = DatabaseManager()
                if not db.connect():
                    return jsonify({"error": "Database connection failed"}), 500
                
                try:
                    # Account totali
                    db.cursor.execute("SELECT COUNT(*) FROM accounts")
                    total_accounts = db.cursor.fetchone()[0] or 0
                    
                    # Carte possedute (uniche)
                    db.cursor.execute("SELECT COUNT(DISTINCT card_id) FROM account_inventory")
                    unique_owned = db.cursor.fetchone()[0] or 0
                    
                    # Copie totali
                    db.cursor.execute("SELECT COALESCE(SUM(quantity), 0) FROM account_inventory")
                    total_copies = db.cursor.fetchone()[0] or 0
                    
                    # Carte nel database
                    db.cursor.execute("SELECT COUNT(*) FROM cards")
                    total_cards_db = db.cursor.fetchone()[0] or 0
                    
                    # Set totali
                    db.cursor.execute("SELECT COUNT(*) FROM sets")
                    total_sets = db.cursor.fetchone()[0] or 0
                    
                    # Completamento totale
                    completion = (unique_owned / total_cards_db * 100) if total_cards_db > 0 else 0
                    
                    return jsonify({
                        "totalAccounts": total_accounts,
                        "uniqueOwned": unique_owned,
                        "totalCopies": total_copies,
                        "totalCardsDB": total_cards_db,
                        "totalSets": total_sets,
                        "overallCompletion": round(completion, 1)
                    })
                except Exception as e:
                    print(f"❌ Errore stats_summary: {e}")
                    return jsonify({"error": str(e)}), 500
                finally:
                    db.close()

            @app.route('/stats/data/top-accounts', methods=['GET'])
            @require_password
            def stats_top_accounts():
                """Restituisce i top account"""
                db = DatabaseManager()
                if not db.connect():
                    return jsonify({"error": "Database connection failed"}), 500
                
                try:
                    db.cursor.execute("""
                        SELECT 
                            a.device_account,
                            a.account_name,
                            a.shiny_dust,
                            a.hourglasses,
                            COUNT(DISTINCT ai.card_id) as unique_cards,
                            COALESCE(SUM(ai.quantity), 0) as total_copies
                        FROM accounts a
                        LEFT JOIN account_inventory ai ON a.device_account = ai.account_id
                        GROUP BY a.device_account
                        ORDER BY total_copies DESC
                        LIMIT 10
                    """)
                    
                    accounts = db.cursor.fetchall()
                    result = []
                    for acc in accounts:
                        result.append({
                            "device_account": acc[0],
                            "account_name": acc[1],
                            "shiny_dust": acc[2],
                            "hourglasses": acc[3],
                            "unique_cards": acc[4],
                            "total_copies": acc[5]
                        })
                    
                    return jsonify(result)
                except Exception as e:
                    return jsonify({"error": str(e)}), 500
                finally:
                    db.close()

            @app.route('/stats/data/top-cards', methods=['GET'])
            @require_password
            def stats_top_cards():
                """Restituisce le top cards"""
                db = DatabaseManager()
                if not db.connect():
                    return jsonify({"error": "Database connection failed"}), 500
                
                try:
                    db.cursor.execute("""
                        SELECT 
                            c.id,
                            c.card_number,
                            c.card_name,
                            c.set_code,
                            SUM(ai.quantity) as total_copies,
                            COUNT(DISTINCT ai.account_id) as num_owners
                        FROM cards c
                        JOIN account_inventory ai ON c.id = ai.card_id
                        GROUP BY c.id
                        ORDER BY total_copies DESC
                        LIMIT 20
                    """)
                    
                    cards = db.cursor.fetchall()
                    result = []
                    for card in cards:
                        result.append({
                            "id": card[0],
                            "cardNumber": card[1],
                            "cardName": card[2],
                            "setCode": card[3],
                            "totalCopies": card[4],
                            "numOwners": card[5]
                        })
                    
                    return jsonify(result)
                except Exception as e:
                    return jsonify({"error": str(e)}), 500
                finally:
                    db.close()

            @app.route('/stats/data/rarity-distribution', methods=['GET'])
            @require_password
            def stats_rarity_distribution():
                """Restituisce distribuzione per rarità"""
                db = DatabaseManager()
                if not db.connect():
                    return jsonify({"error": "Database connection failed"}), 500
                
                try:
                    rarities = ['Common', 'Uncommon', 'Rare', 'Double Rare', 'Crown Rare']
                    result = []
                    
                    for rarity in rarities:
                        db.cursor.execute("SELECT COUNT(*) FROM cards WHERE rarity = ?", (rarity,))
                        total_in_db = db.cursor.fetchone()[0] or 0
                        
                        db.cursor.execute("""
                            SELECT COUNT(DISTINCT c.id)
                            FROM cards c
                            JOIN account_inventory ai ON c.id = ai.card_id
                            WHERE c.rarity = ?
                        """, (rarity,))
                        owned_unique = db.cursor.fetchone()[0] or 0
                        
                        db.cursor.execute("""
                            SELECT COALESCE(SUM(ai.quantity), 0)
                            FROM cards c
                            JOIN account_inventory ai ON c.id = ai.card_id
                            WHERE c.rarity = ?
                        """, (rarity,))
                        total_copies = db.cursor.fetchone()[0] or 0
                        
                        completion = (owned_unique / total_in_db * 100) if total_in_db > 0 else 0
                        
                        result.append({
                            "rarity": rarity,
                            "inDatabase": total_in_db,
                            "ownedUnique": owned_unique,
                            "totalCopies": total_copies,
                            "completion": round(completion, 1)
                        })
                    
                    return jsonify(result)
                except Exception as e:
                    return jsonify({"error": str(e)}), 500
                finally:
                    db.close()



            @app.route('/stats', methods=['GET', 'POST'])
            @require_password
            def stats():
                conn = sqlite3.connect(DB_FILENAME)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT a.account_name, COUNT(DISTINCT ai.card_id) as unique_cards, 
                        SUM(ai.quantity) as total_copies 
                    FROM accounts a 
                    JOIN account_inventory ai ON a.device_account = ai.account_id
                """)
                top_accounts = cursor.fetchall()
                cursor.execute("""
                    SELECT c.card_name, c.set_code, c.rarity,
                           SUM(ai.quantity) as total_copies
                    FROM cards c
                    JOIN account_inventory ai ON c.id = ai.card_id
                    WHERE ai.quantity > 0
                    GROUP BY c.id
                    ORDER BY total_copies DESC LIMIT 5
                """)
                top_cards = cursor.fetchall()
                conn.close()
                return render_template('stats.html',
                                      top_accounts=top_accounts,
                                      top_cards=top_cards)
            
            # ⬇️ USA make_server per poterlo fermare correttamente ⬇️
            self.server = make_server('0.0.0.0', 5000, app, threaded=True)
            self.log_signal.emit("🌐 Flask server started on http://localhost:5000")
            self.started_signal.emit()
            
            # Esegui il server (blocca fino a shutdown)
            self.server.serve_forever()
            
        except OSError as e:
            if "Address already in use" in str(e):
                self.error_signal.emit("Port 5000 already in use. Stop the other server first.")
            else:
                self.error_signal.emit(f"Server error: {str(e)}")
        except Exception as e:
            import traceback
            self.error_signal.emit(f"Flask error: {str(e)}\n{traceback.format_exc()}")
        finally:
            self.log_signal.emit("🌐 Flask server stopped")
            self.stopped_signal.emit()







    def stop_server(self):
        """Ferma il server Flask in modo sicuro."""
        if self.server:
            self.log_signal.emit("🌐 Stopping Flask server...")
            self.server.shutdown()  # ⬅️ Shutdown thread-safe
            self.server = None
        self.quit()
        self.wait(3000)  # Aspetta max 3 secondi


def require_password(f):
    """Decorator to require password for public access"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # ? ALWAYS reload .env (so the password is up to date)
        load_dotenv(override=True)
        cloudflare_password = os.getenv('CLOUDFLARE_PASSWORD', '').strip()
        
        # Check if it's a real localhost access (not via tunnel)
        is_localhost = (
            request.remote_addr == '127.0.0.1' and
            not request.headers.get('CF-Connecting-IP')
        )
        
        if is_localhost:
            return f(*args, **kwargs)
        
        if session.get('authenticated'):
            return f(*args, **kwargs)
        
        if request.method == 'POST':
            password = request.form.get('password', '').strip()
            
            if not cloudflare_password:
                from .translations import t, set_language, get_language
                # Charge la langue depuis settings
                try:
                    if os.path.exists('settings.json'):
                        with open('settings.json', 'r', encoding='utf-8') as settings_file:
                            settings = json.load(settings_file)
                            saved_language = settings.get('language', 'fr')
                            set_language(saved_language)
                    else:
                        set_language('fr')
                except:
                    set_language('fr')
                return render_template('login.html', error=t('web.password_not_configured'))
            
            if password == cloudflare_password:  # Use the updated local variable
                session['authenticated'] = True
                return f(*args, **kwargs)
            else:
                from .translations import t, set_language, get_language
                # Charge la langue depuis settings
                try:
                    if os.path.exists('settings.json'):
                        with open('settings.json', 'r', encoding='utf-8') as settings_file:
                            settings = json.load(settings_file)
                            saved_language = settings.get('language', 'fr')
                            set_language(saved_language)
                    else:
                        set_language('fr')
                except:
                    set_language('fr')
                return render_template('login.html', error=t('web.incorrect_password'))
        
        return render_template('login.html')
    
    return decorated_function



def get_accounts_stats(db: DatabaseManager):
    """
    Recupera statistiche per ogni account.
    Restituisce lista di tuple: (device_account, account_name, shiny_dust, hourglasses, unique_cards, total_copies)
    """
    try:
        db.cursor.execute("""
            SELECT 
                a.device_account,
                a.account_name,
                COALESCE(a.shiny_dust, 0),
                COALESCE(a.hourglasses, 0),
                COUNT(DISTINCT ai.card_id) as unique_cards,
                COALESCE(SUM(ai.quantity), 0) as total_copies
            FROM accounts a
            LEFT JOIN account_inventory ai ON a.device_account = ai.account_id
            GROUP BY a.device_account, a.account_name
            ORDER BY a.account_name
        """)
        return db.cursor.fetchall()
    except Exception as e:
        print(f"❌ Errore get_accounts_stats: {e}")
        return []

def get_rarity_stats(db: DatabaseManager):
    """
    Recupera statistiche per rarity.
    Restituisce lista di tuple: (rarity, total_in_db, owned_unique, total_copies)
    """
    rarities = ['Common', 'Uncommon', 'Rare', 'Double Rare', 'Crown Rare']
    stats = []
    
    try:
        for rarity in rarities:
            # Conta carte totali di questa rarity nel database
            db.cursor.execute("""
                SELECT COUNT(*) FROM cards WHERE rarity = ?
            """, (rarity,))
            total_in_db = db.cursor.fetchone()[0] or 0
            
            # Conta carte possedute (unique) di questa rarity
            db.cursor.execute("""
                SELECT COUNT(DISTINCT c.id)
                FROM cards c
                JOIN account_inventory ai ON c.id = ai.card_id
                WHERE c.rarity = ?
            """, (rarity,))
            owned_unique = db.cursor.fetchone()[0] or 0
            
            # Somma totale copie di questa rarity
            db.cursor.execute("""
                SELECT COALESCE(SUM(ai.quantity), 0)
                FROM cards c
                JOIN account_inventory ai ON c.id = ai.card_id
                WHERE c.rarity = ?
            """, (rarity,))
            total_copies = db.cursor.fetchone()[0] or 0
            
            stats.append((rarity, total_in_db, owned_unique, total_copies))
        
        return stats
    except Exception as e:
        print(f"❌ Errore get_rarity_stats: {e}")
        return [(r, 0, 0, 0) for r in rarities]

def get_completion_percentage(db: DatabaseManager):
    """
    Calcola percentuale di completamento totale della collezione.
    """
    try:
        db.cursor.execute("""
            SELECT 
                COUNT(DISTINCT c.id) as total_cards_db,
                COUNT(DISTINCT CASE WHEN ai.id IS NOT NULL THEN c.id END) as owned_unique
            FROM cards c
            LEFT JOIN account_inventory ai ON c.id = ai.card_id
        """)
        result = db.cursor.fetchone()
        
        if result:
            total_db, owned = result
            if total_db > 0:
                return round((owned / total_db) * 100, 2)
        return 0.0
    except Exception as e:
        print(f"❌ Errore get_completion_percentage: {e}")
        return 0.0
@stats_bp.route('/')
def stats_page():
    """
    Renderizza la pagina stats.html con tutti i dati dal database.
    """
    db = DatabaseManager()
    
    if not db.connect():
        return "❌ Errore connessione al database", 500
    
    try:
        # Recupera tutti i dati
        accounts = get_accounts_stats(db)
        rarities = get_rarity_stats(db)
        completion = get_completion_percentage(db)
        
        # Conta totali
        total_cards_owned = sum(row[5] if row[5] else 0 for row in accounts)
        total_unique_owned = sum(row[4] if row[4] else 0 for row in accounts)
        
        print(f"✅ Dati caricati: {len(accounts)} account, completion: {completion}%")
        
        # Renderizza il template con i dati
        return render_template('stats.html',
            accounts=accounts,
            rarities=rarities,
            completion_percentage=completion,
            total_cards_owned=total_cards_owned,
            total_unique_owned=total_unique_owned,
            total_accounts=len(accounts)
        )
    
    except Exception as e:
        print(f"❌ Errore stats_page: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Errore: {e}", 500
    
    finally:
        db.close()

# ============================================================================
# 📡 API ROUTES PER ACCESSO AI DATI (JSON)
# ============================================================================

@stats_bp.route('/api/accounts', methods=['GET'])
def api_accounts():
    """
    Restituisce dati account in formato JSON.
    """
    db = DatabaseManager()
    
    if not db.connect():
        return jsonify({"error": "Database connection failed"}), 500
    
    try:
        accounts = get_accounts_stats(db)
        result = []
        
        for acc in accounts:
            result.append({
                "device_account": acc[0],
                "account_name": acc[1],
                "shiny_dust": acc[2],
                "hourglasses": acc[3],
                "unique_cards": acc[4],
                "total_copies": acc[5]
            })
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    finally:
        db.close()

@stats_bp.route('/api/rarities', methods=['GET'])
def api_rarities():
    """
    Restituisce statistiche rarity in formato JSON.
    """
    db = DatabaseManager()
    
    if not db.connect():
        return jsonify({"error": "Database connection failed"}), 500
    
    try:
        rarities = get_rarity_stats(db)
        result = []
        
        for rarity in rarities:
            total_in_db = rarity[1]
            owned_unique = rarity[2]
            completion = (owned_unique / total_in_db * 100) if total_in_db > 0 else 0
            
            result.append({
                "rarity": rarity[0],
                "total_in_database": total_in_db,
                "owned_unique": owned_unique,
                "total_copies": rarity[3],
                "completion_percentage": round(completion, 2)
            })
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    finally:
        db.close()
