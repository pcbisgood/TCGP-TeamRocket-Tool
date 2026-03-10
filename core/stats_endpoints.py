# ============================================================================
# 📊 STATS ENDPOINTS - LEGGE DIRETTAMENTE DAL DATABASE
# ============================================================================

"""
Aggiungi questo file come: stats_endpoints.py
Contiene TUTTI gli endpoint per stats.html
"""

from flask import Blueprint, render_template, jsonify
from .database import DatabaseManager
import base64

stats_bp = Blueprint('stats', __name__, url_prefix='/stats')

# ============================================================================
# 🗄️ ENDPOINT: TOP ACCOUNTS
# ============================================================================

@stats_bp.route('/data/top-accounts')
def get_top_accounts():
    """Restituisce i top account per collezione"""
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
            result.append((acc[0], acc[1], acc[2], acc[3], acc[4], acc[5]))
        
        return jsonify({"accounts": result})
    except Exception as e:
        print(f"❌ Errore top_accounts: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

# ============================================================================
# 🎴 ENDPOINT: TOP CARDS (PIÙ POSSEDUTE)
# ============================================================================

@stats_bp.route('/data/top-cards')
def get_top_cards():
    """Restituisce le carte più possedute"""
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
                c.thumbnail_blob,
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
                "thumbnailBlob": base64.b64encode(card[4]).decode() if card[4] else None,
                "totalCopies": card[5],
                "numOwners": card[6]
            })
        
        return jsonify({"cards": result})
    except Exception as e:
        print(f"❌ Errore top_cards: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

# ============================================================================
# 💎 ENDPOINT: RAREST CARDS
# ============================================================================

@stats_bp.route('/data/rarest-cards')
def get_rarest_cards():
    """Restituisce le carte più rare possedute"""
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
                c.rarity,
                c.thumbnail_blob,
                COUNT(DISTINCT ai.account_id) as num_owners
            FROM cards c
            JOIN account_inventory ai ON c.id = ai.card_id
            WHERE c.rarity IN ('Crown Rare', 'Double Rare')
            GROUP BY c.id
            ORDER BY 
                CASE c.rarity 
                    WHEN 'Crown Rare' THEN 1 
                    WHEN 'Double Rare' THEN 2 
                END,
                num_owners ASC
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
                "rarity": card[4],
                "thumbnailBlob": base64.b64encode(card[5]).decode() if card[5] else None,
                "numOwners": card[6]
            })
        
        return jsonify({"cards": result})
    except Exception as e:
        print(f"❌ Errore rarest_cards: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

# ============================================================================
# 📊 ENDPOINT: RARITY DISTRIBUTION
# ============================================================================

@stats_bp.route('/data/rarity-distribution')
def get_rarity_distribution():
    """Restituisce distribuzione per rarità"""
    db = DatabaseManager()
    if not db.connect():
        return jsonify({"error": "Database connection failed"}), 500
    
    try:
        rarities = ['Common', 'Uncommon', 'Rare', 'Double Rare', 'Crown Rare']
        result = []
        
        for rarity in rarities:
            # Totale nel database
            db.cursor.execute("SELECT COUNT(*) FROM cards WHERE rarity = ?", (rarity,))
            total_in_db = db.cursor.fetchone()[0] or 0
            
            # Possedute (uniche)
            db.cursor.execute("""
                SELECT COUNT(DISTINCT c.id)
                FROM cards c
                JOIN account_inventory ai ON c.id = ai.card_id
                WHERE c.rarity = ?
            """, (rarity,))
            owned_unique = db.cursor.fetchone()[0] or 0
            
            # Totale copie
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
        
        return jsonify({"rarities": result})
    except Exception as e:
        print(f"❌ Errore rarity_distribution: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

# ============================================================================
# 📦 ENDPOINT: SET COMPLETION
# ============================================================================

@stats_bp.route('/data/set-completion')
def get_set_completion():
    """Restituisce completamento per set"""
    db = DatabaseManager()
    if not db.connect():
        return jsonify({"error": "Database connection failed"}), 500
    
    try:
        db.cursor.execute("""
            SELECT 
                s.set_code,
                s.set_name,
                s.cover_image_blob,
                s.total_cards,
                COUNT(DISTINCT CASE WHEN ai.id IS NOT NULL THEN c.id END) as owned_cards
            FROM sets s
            LEFT JOIN cards c ON s.set_code = c.set_code
            LEFT JOIN account_inventory ai ON c.id = ai.card_id
            GROUP BY s.set_code
            ORDER BY s.set_name
        """)
        
        sets = db.cursor.fetchall()
        result = []
        
        for s in sets:
            total = s[3] or 0
            owned = s[4] or 0
            completion = (owned / total * 100) if total > 0 else 0
            
            result.append({
                "setCode": s[0],
                "setName": s[1],
                "coverBlob": base64.b64encode(s[2]).decode() if s[2] else None,
                "totalCards": total,
                "ownedCards": owned,
                "completion": round(completion, 1)
            })
        
        return jsonify({"sets": result})
    except Exception as e:
        print(f"❌ Errore set_completion: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

# ============================================================================
# 📈 ENDPOINT: SUMMARY STATS
# ============================================================================

@stats_bp.route('/data/summary')
def get_summary_stats():
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
        
        # Wishlist
        db.cursor.execute("SELECT COUNT(*) FROM wishlist")
        wishlist_count = db.cursor.fetchone()[0] or 0
        
        # Completamento totale
        completion = (unique_owned / total_cards_db * 100) if total_cards_db > 0 else 0
        
        return jsonify({
            "totalAccounts": total_accounts,
            "uniqueOwned": unique_owned,
            "totalCopies": total_copies,
            "totalCardsDB": total_cards_db,
            "totalSets": total_sets,
            "wishlistCount": wishlist_count,
            "overallCompletion": round(completion, 1)
        })
    except Exception as e:
        print(f"❌ Errore summary_stats: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

# ============================================================================
# 📄 ENDPOINT: MAIN PAGE - Renderizza stats.html con tutti i dati
# ============================================================================

@stats_bp.route('/')
def stats_page():
    """Renderizza la pagina stats.html"""
    db = DatabaseManager()
    if not db.connect():
        return "❌ Errore connessione al database", 500
    
    try:
        # Statistiche generali
        db.cursor.execute("""
            SELECT 
                COUNT(DISTINCT a.device_account) as total_accounts,
                COUNT(DISTINCT ai.card_id) as unique_owned,
                COALESCE(SUM(ai.quantity), 0) as total_copies,
                COUNT(DISTINCT c.id) as total_cards_db,
                COUNT(DISTINCT s.set_code) as total_sets,
                COUNT(DISTINCT w.id) as wishlist_count
            FROM accounts a
            LEFT JOIN account_inventory ai ON a.device_account = ai.account_id
            LEFT JOIN cards c ON c.id = ai.card_id
            LEFT JOIN sets s ON s.set_code = c.set_code
            LEFT JOIN wishlist w ON w.id IS NOT NULL
        """)
        
        stats = db.cursor.fetchone()
        total_accounts = stats[0] or 0
        total_unique_owned = stats[1] or 0
        total_cards_owned = stats[2] or 0
        total_cards_db = stats[3] or 0
        total_sets = stats[4] or 0
        wishlist_count = stats[5] or 0
        
        overall_completion = (total_unique_owned / total_cards_db * 100) if total_cards_db > 0 else 0
        
        # Top accounts
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
        top_accounts = db.cursor.fetchall()
        
        # Top cards
        db.cursor.execute("""
            SELECT 
                c.id,
                c.card_number,
                c.card_name,
                c.set_code,
                c.thumbnail_blob,
                SUM(ai.quantity) as total_copies,
                COUNT(DISTINCT ai.account_id) as num_owners
            FROM cards c
            JOIN account_inventory ai ON c.id = ai.card_id
            GROUP BY c.id
            ORDER BY total_copies DESC
            LIMIT 20
        """)
        top_cards = db.cursor.fetchall()
        
        # Rarest cards
        db.cursor.execute("""
            SELECT 
                c.id,
                c.card_number,
                c.card_name,
                c.set_code,
                c.rarity,
                c.thumbnail_blob,
                COUNT(DISTINCT ai.account_id) as num_owners
            FROM cards c
            JOIN account_inventory ai ON c.id = ai.card_id
            WHERE c.rarity IN ('Crown Rare', 'Double Rare')
            GROUP BY c.id
            ORDER BY 
                CASE c.rarity WHEN 'Crown Rare' THEN 1 WHEN 'Double Rare' THEN 2 END,
                num_owners ASC
            LIMIT 20
        """)
        rarest_cards = db.cursor.fetchall()
        
        # Set completion
        db.cursor.execute("""
            SELECT 
                s.set_code,
                s.set_name,
                s.cover_image_blob,
                s.total_cards,
                COUNT(DISTINCT CASE WHEN ai.id IS NOT NULL THEN c.id END) as owned_cards
            FROM sets s
            LEFT JOIN cards c ON s.set_code = c.set_code
            LEFT JOIN account_inventory ai ON c.id = ai.card_id
            GROUP BY s.set_code
            ORDER BY s.set_name
        """)
        sets_completion = db.cursor.fetchall()
        
        # Rarity distribution
        rarities = ['Common', 'Uncommon', 'Rare', 'Double Rare', 'Crown Rare']
        rarity_distribution = []
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
            
            rarity_distribution.append((rarity, total_in_db, owned_unique, total_copies))
        
        return render_template('stats.html',
            totalUniqueOwned=total_unique_owned,
            totalCopies=total_cards_owned,
            totalCardsDB=total_cards_db,
            totalAccounts=total_accounts,
            totalSets=total_sets,
            wishlistCount=wishlist_count,
            overallCompletion=round(overall_completion, 1),
            topAccounts=top_accounts,
            topCards=top_cards,
            rarestCards=rarest_cards,
            setsCompletion=sets_completion,
            rarityDistribution=rarity_distribution
        )
    except Exception as e:
        print(f"❌ Errore stats_page: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Errore: {e}", 500
    finally:
        db.close()
