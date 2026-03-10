# core/wishlist_manager.py
"""
Gestisce lo stato e la logica della Wishlist.
"""

from .database import DatabaseManager

class WishlistManager:
    """
    Gestisce lo stato della wishlist per evitare query
    continue al database.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.wishlist_set = set() # Un set per controlli 'in' super veloci

    def load_wishlist(self):
        """
        (Ri)Carica l'intera wishlist dal database alla cache interna (self.wishlist_set).
        """
        print("Sincronizzazione Wishlist...")
        self.wishlist_set = self.db.get_wishlist()
        print(f"✅ Wishlist caricata: {len(self.wishlist_set)} carte.")

    def is_wished(self, card_id: int) -> bool:
        """
        Controlla se una carta è nella wishlist (usa la cache interna).
        """
        return card_id in self.wishlist_set

    def toggle_wishlist(self, card_id: int) -> bool:
        """
        Esegue il toggle nel database e aggiorna la cache interna.
        Restituisce il *nuovo* stato (True se è in wishlist, False altrimenti).
        """
        # 1. Esegui la modifica sul DB
        new_state_is_wished = self.db.toggle_wishlist_status(card_id)
        
        # 2. Aggiorna la cache interna
        if new_state_is_wished:
            self.wishlist_set.add(card_id)
        else:
            self.wishlist_set.discard(card_id) # 'discard' non dà errore se non c'è
            
        return new_state_is_wished