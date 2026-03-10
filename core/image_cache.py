"""image_cache.py - Cache thread-safe per immagini"""

from PyQt5.QtGui import QPixmap
from threading import Lock
from collections import OrderedDict


class LRUImageCache:
    """Cache LRU thread-safe ottimizzato"""
    
    def __init__(self, max_size=500):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.lock = Lock()
    
    def get(self, key):
        """Recupera immagine dal cache"""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)  # Sposta alla fine (più recente)
                return self.cache[key]
        return None
    
    def put(self, key, value):
        """Memorizza immagine nel cache"""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    self.cache.popitem(last=False)  # Rimuovi il meno recente
            self.cache[key] = value
    
    def clear(self):
        """Pulisce tutto il cache"""
        with self.lock:
            self.cache.clear()
    
    def size(self):
        """Ritorna numero immagini in cache"""
        return len(self.cache)
