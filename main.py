"""main.py - Entry point principale dell'applicazione"""
import sys
import os
import json
from PyQt5.QtWidgets import QApplication
from core.ui_main_window import MainWindow
from core.utils import apply_dark_theme

# Importa la funzione per ottenere il percorso
from config import create_default_settings, SETTINGS_FILE
import pkgutil
import core

for module in pkgutil.iter_modules(core.__path__):
    __import__(f"core.{module.name}")
# Percorso del file di impostazioni

def main():
    """Entry point principale"""
    
    create_default_settings()
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()