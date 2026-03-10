"""utils.py - Utility functions generiche"""

# Import standard library
import os
import sys
from typing import Optional, Any

# Import PyQt5
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor

# Import configurazione
from config import ICON_PATH, get_resource_path


def apply_dark_theme(app):
    """Applica il tema scuro all'applicazione."""
    app.setStyle("Fusion")
    
    darkpalette = QPalette()
    darkpalette.setColor(QPalette.Window, QColor(53, 53, 53))
    darkpalette.setColor(QPalette.WindowText, Qt.white)
    darkpalette.setColor(QPalette.Base, QColor(35, 35, 35))
    darkpalette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    darkpalette.setColor(QPalette.ToolTipBase, QColor(25, 25, 25))
    darkpalette.setColor(QPalette.ToolTipText, Qt.white)
    darkpalette.setColor(QPalette.Text, Qt.white)
    darkpalette.setColor(QPalette.Button, QColor(53, 53, 53))
    darkpalette.setColor(QPalette.ButtonText, Qt.white)
    darkpalette.setColor(QPalette.BrightText, Qt.red)
    
    # ⬇️ CAMBIA QUESTI DA BLU A GIALLO ⬇️
    darkpalette.setColor(QPalette.Link, QColor(243, 156, 18))  # #f39c12 - Giallo oro
    darkpalette.setColor(QPalette.Highlight, QColor(243, 156, 18))  # #f39c12 - Giallo oro
    darkpalette.setColor(QPalette.HighlightedText, Qt.black)
    
    darkpalette.setColor(QPalette.Disabled, QPalette.Text, Qt.darkGray)
    darkpalette.setColor(QPalette.Disabled, QPalette.ButtonText, Qt.darkGray)
    
    app.setPalette(darkpalette)
    
    # Additional stylesheet
    app.setStyleSheet("""
        QToolTip { 
            color: #ffffff; 
            background-color: #2a82da; 
            border: 1px solid white; 
        }
        QGroupBox {
            border: 2px solid #555;
            border-radius: 5px;
            margin-top: 1ex;
            padding: 15px;  /* ⬅️ Padding interno del box */
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 5px 10px;  /* ⬅️ Padding del titolo (verticale, orizzontale) */
            color: #f39c12;
        }

        QTabWidget::pane {
            border: 1px solid #444;
            border-radius: 3px;
        }
        QTabBar::tab {
            background: #353535;
            border: 1px solid #444;
            padding: 8px 16px;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background: #2a82da;
            color: white;
        }
        QTabBar::tab:hover {
            background: #454545;
        }
        QPushButton {
            border: 1px solid #555;
            border-radius: 3px;
            padding: 5px 15px;
            background-color: #454545;
        }
        QPushButton:hover {
            background-color: #555;
        }
        QPushButton:pressed {
            background-color: #353535;
        }
        QPushButton:disabled {
            background-color: #2a2a2a;
            color: #666;
        }
        QLineEdit {
            border: 1px solid #555;
            border-radius: 3px;
            padding: 5px;
            background-color: #2a2a2a;
        }
        QLineEdit:focus {
            border: 1px solid #2a82da;
        }
        QTextEdit {
            border: 1px solid #555;
            border-radius: 3px;
            background-color: #2a2a2a;
        }
        QTableWidget {
            border: 1px solid #555;
            gridline-color: #555;
            background-color: #2a2a2a;
        }
        QTableWidget::item:selected {
            background-color: #2a82da;
        }
        QHeaderView::section {
            background-color: #454545;
            padding: 5px;
            border: 1px solid #555;
            font-weight: bold;
        }
        QProgressBar {
            border: 1px solid #555;
            border-radius: 3px;
            text-align: center;
            background-color: #2a2a2a;
        }
        QProgressBar::chunk {
            background-color: #f39c12;
            border-radius: 2px;
        }
        QCheckBox {
            spacing: 5px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
        }
        QCheckBox::indicator:unchecked {
            border: 1px solid #555;
            background-color: #2a2a2a;
            border-radius: 3px;
        }
        QCheckBox::indicator:checked {
            border: 1px solid #2a82da;
            background-color: #2a82da;
            border-radius: 3px;
        }
        QSpinBox {
            border: 1px solid #555;
            border-radius: 3px;
            padding: 3px;
            background-color: #2a2a2a;
        }
        QScrollBar:vertical {
            border: none;
            background: #2a2a2a;
            width: 14px;
            margin: 0;
        }
        QScrollBar::handle:vertical {
            background: #555;
            min-height: 20px;
            border-radius: 7px;
        }
        QScrollBar::handle:vertical:hover {
            background: #666;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        QScrollBar:horizontal {
            border: none;
            background: #2a2a2a;
            height: 14px;
            margin: 0;
        }
        QScrollBar::handle:horizontal {
            background: #555;
            min-width: 20px;
            border-radius: 7px;
        }
        QScrollBar::handle:horizontal:hover {
            background: #666;
        }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            width: 0px;
        }
    """)
