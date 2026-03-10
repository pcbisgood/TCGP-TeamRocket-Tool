# core/notification_manager.py
"""
Gestisce l'invio di notifiche "Rich" (Windows Toasts)
per gli eventi dell'applicazione, come la Wishlist.
"""

import os
import tempfile
import io 
import re 
import time
from config import ICON_PATH 

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ PIL (Pillow) not installed. Wishlist image notifications will be limited.")

try:
    from windows_toasts import Toast, WindowsToaster, ToastDisplayImage, ToastImagePosition
    WINDOWS_TOAST_AVAILABLE = True
except ImportError:
    WINDOWS_TOAST_AVAILABLE = False

# Per testare se l'immagine è valida
try:
    from PyQt5.QtGui import QPixmap
except ImportError:
    QPixmap = None


def send_toast_notification(card_data: dict) -> bool:
    """
    Tenta di inviare una notifica "Rich" (Windows Toast) per una carta.
    Il blob viene salvato temporaneamente su disco perché Windows Toast richiede un file path.
    """
    print("🔥 SEND_TOAST_NOTIFICATION CHIAMATA!")  # Debug
    
    if not WINDOWS_TOAST_AVAILABLE or not PIL_AVAILABLE:
        print("⚠️ Windows Toast o PIL non disponibili")
        return False

    try:
        card_name = card_data.get('card_name', 'Unknown')
        set_code = card_data.get('set_code', '')
        card_number = card_data.get('card_number', 'unknown')
        rarity = card_data.get('rarity', '')
        card_blob = card_data.get('thumbnail_blob')
        card_image_path = card_data.get('local_image_path')

        toaster = WindowsToaster('🎴 TCGP Team Rocket Tool')
        toast = Toast()
        
        toast.text_fields = [
            '❤️ Wishlist Card Found!',
            f'{card_name}',
            f'{set_code} #{card_number} • {rarity}'
        ]

        # ================================================================
        # 1. PREPARA L'IMMAGINE DELLA CARTA DAL BLOB
        # ================================================================
        card_image_for_toast = None
        temp_image_path = None 
        
        # Crea un nome file sicuro in una cartella dedicata
        safe_card_num = re.sub(r'[\\/*?:"<>|]', "", str(card_number))
        
        # Usa una cartella dedicata invece di temp (più stabile)
        toast_dir = os.path.join(os.path.expanduser("~"), "AppData", "Local", "TCGP_Toasts")
        os.makedirs(toast_dir, exist_ok=True)
        
        temp_filename = f'toast_card_{set_code}_{safe_card_num}.png'
        temp_image_path = os.path.join(toast_dir, temp_filename)

        try:
            img = None
            
            # Priorità 1: Usare il BLOB dal database
            if card_blob:
                print(f"🔍 BLOB presente, dimensione: {len(card_blob)} bytes")
                img = Image.open(io.BytesIO(card_blob))
                print(f"✅ Immagine caricata dal BLOB: {img.size}")

            # Priorità 2: Usare il file path (fallback)
            elif card_image_path and os.path.exists(card_image_path):
                print(f"📁 Caricamento da file: {card_image_path}")
                img = Image.open(card_image_path)
            else:
                print("⚠️ Nessuna immagine disponibile (né blob né file)")
            
            # Se abbiamo un'immagine, salviamola temporaneamente
            if img:
                # Converti in RGB se necessario (PNG richiede RGB o RGBA)
                if img.mode not in ('RGB', 'RGBA'):
                    img = img.convert('RGBA')
                
                # Ridimensiona per la posizione AppLogoOverride (quadrata, ~80x80)
                img.thumbnail((80, 80), Image.Resampling.LANCZOS)
                
                # Salva con massima qualità
                img.save(temp_image_path, 'PNG', optimize=False)
                
                # ✅ ASPETTA che il file sia effettivamente scritto
                import time
                time.sleep(0.1)  # 100ms di attesa
                
                # Verifica che il file esista E sia leggibile
                if os.path.exists(temp_image_path) and os.path.getsize(temp_image_path) > 0:
                    card_image_for_toast = os.path.abspath(temp_image_path)
                    print(f"✅ File temporaneo creato: {card_image_for_toast}, size: {os.path.getsize(temp_image_path)} bytes")
                else:
                    print(f"❌ File temporaneo NON valido: {temp_image_path}")
                    card_image_for_toast = None
        except Exception as e:
            print(f"❌ Errore preparazione immagine carta: {e}")
            import traceback
            traceback.print_exc()
            card_image_for_toast = None
        
        # Aggiungi l'immagine della carta come icona app
        if card_image_for_toast and os.path.exists(card_image_for_toast):
            try:
                # ✅ Verifica nuovamente che il file sia valido
                if QPixmap:
                    test_pixmap = QPixmap(card_image_for_toast)
                    if test_pixmap.isNull():
                        print(f"⚠️ QPixmap invalido per: {card_image_for_toast}")
                        card_image_for_toast = None
                
                if card_image_for_toast:
                    # ✅ USA AppLogo invece di AppLogoOverride (che non esiste)
                    toast.AddImage(ToastDisplayImage.fromPath(
                        card_image_for_toast,
                        position=ToastImagePosition.AppLogo,
                        circleCrop=False  # Mantieni forma quadrata
                    ))
                    print(f"✅ Immagine carta aggiunta: {card_image_for_toast}")
            except Exception as e:
                print(f"❌ Errore aggiunta immagine Toast: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("⚠️ Immagine carta non disponibile, verrà mostrata l'icona di default")
        
        # ================================================================
        # 3. MOSTRA LA NOTIFICA
        # ================================================================
        toaster.show_toast(toast)
        print(f"🎉 Notifica Wishlist (Toast) inviata per: {card_name}")
        
        # Nota: Non puliamo il file temporaneo immediatamente
        # perché Windows Toast potrebbe averne bisogno per qualche istante

        return True

    except Exception as e:
        print(f"❌ Errore invio Windows Toast: {e}")
        import traceback
        traceback.print_exc()
        return False