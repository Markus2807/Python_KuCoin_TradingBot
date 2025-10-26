
import os
import warnings
warnings.filterwarnings('ignore')

# Raspberry Pi spezifische Einstellungen
import platform
if platform.system() == "Linux" and 'raspberrypi' in platform.uname().release.lower():
    # Für Raspberry Pi Display
    os.environ['DISPLAY'] = ':0'
    # Für bessere Performance auf Raspberry Pi
    os.environ['TK_SILENCE_DEPRECATION'] = '1'

from trading_bot import KuCoinTradingBot
from gui import TradingBotGUI

def load_env_file():
    """Lädt Umgebungsvariablen aus .env Datei ohne dotenv Modul"""
    env_vars = {}
    env_files = ['api.env', '.env']
    env_file_found = None
    
    for env_file in env_files:
        if os.path.exists(env_file):
            env_file_found = env_file
            break
    
    if env_file_found:
        print(f"📁 Lade Umgebungsvariablen aus: {env_file_found}")
        try:
            with open(env_file_found, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip().strip('"').strip("'")
        except Exception as e:
            print(f"⚠️  Fehler beim Laden der .env Datei: {e}")
    else:
        print("⚠️  Keine .env Datei gefunden. Verwende Standardwerte.")
    
    return env_vars

def main():
    print("🚀 Starte KuCoin Trading Bot - Optimiert für Raspberry Pi...")
    
    # Systeminfo anzeigen
    import platform
    system_info = f"{platform.system()} {platform.release()}"
    print(f"💻 System: {system_info}")
    
    # Lade API-Daten aus .env Datei
    env_vars = load_env_file()
    
    API_KEY = env_vars.get('KUCOIN_API_KEY')
    API_SECRET = env_vars.get('KUCOIN_API_SECRET')
    API_PASSPHRASE = env_vars.get('KUCOIN_API_PASSPHRASE')
    SANDBOX = env_vars.get('KUCOIN_SANDBOX', 'False').lower() == 'true'
    
    if not all([API_KEY, API_SECRET, API_PASSPHRASE]):
        print("❌ Fehler: API-Daten nicht gefunden!")
        print("💡 Bitte erstelle eine .env oder api.env Datei mit folgenden Inhalten:")
        print("   KUCOIN_API_KEY=dein_api_key")
        print("   KUCOIN_API_SECRET=dein_api_secret")
        print("   KUCOIN_API_PASSPHRASE=dein_api_passphrase")
        print("   KUCOIN_SANDBOX=False")
        return
    
    print(f"✅ API-Daten geladen - Sandbox Modus: {SANDBOX}")
    
    try:
        bot = KuCoinTradingBot(
            api_key=API_KEY,
            api_secret=API_SECRET,
            api_passphrase=API_PASSPHRASE,
            sandbox=SANDBOX
        )
        
        print("🎨 Starte GUI...")
        gui = TradingBotGUI(bot)
        print("✅ GUI erfolgreich gestartet")
        gui.run()
        
    except Exception as e:
        print(f"❌ Fehler beim Starten der GUI: {e}")
        print("💡 Tipps zur Problembehebung:")
        print("   1. Stellen Sie sicher, dass ein Display angeschlossen ist")
        print("   2. Prüfen Sie mit 'echo $DISPLAY' ob :0 angezeigt wird")
        print("   3. Installieren Sie Tkinter: sudo apt install python3-tk")
        print("   4. Starten Sie den Raspberry Pi neu")

if __name__ == "__main__":
    main()