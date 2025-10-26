
import os
import warnings
warnings.filterwarnings('ignore')
import time
import schedule
from datetime import datetime
import sys

# Setze DISPLAY Environment Variable für Headless-Betrieb
os.environ['DISPLAY'] = ':0'

from trading_bot import KuCoinTradingBot

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

class HeadlessTradingBot:
    def __init__(self, bot):
        self.bot = bot
        self.bot.set_headless_reference(self)
        self.setup_schedule()
        
    def set_headless_reference(self, headless):
        """Setzt Referenz für Headless-Betrieb"""
        self.headless_reference = headless
        
    def update_bot_activity(self, message):
        """Aktualisiert Bot-Aktivität (ersetzt GUI-Log)"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] {message}")
        
    def setup_schedule(self):
        """Richtet den Zeitplan für automatische Aufgaben ein"""
        # Backtest alle 15 Minuten
        schedule.every(15).minutes.do(self.run_scheduled_backtest)
        
        # Kontostand aktualisieren alle 5 Minuten
        schedule.every(5).minutes.do(self.update_balances)
        
        # Stop-Loss prüfen jede Minute
        schedule.every(1).minutes.do(self.check_stop_loss)
        
        # Cache aktualisieren alle 10 Minuten
        schedule.every(10).minutes.do(self.update_caches)
        
        print("✅ Zeitplan für automatische Aufgaben eingerichtet")
        
    def run_scheduled_backtest(self):
        """Führt geplanten Backtest durch"""
        print("🕐 Führe geplanten Backtest durch...")
        try:
            results = self.bot.run_complete_backtest()
            if results:
                print(f"✅ Backtest abgeschlossen - {len(results)} Kryptos analysiert")
                self.execute_auto_trades(results)
            else:
                print("❌ Backtest fehlgeschlagen")
        except Exception as e:
            print(f"❌ Fehler beim Backtest: {e}")
            
    def execute_auto_trades(self, results):
        """Führt automatische Trades basierend auf Ergebnissen aus"""
        if not self.bot.auto_trading:
            return
            
        print("🤖 Prüfe Handelsmöglichkeiten...")
        for crypto, data in results.items():
            if "BUY" in data['current_signal'] and data['confidence'] >= 70:
                print(f"🎯 Signal für {crypto}: {data['current_signal']} (Confidence: {data['confidence']}%)")
                success = self.bot.execute_trade(crypto, data['current_signal'])
                if success:
                    print(f"✅ Trade für {crypto} ausgeführt")
                else:
                    print(f"❌ Trade für {crypto} fehlgeschlagen")
                    
    def update_balances(self):
        """Aktualisiert Kontostände"""
        print("💰 Aktualisiere Kontostände...")
        try:
            balance = self.bot.get_balance_summary()
            if balance:
                total_value = balance['total_portfolio_value']
                print(f"💼 Portfolio Wert: ${total_value:,.2f}")
        except Exception as e:
            print(f"❌ Fehler beim Aktualisieren der Kontostände: {e}")
            
    def check_stop_loss(self):
        """Prüft Stop-Loss Levels"""
        self.bot.check_stop_loss()
        
    def update_caches(self):
        """Aktualisiert Caches"""
        print("🔄 Aktualisiere Caches...")
        self.bot.update_caches()
        
    def run(self):
        """Startet den Headless-Betrieb"""
        print("🚀 Starte KuCoin Trading Bot - Headless Modus")
        print("📊 Führe initialen Backtest durch...")
        
        # Initialen Backtest durchführen
        initial_results = self.bot.run_complete_backtest()
        if initial_results:
            print(f"✅ Initialer Backtest abgeschlossen - {len(initial_results)} Kryptos analysiert")
            
            # Zeige Ergebnisse an
            for crypto, data in initial_results.items():
                signal_emoji = "🟢" if "BUY" in data['current_signal'] else "🔴" if "SELL" in data['current_signal'] else "🟡"
                print(f"  {signal_emoji} {crypto}: {data['current_signal']} ({data['confidence']}%)")
        else:
            print("❌ Initialer Backtest fehlgeschlagen")
            
        # Zeige Auto-Trading Status
        status = "AKTIV" if self.bot.auto_trading else "INAKTIV"
        print(f"🤖 Auto-Trading: {status}")
        
        # Hauptschleife
        print("⏰ Bot läuft im Headless-Modus. Drücke Ctrl+C zum Beenden.")
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Bot wird beendet...")

def main():
    print("🚀 Starte KuCoin Trading Bot - Optimiert für Raspberry Pi (Headless)...")
    
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
    
    # Bot initialisieren
    bot = KuCoinTradingBot(
        api_key=API_KEY,
        api_secret=API_SECRET,
        api_passphrase=API_PASSPHRASE,
        sandbox=SANDBOX
    )
    
    # Headless Modus starten
    headless_bot = HeadlessTradingBot(bot)
    headless_bot.run()

if __name__ == "__main__":
    main()