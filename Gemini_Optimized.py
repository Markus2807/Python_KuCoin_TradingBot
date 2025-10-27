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

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
import schedule
from datetime import datetime, timedelta
import hmac
import hashlib
import base64
import json
import requests
from urllib.parse import urlencode
import csv
from collections import defaultdict, deque
import math
import random


# =============================================================================
# ENV LOADER
# =============================================================================

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


# =============================================================================
# API CLIENT (KuCoinAPI)
# =============================================================================

class KuCoinAPI:
    def __init__(self, api_key='', api_secret='', api_passphrase='', sandbox=False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase
        self.base_url = 'https://api-sandbox.kucoin.com' if sandbox else 'https://api.kucoin.com'
        self.request_count = 0
        self.last_request_time = None
        self.time_diff = 0  # Zeit-Synchronisations-Differenz in ms
        
        # Symbol-Informationen für Order-Validierung
        self.symbols_info = {}
        
        print(f"✅ KuCoin API initialisiert - Modus: {'SANDBOX' if sandbox else 'LIVE'}")
        self.get_symbols_info()
        self._sync_time()

    def _get_signature(self, endpoint, method, params_or_body):
        """Erzeugt die KuCoin API Signatur"""
        timestamp = str(int(time.time() * 1000) + self.time_diff)
        
        if method == 'GET' and params_or_body:
            query_string = urlencode(params_or_body)
            signature_string = f"{timestamp}{method}{endpoint}?{query_string}"
        elif method == 'POST':
            signature_string = f"{timestamp}{method}{endpoint}{params_or_body}"
        else: # GET ohne Params oder DELETE
            signature_string = f"{timestamp}{method}{endpoint}"

        # Hash SHA256 des Signatur-Strings
        hashed = hmac.new(self.api_secret.encode('utf-8'), signature_string.encode('utf-8'), hashlib.sha256).digest()
        signature = base64.b64encode(hashed).decode('utf-8')

        return timestamp, signature

    def _make_request(self, method, endpoint, params=None, body=None):
        """Führt einen API-Call aus"""
        url = f"{self.base_url}{endpoint}"
        
        if method == 'POST' and body:
            body_str = json.dumps(body)
        else:
            body_str = ''

        timestamp, signature = self._get_signature(endpoint, method, params if method == 'GET' and params else body_str)
        
        headers = {
            'KC-API-KEY': self.api_key,
            'KC-API-SIGN': signature,
            'KC-API-TIMESTAMP': timestamp,
            'KC-API-PASSPHRASE': base64.b64encode(hmac.new(self.api_secret.encode('utf-8'), self.api_passphrase.encode('utf-8'), hashlib.sha256).digest()).decode('utf-8'),
            'KC-API-KEY-VERSION': '2'
        }
        
        if method == 'POST':
            headers['Content-Type'] = 'application/json'

        try:
            # Rate Limit (KuCoin erlaubt ~100 Anfragen/3s)
            if self.last_request_time and (time.time() - self.last_request_time) < 0.05:
                time.sleep(0.05)
                
            response = requests.request(method, url, headers=headers, params=params if method == 'GET' else None, data=body_str if method == 'POST' else None, timeout=10)
            self.last_request_time = time.time()
            self.request_count += 1
            
            response.raise_for_status() # Löst HTTPError für 4xx/5xx Fehler aus
            return response.json()
            
        except requests.exceptions.HTTPError as errh:
            print(f"❌ Http Error: {errh}")
            if response is not None:
                 print(f"   Response: {response.text}")
            return None
        except requests.exceptions.ConnectionError as errc:
            print(f"❌ Error Connecting: {errc}")
            return None
        except requests.exceptions.Timeout as errt:
            print(f"❌ Timeout Error: {errt}")
            return None
        except requests.exceptions.RequestException as err:
            print(f"❌ General Request Error: {err}")
            return None
        except Exception as e:
            print(f"❌ Unbekannter Fehler in _make_request: {e}")
            return None

    def _sync_time(self):
        """Synchronisiert lokale Zeit mit KuCoin Server-Zeit."""
        try:
            response = requests.get(f"{self.base_url}/api/v1/timestamp", timeout=10)
            if response.status_code == 200:
                kucoin_time = int(response.json()['data'])
                local_time = int(time.time() * 1000)
                self.time_diff = kucoin_time - local_time
                print(f"⏰ Zeit-Synchronisation abgeschlossen. Differenz: {self.time_diff}ms")
            else:
                print("⚠️  Zeit-Synchronisation fehlgeschlagen. Verwende lokale Zeit.")
        except Exception as e:
            print(f"⚠️  Fehler bei Zeit-Synchronisation: {e}")

    def get_symbols_info(self):
        """Holt Informationen über alle Handels-Paare für Order-Validierung"""
        data = self._make_request('GET', '/api/v1/symbols')
        if data and data['code'] == '200000':
            for symbol in data['data']:
                try:
                    self.symbols_info[symbol['symbol']] = {
                        'base_increment': float(symbol['baseIncrement']),
                        'quote_increment': float(symbol['quoteIncrement']),
                        'base_min_size': float(symbol['baseMinSize']),
                        'base_max_size': float(symbol['baseMaxSize'])
                    }
                except Exception as e:
                    print(f"Fehler beim Parsen der Symbol-Info für {symbol.get('symbol', 'Unbekannt')}: {e}")
            print(f"✅ {len(self.symbols_info)} Symbole geladen und validiert.")
        return self.symbols_info

    def get_kline(self, symbol, interval, limit=100):
        """Holt historische Kerzendaten (K-Lines)"""
        params = {
            'symbol': symbol,
            'type': interval,
            'limit': limit
        }
        data = self._make_request('GET', '/api/v1/market/candles', params=params)
        
        if data and data['code'] == '200000' and data['data']:
            # Konvertiert [time, open, close, high, low, volume, amount] in eine Liste von Dictionaries
            klines = []
            for kline in data['data']:
                try:
                    klines.append({
                        'time': datetime.fromtimestamp(int(kline[0])).strftime('%Y-%m-%d %H:%M:%S'),
                        'open': float(kline[1]),
                        'close': float(kline[2]),
                        'high': float(kline[3]),
                        'low': float(kline[4]),
                        'volume': float(kline[5]),
                        'amount': float(kline[6]),
                    })
                except (ValueError, IndexError):
                    print(f"⚠️  Fehler beim Parsen einer Kline für {symbol}: {kline}")
            return klines
        return None

    def get_account_balance(self, currency=None):
        """Holt das Kontoguthaben"""
        endpoint = '/api/v1/accounts'
        params = {'currency': currency} if currency else None
        data = self._make_request('GET', endpoint, params=params)
        
        if data and data['code'] == '200000':
            # Rückgabe einer Liste von Konten
            return data['data']
        else:
            error_msg = data.get('msg', 'Unbekannter Fehler') if data else 'Keine Verbindung'
            print(f"❌ Balance API Fehler: {error_msg}")
            return None

    def place_limit_order(self, symbol, side, amount, price):
        """Platziert eine Limit-Order"""
        # Konvertiere amount und price in Strings, um Genauigkeitsfehler zu vermeiden
        body = {
            'clientOid': str(int(time.time() * 1000)),
            'side': side.lower(), # 'buy' oder 'sell'
            'symbol': symbol,
            'type': 'limit',
            'size': str(amount),
            'price': str(price),
            'remark': 'AutoTrade'
        }

        data = self._make_request('POST', '/api/v1/orders', body)
        
        if data and data['code'] == '200000':
            print(f"✅ Order ({side}) erfolgreich platziert für {symbol}: {amount} @ {price}")
            return data['data']
        else:
            error_msg = data.get('msg', 'Unbekannter Fehler') if data else 'Keine Verbindung'
            print(f"❌ Order API Fehler für {symbol}: {error_msg} | Details: {data.get('data')}")
            return None

    def get_ticker(self, symbol):
        """Holt aktuellen Preis"""
        params = {'symbol': symbol}
        data = self._make_request('GET', '/api/v1/market/orderbook/level1', params=params)
        
        if data and data['code'] == '200000':
            try:
                return float(data['data']['price'])
            except (ValueError, TypeError):
                return None
        else:
            error_msg = data.get('msg', 'Unbekannter Fehler') if data else 'Keine Verbindung'
            print(f"❌ Ticker API Fehler für {symbol}: {error_msg}")
            return None

    def test_connection(self):
        """Testet die API-Verbindung"""
        print("🔍 Teste API-Verbindung...")
        balance = self.get_account_balance()
        if balance:
            print("✅ API-Verbindung erfolgreich!")
            return True
        else:
            print("❌ API-Verbindung fehlgeschlagen!")
            return False

    def get_api_stats(self):
        """Gibt API-Statistiken zurück"""
        return {
            'request_count': self.request_count,
            'last_request_time': datetime.fromtimestamp(self.last_request_time).strftime('%H:%M:%S') if self.last_request_time else 'N/A'
        }


# =============================================================================
# TAX LOGGER (TaxLogger)
# =============================================================================

class TaxLogger:
    """Klasse für Finanzamt-konforme Protokollierung aller Trades"""
    
    def __init__(self, log_directory="trade_logs"):
        self.log_directory = log_directory
        self.setup_logging()
        
    def setup_logging(self):
        """Erstellt Log-Verzeichnis und Dateien"""
        if not os.path.exists(self.log_directory):
            os.makedirs(self.log_directory)
            
        trade_log_path = os.path.join(self.log_directory, "trades_finanzamt.csv")
        if not os.path.exists(trade_log_path):
            with open(trade_log_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow([
                    'Datum_Uhrzeit', 'Typ', 'Symbol', 'Menge', 'Preis_pro_Einheit',
                    'Gesamtbetrag', 'Gebuehren', 'Netto_Betrag', 'Gewinn_Verlust',
                    'Gewinn_Verlust_prozent', 'Handelsgrund', 'Order_ID', 'Portfolio_Wert'
                ])
                
        self.csv_log_path = trade_log_path
        self.json_log_path = os.path.join(self.log_directory, "trading_history.json")
        if not os.path.exists(self.json_log_path):
            with open(self.json_log_path, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=4)

    def log_trade(self, trade_data):
        """Loggt einen abgeschlossenen Trade in CSV und JSON"""
        
        # 1. Logge in CSV für Finanzamt
        try:
            with open(self.csv_log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow([
                    trade_data['timestamp'],
                    trade_data['type'],
                    trade_data['symbol'],
                    f"{trade_data['amount']:.8f}",
                    f"{trade_data['price']:.8f}",
                    f"{trade_data['total_value']:.8f}",
                    f"{trade_data['fees']:.8f}",
                    f"{trade_data['net_amount']:.8f}",
                    f"{trade_data['profit_loss']:.8f}",
                    f"{trade_data['profit_loss_percent']:.8f}",
                    trade_data['reason'],
                    trade_data['order_id'],
                    f"{trade_data['portfolio_value']:.2f}"
                ])
        except Exception as e:
            print(f"❌ Fehler beim CSV-Logging: {e}")

        # 2. Logge in JSON für GUI-Anzeige
        try:
            history = []
            if os.path.exists(self.json_log_path):
                with open(self.json_log_path, 'r', encoding='utf-8') as f:
                    # Stellt sicher, dass die Datei nicht leer ist
                    content = f.read()
                    if content:
                        history = json.loads(content)
                    
            history.append(trade_data)
            
            with open(self.json_log_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=4)
                
        except Exception as e:
            print(f"❌ Fehler beim JSON-Logging: {e}")

    def get_recent_trades(self, limit=50):
        """Holt die neuesten Trades aus der JSON-Datei"""
        try:
            if not os.path.exists(self.json_log_path):
                return []
                
            with open(self.json_log_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content:
                    return []
                history = json.loads(content)
            
            # Neueste zuerst (letzte Einträge in der Liste)
            return history[-limit:]
        except Exception as e:
            print(f"❌ Fehler beim Lesen der Trade-Historie: {e}")
            # Erstellt eine leere Datei, falls die aktuelle korrupt ist
            self.setup_logging() 
            return []


# =============================================================================
# TRADING BOT (KuCoinTradingBot)
# =============================================================================

class KuCoinTradingBot:
    def __init__(self, api_key, api_secret, api_passphrase, sandbox=False):
        self.api = KuCoinAPI(api_key, api_secret, api_passphrase, sandbox)
        self.tax_logger = TaxLogger()
        self.gui = None # Wird von der GUI gesetzt
        self.active_trades = {} # {'BTC-USDT': {'buy_price': 10000, ...}}
        self.current_recommendations = {}
        
        # Trading Einstellungen
        self.auto_trading = False 
        self.stop_loss_percent = 2.0
        self.trade_size_percent = 10.0
        self.max_open_trades = 5
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.backtest_interval = '15min'
        
        # Konfigurierbare Kryptowährungen
        self.trading_pairs = ['BTC-USDT', 'ETH-USDT', 'ADA-USDT', 'DOT-USDT', 'LINK-USDT']
        
        # Cache & Scheduling
        self.price_cache = {}
        self.balance_cache = None
        self.last_update = None
        self.next_scheduled_update = None
        self.last_trade_time = None
        self.update_lock = threading.Lock()
        
        self.load_active_trades()
        self.setup_scheduler()
        
        # Prüfe API-Verbindung beim Start
        self.api.test_connection()
        
    def set_gui_reference(self, gui):
        """Setzt die Referenz zur GUI für Updates"""
        self.gui = gui
        
    def update_bot_activity(self, message):
        """Sendet Aktivitätsmeldung an Konsole und GUI"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        if self.gui:
            self.gui.root.after(0, self.gui.add_activity_log, log_message)

    def setup_scheduler(self):
        """Richtet den automatisierten Update-Job ein"""
        schedule.every(2).minutes.do(self.update_caches)
        schedule.every(5).minutes.do(self.run_trading_logic)
        self.update_bot_activity("✅ Scheduler initialisiert.")

    def update_caches(self):
        """Aktualisiert alle Caches (Preis und Balance)"""
        with self.update_lock:
            self.update_bot_activity("🔄 Starte Cache-Update...")
            
            # 1. Price Cache (Parallel für Geschwindigkeit)
            price_threads = []
            for symbol in self.trading_pairs:
                t = threading.Thread(target=self._fetch_price_and_store, args=(symbol,))
                price_threads.append(t)
                t.start()
            for t in price_threads:
                t.join() 
            self.update_bot_activity(f"📈 Price Cache aktualisiert ({len(self.price_cache)} Paare).")
                
            # 2. Balance Cache
            self._fetch_balance_and_store()
            self.update_bot_activity("💰 Balance Cache aktualisiert.")
            
            self.last_update = datetime.now()
            self.next_scheduled_update = schedule.next_run()
            
    def _fetch_price_and_store(self, symbol):
        """Hilfsfunktion zum parallelen Abrufen des Tickers"""
        price = self.api.get_ticker(symbol)
        if price:
            self.price_cache[symbol] = price
            
    def _fetch_balance_and_store(self):
        """Holt Kontostände und berechnet Gesamtwert"""
        accounts = self.api.get_account_balance()
        if accounts:
            usdt_balance = 0.0
            total_portfolio_value = 0.0
            asset_details = []
            
            # Sammle alle Assets (nicht nur die mit Balance > 0)
            for account in accounts:
                currency = account['currency']
                balance = float(account['balance'])
                
                # Finde den aktuellen Preis in USDT
                current_price_usd = 1.0 # Standardwert für USDT
                if currency != 'USDT':
                    symbol = f"{currency}-USDT"
                    if symbol in self.price_cache:
                        current_price_usd = self.price_cache[symbol]
                    elif currency in self.price_cache: # Nur falls es der Basis-Asset ist
                         current_price_usd = self.price_cache[currency]
                    else:
                        # Versuche Ticker direkt abzurufen (nur als Fallback)
                        current_price_usd = self.api.get_ticker(symbol) or 0.0

                value_usd = balance * current_price_usd
                total_portfolio_value += value_usd
                
                if currency == 'USDT':
                    usdt_balance = balance
                
                asset_details.append({
                    'asset': currency,
                    'balance': balance,
                    'value_usd': value_usd,
                    'current_price': current_price_usd
                })
            
            # Zweiter Durchlauf, um Prozentanteile zu berechnen
            for asset in asset_details:
                asset['share'] = (asset['value_usd'] / total_portfolio_value) * 100 if total_portfolio_value > 0 else 0.0
                
            self.balance_cache = {
                'usdt_balance': usdt_balance,
                'total_portfolio_value': total_portfolio_value,
                'asset_details': asset_details
            }
        else:
            self.balance_cache = None

    def get_balance_summary(self):
        """Gibt die gespeicherte Balance-Zusammenfassung zurück"""
        if self.balance_cache is None:
            # Versuche, den Cache einmalig zu füllen
            self._fetch_balance_and_store()
        return self.balance_cache

    def calculate_indicators(self, klines):
        """Berechnet RSI und EMA basierend auf den Kerzendaten"""
        
        # Vereinfachte Berechnung für RSI (14 Perioden)
        if not klines or len(klines) < 15:
            return {'rsi': None, 'last_close': None}
            
        close_prices = [kline['close'] for kline in klines]
        
        # RSI Berechnung
        gains = []
        losses = []
        for i in range(1, len(close_prices)):
            change = close_prices[i] - close_prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
                
        # Durchschnittlicher Gain/Loss über 14 Perioden (Erster Smoothed Average)
        avg_gain = sum(gains[:14]) / 14 if len(gains) >= 14 else 0
        avg_loss = sum(losses[:14]) / 14 if len(losses) >= 14 else 0
        
        # Smoothed RSI
        for i in range(14, len(gains)):
            # Smoothed Average
            avg_gain = (avg_gain * 13 + gains[i]) / 14
            avg_loss = (avg_loss * 13 + losses[i]) / 14
            
        rs = avg_gain / avg_loss if avg_loss > 0 else (100 if avg_gain > 0 else 0)
        rsi = 100 - (100 / (1 + rs)) if rs is not None else None

        return {
            'rsi': rsi,
            'last_close': close_prices[-1]
        }

    def generate_signal(self, symbol, indicators):
        """Generiert ein Handelssignal (BUY/SELL) basierend auf RSI"""
        
        rsi = indicators['rsi']
        
        # Priorität: Aktiven Trade schließen (Sell)
        if symbol in self.active_trades:
            # Trailing Stop Loss / Take Profit Logik
            buy_price = self.active_trades[symbol]['buy_price']
            current_price = self.price_cache.get(symbol)
            if not current_price:
                return {'current_signal': 'HOLD', 'confidence': 50}

            # Stop Loss
            stop_loss_price = buy_price * (1 - self.stop_loss_percent / 100)
            if current_price <= stop_loss_price:
                return {'current_signal': 'SELL_SL', 'confidence': 100} # SL = Stop Loss
                
            # Take Profit (Wenn RSI überverkauft ist, nimm Gewinn mit)
            if rsi and rsi >= self.rsi_overbought:
                return {'current_signal': 'SELL_TP', 'confidence': 90} # TP = Take Profit
                
            return {'current_signal': 'HOLD_ACTIVE', 'confidence': 75}
            
        # Priorität: Neuen Trade eröffnen (Buy)
        else:
            if rsi is None:
                return {'current_signal': 'NONE', 'confidence': 0}
            
            if rsi <= self.rsi_oversold:
                # RSI Oversold ist ein starkes BUY Signal
                confidence = 100 - rsi
                return {'current_signal': 'BUY', 'confidence': confidence}
                
            elif rsi >= self.rsi_overbought:
                # RSI Overbought ist ein starkes SELL Signal (Kurzfristig)
                confidence = rsi - 100
                return {'current_signal': 'SELL_SHORTTERM', 'confidence': confidence}
                
            else:
                return {'current_signal': 'HOLD', 'confidence': 50}
                

    def run_complete_backtest(self):
        """Führt eine vollständige technische Analyse für alle Paare durch"""
        self.update_bot_activity("📊 Starte vollständigen Backtest...")
        results = {}
        for symbol in self.trading_pairs:
            try:
                klines = self.api.get_kline(symbol, self.backtest_interval, limit=200)
                if klines:
                    indicators = self.calculate_indicators(klines)
                    signal = self.generate_signal(symbol, indicators)
                    
                    results[symbol] = {
                        'indicators': indicators,
                        'current_signal': signal['current_signal'],
                        'confidence': signal['confidence'],
                        'last_price': self.price_cache.get(symbol)
                    }
                    
                else:
                    self.update_bot_activity(f"⚠️  Keine Daten für {symbol} verfügbar.")
            except Exception as e:
                self.update_bot_activity(f"❌ Fehler im Backtest für {symbol}: {e}")
                
        self.current_recommendations = results
        self.update_bot_activity(f"✅ Backtest abgeschlossen. {len(results)} Kryptos analysiert.")
        
        # Aktualisiere GUI-Elemente
        if self.gui:
            self.gui.root.after(0, self.gui.update_recommendations)
            
        return results
        
    def get_recommendations(self):
        """Gibt die aktuellen Handelssignale zurück"""
        return self.current_recommendations

    def run_trading_logic(self):
        """Kernlogik: Prüft Trades und führt sie aus"""
        
        if self.update_lock.locked():
            self.update_bot_activity("⚠️  Überspringe Trading-Logik: Cache-Update läuft noch.")
            return

        self.update_bot_activity("🤖 Starte Trading-Logik...")
        
        # 1. Backtest durchführen (um aktuelle Signale zu erhalten)
        self.run_complete_backtest()
        
        # 2. Prüfe aktive Trades auf Stop-Loss/Take-Profit
        self._check_and_close_trades()
        
        # 3. Führe neue Trades aus
        if self.auto_trading:
            self._execute_new_trades()
        else:
            self.update_bot_activity("❌ Auto-Trading ist inaktiv. Keine neuen Trades ausgeführt.")
            
        self.save_active_trades()
        
        # Aktualisiere GUI-Elemente
        if self.gui:
            self.gui.root.after(0, self.gui.update_active_trades)


    def _check_and_close_trades(self):
        """Prüft aktive Trades auf Sell-Signale (SL/TP)"""
        trades_to_close = []
        for symbol, trade in list(self.active_trades.items()):
            recommendation = self.current_recommendations.get(symbol)
            if recommendation:
                signal = recommendation['current_signal']
                
                if 'SELL' in signal:
                    current_price = self.price_cache.get(symbol)
                    if current_price:
                        # Berechne P/L
                        buy_price = trade['buy_price']
                        amount = trade['amount']
                        profit_loss = (current_price - buy_price) * amount
                        profit_loss_percent = (current_price / buy_price - 1) * 100
                        
                        reason = "Stop Loss" if 'SL' in signal else "Take Profit" if 'TP' in signal else "RSI Sell"
                        
                        self.update_bot_activity(f"🔴 Verkaufs-Signal ({reason}) für {symbol}! P/L: ${profit_loss:+.2f} ({profit_loss_percent:+.2f}%)")
                        
                        # Führe Verkauf aus (Hier: Simuliere Verkauf, da Limit Order verwendet wird)
                        # In einem echten Bot müsste hier die Sell Limit Order platziert werden.
                        
                        # Simuliere Trade-Schließung und Logging
                        log_data = {
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'type': 'SELL',
                            'symbol': symbol,
                            'amount': amount,
                            'price': current_price,
                            'total_value': amount * current_price,
                            'fees': (amount * current_price) * 0.001, # Simuliere 0.1% Gebühr
                            'net_amount': (amount * current_price) * (1 - 0.001),
                            'profit_loss': profit_loss,
                            'profit_loss_percent': profit_loss_percent,
                            'handelsgrund': reason,
                            'order_id': f"SIM-SELL-{int(time.time()*100)}",
                            'portfolio_value': self.get_balance_summary()['total_portfolio_value']
                        }
                        self.tax_logger.log_trade(log_data)
                        
                        trades_to_close.append(symbol)
                        self.update_bot_activity(f"✅ Trade geschlossen: {symbol} | Grund: {reason}")
                        
        for symbol in trades_to_close:
            del self.active_trades[symbol]


    def _execute_new_trades(self):
        """Führt Kauf-Trades aus, wenn die Bedingungen erfüllt sind."""
        
        if len(self.active_trades) >= self.max_open_trades:
            self.update_bot_activity(f"⚠️  Max. Trades ({self.max_open_trades}) erreicht. Kaufe nicht.")
            return

        usdt_balance = self.get_balance_summary()['usdt_balance']
        if usdt_balance <= 10:
            self.update_bot_activity("❌ USDT Balance zu gering für neue Trades.")
            return
            
        usdt_to_spend = usdt_balance * (self.trade_size_percent / 100)

        # Führe Trades basierend auf den Empfehlungen aus
        for symbol, data in self.current_recommendations.items():
            if symbol not in self.active_trades and 'BUY' in data['current_signal'] and data['confidence'] >= 70:
                
                if len(self.active_trades) >= self.max_open_trades:
                    self.update_bot_activity(f"⚠️  Max. Trades ({self.max_open_trades}) erreicht. Stoppe Kauf-Versuche.")
                    break
                    
                current_price = self.price_cache.get(symbol)
                if not current_price:
                    self.update_bot_activity(f"❌ Aktueller Preis für {symbol} nicht gefunden. Kaufe nicht.")
                    continue
                    
                base_asset, quote_asset = symbol.split('-')
                
                # Berechne die Menge
                raw_amount = usdt_to_spend / current_price
                
                # Prüfe KuCoin Order-Regeln für die Mindestmenge und Inkremente
                if symbol not in self.api.symbols_info:
                    self.update_bot_activity(f"❌ Symbol-Info für {symbol} nicht geladen. Kaufe nicht.")
                    continue
                    
                symbol_info = self.api.symbols_info[symbol]
                
                # Runden auf das korrekte Inkrement
                if symbol_info['base_increment'] > 0:
                    valid_amount = math.floor(raw_amount / symbol_info['base_increment']) * symbol_info['base_increment']
                else:
                    valid_amount = raw_amount
                    
                if valid_amount < symbol_info['base_min_size']:
                    self.update_bot_activity(f"❌ {symbol}: Berechnete Menge ({valid_amount:.4f}) ist kleiner als Mindestgröße ({symbol_info['base_min_size']}). Kaufe nicht.")
                    continue

                # PLATZIERE DEN TRADE (Simuliert)
                # In einem echten Bot: order_id = self.api.place_limit_order(symbol, 'BUY', valid_amount, current_price)
                order_id = f"SIM-BUY-{int(time.time()*100)}"
                
                if order_id:
                    self.active_trades[symbol] = {
                        'order_id': order_id,
                        'buy_price': current_price,
                        'amount': valid_amount,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'status': 'OPEN'
                    }
                    
                    # Logge den Kauf-Trade (Simuliert)
                    log_data = {
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'type': 'BUY',
                        'symbol': symbol,
                        'amount': valid_amount,
                        'price': current_price,
                        'total_value': valid_amount * current_price,
                        'fees': (valid_amount * current_price) * 0.001, # Simuliere 0.1% Gebühr
                        'net_amount': valid_amount * current_price * (1 + 0.001),
                        'profit_loss': 0.0,
                        'profit_loss_percent': 0.0,
                        'handelsgrund': f"Auto-Trade: {data['current_signal']}",
                        'order_id': order_id,
                        'portfolio_value': self.get_balance_summary()['total_portfolio_value']
                    }
                    self.tax_logger.log_trade(log_data)
                    
                    self.update_bot_activity(f"🟢 Trade eröffnet: {symbol} - {valid_amount:.4f} @ ${current_price:.2f}")
                    
                    # Warte eine kurze Zeit, um Rate-Limits zu vermeiden
                    time.sleep(1)


    def load_active_trades(self):
        """Lädt aktive Trades aus einer Datei (für Persistenz)"""
        file_path = os.path.join(self.tax_logger.log_directory, "active_trades.json")
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.active_trades = json.load(f)
                self.update_bot_activity(f"✅ {len(self.active_trades)} aktive Trades geladen.")
            except Exception as e:
                self.update_bot_activity(f"❌ Fehler beim Laden der aktiven Trades: {e}")
                self.active_trades = {}
        else:
            self.update_bot_activity("ℹ️ Keine gespeicherten aktiven Trades gefunden.")
            
    def save_active_trades(self):
        """Speichert aktive Trades in einer Datei"""
        file_path = os.path.join(self.tax_logger.log_directory, "active_trades.json")
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.active_trades, f, indent=4)
        except Exception as e:
            self.update_bot_activity(f"❌ Fehler beim Speichern der aktiven Trades: {e}")


# =============================================================================
# GUI (TradingBotGUI)
# =============================================================================

class TradingBotGUI:
    def __init__(self, bot):
        self.bot = bot
        self.bot.set_gui_reference(self)
        
        self.root = tk.Tk()
        self.root.title("KuCoin Trading Bot - Gemini Optimized")
        
        # Style für ein modernes Aussehen (Blue/Dark Theme)
        self.style = ttk.Style()
        self.style.theme_create("dark_style", parent="alt", settings={
            "TNotebook": {"configure": {"tabmargins": [2, 5, 2, 0], "background": '#2c3e50'}},
            "TNotebook.Tab": {"configure": {"padding": [15, 5], "background": '#34495e', "foreground": 'white'}, 
                              "map": {"background": [("selected", '#3498db')], "foreground": [("selected", 'white')]}},
            "TFrame": {"configure": {"background": '#ecf0f1'}},
            "TLabel": {"configure": {"background": '#ecf0f1', "foreground": '#2c3e50', "font": ('Arial', 10)}},
            "TButton": {"configure": {"background": '#3498db', "foreground": 'white', "padding": 6, "font": ('Arial', 10, 'bold')},
                        "map": {"background": [("active", '#2980b9')]}},
        })
        self.style.theme_use("dark_style")

        # Ermittle Bildschirmauflösung
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        
        # Fenstergröße setzen
        self.root.geometry("1400x900")
            
        # Aktivitätslog
        self.bot_activity_log = deque(maxlen=200) # Speichert die letzten 200 Einträge
        self.activity_log_text = None # ScrolledText Widget

        # Status Variable
        self.status_var = tk.StringVar(value="Bot initialisiert - Bereit")
        
        # Konfigurations-Variablen (müssen vor setup_config_tab initialisiert werden)
        self.stop_loss_var = tk.StringVar(value=str(self.bot.stop_loss_percent))
        self.trade_size_var = tk.StringVar(value=str(self.bot.trade_size_percent))
        self.rsi_oversold_var = tk.StringVar(value=str(self.bot.rsi_oversold))
        self.rsi_overbought_var = tk.StringVar(value=str(self.bot.rsi_overbought))
        self.max_trades_var = tk.StringVar(value=str(self.bot.max_open_trades))
        self.interval_var = tk.StringVar(value=self.bot.backtest_interval)
        self.auto_trading_var = tk.BooleanVar(value=self.bot.auto_trading)
        
        # Balance Variablen
        self.usdt_balance_var = tk.StringVar(value="USDT Balance: Wird geladen...")
        self.portfolio_value_var = tk.StringVar(value="Gesamt-Portfolio Wert: Wird geladen...")
        
        self.setup_gui()
        self.start_auto_updates()

    def setup_gui(self):
        # Haupt-Frame und Notebook (Tabs)
        main_frame = ttk.Frame(self.root, padding="5 5 5 5")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Statusleiste
        status_frame = ttk.Frame(main_frame, padding="5 5 5 5")
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        ttk.Label(status_frame, text="Status:", font=('Arial', 10, 'bold'), background='#ecf0f1').pack(side=tk.LEFT)
        ttk.Label(status_frame, textvariable=self.status_var, background='#ecf0f1').pack(side=tk.LEFT, padx=5)

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tabs erstellen
        self.tab_trading = ttk.Frame(self.notebook, padding="10")
        self.tab_config = ttk.Frame(self.notebook, padding="10")
        self.tab_monitoring = ttk.Frame(self.notebook, padding="10")
        self.tab_tax = ttk.Frame(self.notebook, padding="10")

        self.notebook.add(self.tab_trading, text="Trading Übersicht")
        self.notebook.add(self.tab_config, text="Konfiguration")
        self.notebook.add(self.tab_monitoring, text="Bot Monitoring")
        self.notebook.add(self.tab_tax, text="Finanzamt (History)")

        # Tab-Inhalte einrichten
        self.setup_trading_tab()
        self.setup_config_tab()
        self.setup_monitoring_tab()
        self.setup_tax_tab()

    def add_activity_log(self, message):
        """Fügt einen Eintrag zum Aktivitätslog (GUI und Deque) hinzu"""
        self.bot_activity_log.append(message)
        if self.activity_log_text:
            self.activity_log_text.config(state='normal')
            self.activity_log_text.insert(tk.END, message + "\n")
            self.activity_log_text.see(tk.END) # Scrolle nach unten
            self.activity_log_text.config(state='disabled')
            
    def update_status(self, message):
        """Aktualisiert die Status-Anzeige"""
        self.status_var.set(message)
        print(f"Status: {message}")

    # =========================================================================
    # TAB 1: TRADING ÜBERSICHT
    # =========================================================================
    
    def setup_trading_tab(self):
        # Frame für Kontostand und Knöpfe (Oben)
        top_frame = ttk.Frame(self.tab_trading)
        top_frame.pack(fill=tk.X, pady=(0, 10))

        # Kontostand Anzeigen
        ttk.Label(top_frame, textvariable=self.usdt_balance_var, font=('Arial', 12, 'bold')).pack(side=tk.LEFT, padx=10)
        ttk.Label(top_frame, textvariable=self.portfolio_value_var, font=('Arial', 12, 'bold')).pack(side=tk.LEFT, padx=10)

        # Knöpfe
        ttk.Button(top_frame, text="Quick Check", command=self.quick_check, style='TButton').pack(side=tk.RIGHT, padx=5)
        ttk.Button(top_frame, text="Voller Backtest", command=self.run_backtest_thread, style='TButton').pack(side=tk.RIGHT, padx=5)

        # Aktive Trades & Empfehlungen Container (Mitte)
        middle_frame = ttk.Frame(self.tab_trading)
        middle_frame.pack(fill=tk.BOTH, expand=True)
        
        # Asset Details Treeview (Links)
        self.asset_frame = ttk.LabelFrame(middle_frame, text="Asset Details", padding="10")
        self.asset_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.setup_asset_tree(self.asset_frame)
        
        # Recommendations Treeview (Mitte)
        self.rec_frame = ttk.LabelFrame(middle_frame, text="Trading Signale", padding="10")
        self.rec_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.setup_recommendations_tree(self.rec_frame)
        
        # Aktive Trades Treeview (Rechts)
        self.active_frame = ttk.LabelFrame(middle_frame, text="Aktive Trades", padding="10")
        # FIX: LEFT muss tk.LEFT sein
        self.active_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.setup_active_trades_tree(self.active_frame)

    def setup_asset_tree(self, parent_frame):
        self.asset_tree = ttk.Treeview(parent_frame, columns=("Balance", "Value_USD", "Share"), show='headings')
        self.asset_tree.heading("Balance", text="Balance")
        self.asset_tree.heading("Value_USD", text="Wert (USD)")
        self.asset_tree.heading("Share", text="Anteil")
        self.asset_tree.column("Balance", width=100, anchor=tk.E)
        self.asset_tree.column("Value_USD", width=100, anchor=tk.E)
        self.asset_tree.column("Share", width=80, anchor=tk.E)
        self.asset_tree.pack(fill=tk.BOTH, expand=True)
        
    def setup_recommendations_tree(self, parent_frame):
        self.rec_tree = ttk.Treeview(parent_frame, columns=("Signal", "Confidence", "Price"), show='headings')
        self.rec_tree.heading("Signal", text="Signal")
        self.rec_tree.heading("Confidence", text="Confidence")
        self.rec_tree.heading("Price", text="Preis")
        self.rec_tree.column("Signal", width=100, anchor=tk.CENTER)
        self.rec_tree.column("Confidence", width=80, anchor=tk.CENTER)
        self.rec_tree.column("Price", width=100, anchor=tk.E)
        self.rec_tree.pack(fill=tk.BOTH, expand=True)

    def setup_active_trades_tree(self, parent_frame):
        self.trade_tree = ttk.Treeview(parent_frame, columns=("Price", "Amount", "P/L", "P/L%"), show='headings')
        self.trade_tree.heading("Price", text="Kaufpreis")
        self.trade_tree.heading("Amount", text="Menge")
        self.trade_tree.heading("P/L", text="P/L ($)")
        self.trade_tree.heading("P/L%", text="P/L (%)")
        self.trade_tree.column("Price", width=90, anchor=tk.E)
        self.trade_tree.column("Amount", width=90, anchor=tk.E)
        self.trade_tree.column("P/L", width=80, anchor=tk.E)
        self.trade_tree.column("P/L%", width=80, anchor=tk.E)
        self.trade_tree.pack(fill=tk.BOTH, expand=True)

    # =========================================================================
    # TAB 2: KONFIGURATION
    # =========================================================================
    
    def setup_config_tab(self):
        config_frame = ttk.Frame(self.tab_config, padding="20")
        config_frame.pack(fill=tk.BOTH, expand=True)
        
        # Titel
        ttk.Label(config_frame, text="Bot Konfiguration", font=('Arial', 16, 'bold')).grid(row=0, column=0, columnspan=2, pady=(0, 20), sticky=tk.W)

        # Auto-Trading Checkbox
        self.auto_trading_check = ttk.Checkbutton(config_frame, text="Auto-Trading aktivieren", 
                                                  variable=self.auto_trading_var, command=self.toggle_auto_trading)
        self.auto_trading_check.grid(row=1, column=0, columnspan=2, pady=(0, 15), sticky=tk.W)

        # Stop Loss
        ttk.Label(config_frame, text="Stop Loss (%):").grid(row=2, column=0, pady=5, sticky=tk.W)
        self.sl_entry = ttk.Entry(config_frame, textvariable=self.stop_loss_var)
        self.sl_entry.grid(row=2, column=1, pady=5, sticky=tk.EW)
        
        # Trade Size
        ttk.Label(config_frame, text="Trade Größe (% des Kapitals):").grid(row=3, column=0, pady=5, sticky=tk.W)
        self.ts_entry = ttk.Entry(config_frame, textvariable=self.trade_size_var)
        self.ts_entry.grid(row=3, column=1, pady=5, sticky=tk.EW)
        
        # Max Open Trades
        ttk.Label(config_frame, text="Max. Offene Trades:").grid(row=4, column=0, pady=5, sticky=tk.W)
        self.mt_entry = ttk.Entry(config_frame, textvariable=self.max_trades_var)
        self.mt_entry.grid(row=4, column=1, pady=5, sticky=tk.EW)
        
        # RSI Oversold
        ttk.Label(config_frame, text="RSI Oversold Level:").grid(row=5, column=0, pady=5, sticky=tk.W)
        self.rsi_os_entry = ttk.Entry(config_frame, textvariable=self.rsi_oversold_var)
        self.rsi_os_entry.grid(row=5, column=1, pady=5, sticky=tk.EW)
        
        # RSI Overbought
        ttk.Label(config_frame, text="RSI Overbought Level:").grid(row=6, column=0, pady=5, sticky=tk.W)
        self.rsi_ob_entry = ttk.Entry(config_frame, textvariable=self.rsi_overbought_var)
        self.rsi_ob_entry.grid(row=6, column=1, pady=5, sticky=tk.EW)
        
        # Backtest Interval
        ttk.Label(config_frame, text="Backtest Interval:").grid(row=7, column=0, pady=5, sticky=tk.W)
        intervals = ['1min', '5min', '15min', '30min', '1h', '4h', '1day']
        self.interval_menu = ttk.Combobox(config_frame, textvariable=self.interval_var, values=intervals, state='readonly')
        self.interval_menu.grid(row=7, column=1, pady=5, sticky=tk.EW)
        
        # Speichern Knopf
        ttk.Button(config_frame, text="Konfiguration Speichern und Anwenden", command=self.save_config, style='TButton').grid(row=8, column=0, columnspan=2, pady=20, sticky=tk.EW)

        # Konfigurationsspalte strecken
        config_frame.grid_columnconfigure(1, weight=1)

    def toggle_auto_trading(self):
        self.bot.auto_trading = self.auto_trading_var.get()
        self.update_status("Auto-Trading: " + ("AKTIV" if self.bot.auto_trading else "INAKTIV"))
        
    def save_config(self):
        """Speichert die GUI-Einstellungen in der Bot-Logik"""
        try:
            self.bot.stop_loss_percent = float(self.stop_loss_var.get())
            self.bot.trade_size_percent = float(self.trade_size_var.get())
            self.bot.max_open_trades = int(self.max_trades_var.get())
            self.bot.rsi_oversold = int(self.rsi_oversold_var.get())
            self.bot.rsi_overbought = int(self.rsi_overbought_var.get())
            self.bot.backtest_interval = self.interval_var.get()
            self.bot.auto_trading = self.auto_trading_var.get()
            
            self.update_status("✅ Konfiguration erfolgreich gespeichert und angewendet.")
            messagebox.showinfo("Erfolg", "Bot-Einstellungen wurden aktualisiert.")
            
        except ValueError:
            self.update_status("❌ Fehler beim Speichern der Konfiguration: Ungültiger Wert.")
            messagebox.showerror("Fehler", "Bitte geben Sie gültige numerische Werte ein.")

    # =========================================================================
    # TAB 3: BOT MONITORING
    # =========================================================================
    
    def setup_monitoring_tab(self):
        # ScrolledText für das Aktivitätslog
        log_label = ttk.Label(self.tab_monitoring, text="Bot Aktivitäts-Log:", font=('Arial', 12, 'bold'))
        log_label.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))
        
        self.activity_log_text = scrolledtext.ScrolledText(self.tab_monitoring, wrap=tk.WORD, state='disabled', height=20, 
                                                           bg='#2c3e50', fg='white', font=('Consolas', 9))
        self.activity_log_text.pack(fill=tk.BOTH, expand=True)

        # Fülle das Log mit den initialen Nachrichten (falls vorhanden)
        for line in self.bot_activity_log:
            self.activity_log_text.config(state='normal')
            self.activity_log_text.insert(tk.END, line + "\n")
            self.activity_log_text.config(state='disabled')

    # =========================================================================
    # TAB 4: FINANZAMT (HISTORY)
    # =========================================================================
    
    def setup_tax_tab(self):
        # Frame für Knöpfe
        button_frame = ttk.Frame(self.tab_tax)
        button_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        
        ttk.Label(button_frame, text="Trade Historie:", font=('Arial', 12, 'bold')).pack(side=tk.LEFT, padx=10)
        
        ttk.Button(button_frame, text="Steuer-Report (CSV) Generieren", command=self.generate_tax_report, style='TButton').pack(side=tk.RIGHT, padx=5)
        
        # ScrolledText für die Trade-Logs
        self.tax_log_text = scrolledtext.ScrolledText(self.tab_tax, wrap=tk.NONE, state='disabled', height=20, 
                                                      bg='#ecf0f1', fg='#2c3e50', font=('Consolas', 9))
        self.tax_log_text.pack(fill=tk.BOTH, expand=True)
        
        # Füllt den Log beim Start
        self.update_tax_log()

    def generate_tax_report(self):
        """Erstellt eine CSV-Datei mit der gesamten Trade-Historie"""
        # Die CSV-Datei wird bereits automatisch durch den TaxLogger verwaltet
        file_path = self.bot.tax_logger.csv_log_path
        
        if os.path.exists(file_path):
            self.update_status("✅ Steuer-Report-Datei ist bereit.")
            messagebox.showinfo("Steuer-Report", f"Die Datei mit der gesamten Trade-Historie ist bereit unter:\n{file_path}")
        else:
            self.update_status("❌ Steuer-Report-Datei nicht gefunden.")
            messagebox.showerror("Fehler", "Die CSV-Datei mit der Trade-Historie konnte nicht gefunden werden.")


    # =========================================================================
    # DATA UPDATE LOGIC
    # =========================================================================

    # --- Trading Tab Updates ---
    
    def update_balance_display(self):
        """Aktualisiert die Kontostands-Anzeige und das Asset-Treeview."""
        # Holen Sie die zuletzt abgerufene Zusammenfassung
        balance_summary = self.bot.get_balance_summary()
        
        if balance_summary is None:
            self.usdt_balance_var.set("USDT: Fehler!")
            self.portfolio_value_var.set("Portfolio: Fehler!")
            return

        usdt_balance = balance_summary.get('usdt_balance', 0.0)
        total_portfolio_value = balance_summary.get('total_portfolio_value', 0.0)
        asset_details = balance_summary.get('asset_details', [])

        self.usdt_balance_var.set(f"USDT Balance: ${usdt_balance:,.2f}")
        self.portfolio_value_var.set(f"Gesamt-Portfolio Wert: ${total_portfolio_value:,.2f}")
        
        # Asset Treeview aktualisieren
        if self.asset_tree:
            # Lösche alte Einträge
            for item in self.asset_tree.get_children():
                self.asset_tree.delete(item)
                
            for asset in asset_details:
                # FIX: Zeige nur Assets an, die nicht USDT sind UND einen signifikanten Wert haben
                if asset['asset'] == 'USDT' or asset['balance'] < 0.00000001:
                    continue

                # Hier verwenden wir asset['balance'] für die Menge und asset['value_usd'] für den USD-Wert
                self.asset_tree.insert('', tk.END, text=asset['asset'], 
                                        values=(f"{asset['balance']:,.4f}", f"${asset['value_usd']:,.2f}", f"{asset['share']:.2f}%"))
                                        
    def update_recommendations(self):
        """Aktualisiert die Empfehlungs-Tabelle"""
        if self.rec_tree:
            for item in self.rec_tree.get_children():
                self.rec_tree.delete(item)
                
            recommendations = self.bot.get_recommendations()
            
            for symbol, data in recommendations.items():
                signal_text = f"{data['current_signal']}"
                confidence = f"{data['confidence']:.0f}%"
                last_price = f"{data['last_price']:,.4f}" if data['last_price'] else 'N/A'
                
                # Tagging für farbliche Hervorhebung (optional)
                tag = 'buy' if 'BUY' in data['current_signal'] else 'sell' if 'SELL' in data['current_signal'] else ''
                
                self.rec_tree.insert('', tk.END, text=symbol, tags=(tag,),
                                     values=(signal_text, confidence, last_price))

            # Style-Konfiguration für Tags (muss im GUI-Setup initialisiert sein, hier vereinfacht)
            self.rec_tree.tag_configure('buy', background='#e6ffe6', foreground='green')
            self.rec_tree.tag_configure('sell', background='#ffe6e6', foreground='red')

    def update_active_trades(self):
        """Aktualisiert die aktiven Trades"""
        if self.trade_tree:
            for item in self.trade_tree.get_children():
                self.trade_tree.delete(item)
            
            # Hole aktuelle Preise
            current_prices = self.bot.price_cache
            
            for symbol, trade in self.bot.active_trades.items():
                current_price = current_prices.get(symbol)
                buy_price = trade['buy_price']
                amount = trade['amount']
                
                pl_usd = 0.0
                pl_percent = 0.0
                
                if current_price:
                    pl_usd = (current_price - buy_price) * amount
                    pl_percent = (current_price / buy_price - 1) * 100
                
                tag = 'positive' if pl_usd >= 0 else 'negative'
                
                self.trade_tree.insert('', tk.END, text=symbol, tags=(tag,),
                                       values=(f"{buy_price:,.4f}", f"{amount:,.4f}", f"{pl_usd:+.2f}", f"{pl_percent:+.2f}%"))
                                       
            self.trade_tree.tag_configure('positive', foreground='green')
            self.trade_tree.tag_configure('negative', foreground='red')

    # --- Tax Tab Update ---

    def update_tax_log(self):
        """Lädt und zeigt die letzten Trade-Logs an (behebt KeyError und SyntaxError)"""
        trades = self.bot.tax_logger.get_recent_trades(limit=50)
        
        if self.tax_log_text:
            self.tax_log_text.config(state='normal')
            self.tax_log_text.delete('1.0', tk.END)
            
            if not trades:
                self.tax_log_text.insert(tk.END, "Keine Trade-Historie gefunden.\n")
            else:
                # Header mit fester Breite
                self.tax_log_text.insert(tk.END, f"{'Datum':<20}{'Typ':<10}{'Symbol':<15}{'Preis':<15}{'Menge':<15}{'Gewinn/Verlust':<20}\n")
                self.tax_log_text.insert(tk.END, "="*95 + "\n")
                
                for trade in trades:
                    # Robuste Schlüssel-Abfrage mit .get() und Fallback
                    try:
                        timestamp = trade.get('timestamp', 'N/A')
                        trade_type = trade.get('type', 'N/A')
                        symbol = trade.get('symbol', 'N/A')
                        price = trade.get('price', 0.0)
                        amount = trade.get('amount', 0.0)
                        profit_loss = trade.get('profit_loss', 0.0)
                        profit_loss_percent = trade.get('profit_loss_percent', 0.0)
                        
                        # FIX: Korrigierte f-string Syntax und Formatierung
                        line = (
                            f"{timestamp:<20}"
                            f"{trade_type:<10}"
                            f"{symbol:<15}"
                            f"{price:.4f:<15}"
                            f"{amount:.4f:<15}"
                            f"{profit_loss:+.2f} ({profit_loss_percent:+.2f}%)"
                            "\n"
                        )
                        self.tax_log_text.insert(tk.END, line)
                        
                    except Exception as e:
                        # Loggt fehlerhafte Einträge und ignoriert sie
                        self.tax_log_text.insert(tk.END, f"❌ Fehlerhafter Log-Eintrag übersprungen.\n")
                        print(f"❌ Fehler beim Verarbeiten des Trade-Logs: {e} | Daten: {trade}")
                    
            self.tax_log_text.config(state='disabled')


    # =========================================================================
    # THREADING & SCHEDULER
    # =========================================================================
    
    def run_backtest_thread(self):
        """Startet den Backtest in einem separaten Thread"""
        def run():
            self.update_status("Starte vollständigen Backtest...")
            results = self.bot.run_complete_backtest()
            
            if results:
                buy_signals = sum(1 for data in results.values() if 'BUY' in data['current_signal'])
                self.root.after(0, self.update_status, f"Backtest abgeschlossen: {buy_signals} Kaufsignale gefunden.")
            else:
                self.root.after(0, self.update_status, "Backtest fehlgeschlagen - Keine Ergebnisse")
                messagebox.showerror("Fehler", "Backtest konnte keine Daten abrufen!")
                
        threading.Thread(target=run, daemon=True).start()
        
    def quick_check(self):
        """Führt ein schnelles Cache-Update und Trading-Logik aus"""
        def run():
            self.update_status("Starte Quick Check (Cache Update & Trading Logik)...")
            # Nur Cache aktualisieren und Trading Logik laufen lassen
            self.bot.update_caches()
            self.bot.run_trading_logic()
            self.root.after(0, self.update_status, "Quick Check abgeschlossen.")
            
        threading.Thread(target=run, daemon=True).start()
        
    def start_auto_updates(self):
        """Startet automatische Updates im Hintergrund"""
        def update_loop():
            while True:
                try:
                    # Führe geplante Scheduler-Jobs aus
                    schedule.run_pending()
                    
                    # Führe alle 15 Sekunden GUI-Updates aus (Polling-Interval)
                    self.root.after(0, self.update_balance_display)
                    self.root.after(0, self.update_recommendations)
                    self.root.after(0, self.update_active_trades)
                    self.root.after(0, self.update_tax_log)
                    
                    # Aktualisiere den allgemeinen Status mit Bot-Informationen
                    last_update_str = self.bot.last_update.strftime('%H:%M:%S') if self.bot.last_update else 'N/A'
                    next_run = self.bot.next_scheduled_update.strftime('%H:%M:%S') if self.bot.next_scheduled_update else 'N/A'
                    self.root.after(0, self.update_status, 
                                    f"Bot läuft | Auto-Trade: {'AKTIV' if self.bot.auto_trading else 'INAKTIV'} | Letztes Update: {last_update_str} | Nächster Run: {next_run}")
                                    
                except Exception as e:
                    print(f"❌ Auto-update error: {e}")
                    
                time.sleep(15) # Kurzes GUI Polling-Interval
                
        # Starte den Thread
        threading.Thread(target=update_loop, daemon=True).start()
        
    def run(self):
        """Startet die Tkinter Mainloop"""
        self.root.mainloop()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("🚀 Starte KuCoin Trading Bot...")
    
    # Lade API-Daten
    env_vars = load_env_file()
    
    API_KEY = env_vars.get('KUCOIN_API_KEY')
    API_SECRET = env_vars.get('KUCOIN_API_SECRET')
    API_PASSPHRASE = env_vars.get('KUCOIN_API_PASSPHRASE')
    SANDBOX = env_vars.get('KUCOIN_SANDBOX', 'False').lower() == 'true'
    
    if not all([API_KEY, API_SECRET, API_PASSPHRASE]):
        print("❌ Fehler: API-Daten nicht gefunden!")
        print("💡 Bitte erstelle eine .env oder api.env Datei mit den Zugangsdaten.")
        # Simuliere Demo-Bot, wenn API-Daten fehlen (nur für GUI-Test)
        # API_KEY = "demo_key"
        # API_SECRET = "demo_secret"
        # API_PASSPHRASE = "demo_passphrase"
        # SANDBOX = True
        # print("⚠️  Starte im Demo-Modus mit Dummy-API-Keys.")
        # if not all([API_KEY, API_SECRET, API_PASSPHRASE]):
        #     exit()
        exit()
    
    print(f"✅ API-Daten geladen - Sandbox Modus: {SANDBOX}")
    
    try:
        # Initialisiere Bot und API
        bot = KuCoinTradingBot(
            api_key=API_KEY,
            api_secret=API_SECRET,
            api_passphrase=API_PASSPHRASE,
            sandbox=SANDBOX
        )
        
        print("🎨 Starte GUI (Optimiert für Windows/Linux)...")
        # Führe das erste Cache Update direkt aus, bevor die GUI startet
        bot.update_caches() 
        
        gui = TradingBotGUI(bot)
        print("✅ GUI erfolgreich gestartet")
        gui.run()
        
    except Exception as e:
        print(f"❌ Kritischer Fehler beim Starten: {e}")
        print("🛑 Bot wird beendet.")