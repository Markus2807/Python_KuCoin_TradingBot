import os
import warnings
warnings.filterwarnings('ignore')

# Raspberry Pi spezifische Einstellungen
import platform
if platform.system() == "Linux" and 'raspberrypi' in platform.uname().release.lower():
    os.environ['DISPLAY'] = ':0'
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

# =============================================================================
# API CLIENT
# =============================================================================

class KuCoinAPI:
    def __init__(self, api_key='', api_secret='', api_passphrase='', sandbox=False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase
        self.base_url = 'https://api-sandbox.kucoin.com' if sandbox else 'https://api.kucoin.com'
        self.request_count = 0
        self.last_request_time = None
        
        # Symbol-Informationen für Order-Validierung
        self.symbols_info = {}
        
        print(f"✅ KuCoin API initialisiert - Modus: {'SANDBOX' if sandbox else 'LIVE'}")
        
    def get_symbols_info(self):
        """Holt Informationen über alle Handels-Paare für Order-Validierung"""
        data = self._make_request('GET', '/api/v1/symbols')
        if data and data['code'] == '200000':
            for symbol in data['data']:
                try:
                    base_increment = float(symbol['baseIncrement']) if symbol['baseIncrement'] else 0.0
                    quote_increment = float(symbol['quoteIncrement']) if symbol['quoteIncrement'] else 0.0
                    base_min_size = float(symbol['baseMinSize']) if symbol['baseMinSize'] else 0.0
                    base_max_size = float(symbol['baseMaxSize']) if symbol['baseMaxSize'] else 0.0
                    price_increment = float(symbol['priceIncrement']) if symbol['priceIncrement'] else 0.0
                    
                    min_funds = symbol.get('minFunds')
                    if min_funds is not None:
                        min_funds = float(min_funds)
                    else:
                        min_funds = 0.0
                    
                    self.symbols_info[symbol['symbol']] = {
                        'baseIncrement': base_increment,
                        'quoteIncrement': quote_increment,
                        'baseMinSize': base_min_size,
                        'baseMaxSize': base_max_size,
                        'priceIncrement': price_increment,
                        'minFunds': min_funds
                    }
                except (ValueError, TypeError) as e:
                    continue
                    
            return True
        else:
            return False
    
    def validate_order_size(self, symbol, size, price=None):
        """Validiert die Order-Größe gemäß KuCoin's Anforderungen"""
        if symbol not in self.symbols_info:
            if not self.get_symbols_info():
                return False, "Konnte Symbol-Informationen nicht laden"
        
        symbol_info = self.symbols_info.get(symbol)
        if not symbol_info:
            return False, f"Unbekanntes Symbol: {symbol}"
        
        # Prüfe Mindestgröße
        if size < symbol_info['baseMinSize']:
            return False, f"Größe zu klein. Minimum: {symbol_info['baseMinSize']}"
        
        # Prüfe Maximale Größe
        if size > symbol_info['baseMaxSize']:
            return False, f"Größe zu groß. Maximum: {symbol_info['baseMaxSize']}"
        
        # Prüfe Inkrement (Step-Größe)
        base_increment = symbol_info['baseIncrement']
        if base_increment > 0:
            steps = size / base_increment
            if not steps.is_integer():
                valid_size = round(round(steps) * base_increment, 8)
                return False, f"Ungültige Schrittgröße. Verwende: {valid_size}"
        
        # Prüfe Mindestbetrag wenn Preis gegeben
        if price and symbol_info['minFunds'] > 0:
            order_value = size * price
            if order_value < symbol_info['minFunds']:
                return False, f"Order-Wert zu klein. Minimum: {symbol_info['minFunds']} USDT"
        
        return True, "Validierung erfolgreich"
    
    def calculate_valid_size(self, symbol, desired_size):
        """Berechnet eine gültige Order-Größe basierend auf den Symbol-Regeln"""
        if symbol not in self.symbols_info:
            self.get_symbols_info()
        
        symbol_info = self.symbols_info.get(symbol)
        if not symbol_info:
            return desired_size
        
        base_increment = symbol_info['baseIncrement']
        base_min_size = symbol_info['baseMinSize']
        
        if desired_size < base_min_size:
            desired_size = base_min_size
        
        if base_increment > 0:
            steps = desired_size / base_increment
            valid_steps = round(steps)
            valid_size = round(valid_steps * base_increment, 8)
            
            if valid_size < base_min_size:
                valid_size = base_min_size
            
            return valid_size
        
        return desired_size

    def _generate_signature(self, timestamp, method, endpoint, body=''):
        try:
            str_to_sign = f"{timestamp}{method}{endpoint}{body}"
            signature = base64.b64encode(
                hmac.new(
                    self.api_secret.encode('utf-8'),
                    str_to_sign.encode('utf-8'),
                    hashlib.sha256
                ).digest()
            )
            return signature.decode('utf-8')
        except Exception as e:
            return None
    
    def _get_headers(self, method, endpoint, body=''):
        try:
            timestamp = str(self._get_kucoin_timestamp())
            
            signature = self._generate_signature(timestamp, method, endpoint, body)
            
            if not signature:
                return None
                
            passphrase_signature = base64.b64encode(
                hmac.new(
                    self.api_secret.encode('utf-8'),
                    self.api_passphrase.encode('utf-8'),
                    hashlib.sha256
                ).digest()
            ).decode('utf-8')
            
            headers = {
                'KC-API-KEY': self.api_key,
                'KC-API-SIGN': signature,
                'KC-API-TIMESTAMP': timestamp,
                'KC-API-PASSPHRASE': passphrase_signature,
                'KC-API-KEY-VERSION': '2',
                'Content-Type': 'application/json'
            }
            return headers
        except Exception as e:
            return None
    
    def _make_request(self, method, endpoint, body='', params=None, retry_count=0):
        """Macht API-Request mit Rate-Limiting"""
        self.request_count += 1
        self.last_request_time = datetime.now()
        
        if params:
            query_string = urlencode(params)
            full_endpoint = f"{endpoint}?{query_string}"
        else:
            full_endpoint = endpoint
            
        headers = self._get_headers(method, full_endpoint, body)
        
        if not headers:
            return None
            
        url = f"{self.base_url}{full_endpoint}"
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=30)
            elif method == 'POST':
                response = requests.post(url, headers=headers, json=json.loads(body) if body else None, timeout=30)
            else:
                return None
                
            data = response.json()
            
            if response.status_code == 400 and data.get('code') == '400003':
                if retry_count < 3:
                    time.sleep(1)
                    return self._make_request(method, endpoint, body, params, retry_count + 1)
            
            return data
            
        except requests.exceptions.Timeout:
            return None
        except requests.exceptions.ConnectionError:
            return None
        except Exception as e:
            return None
    
    def get_klines(self, symbol, interval, start_time=None, end_time=None):
        """Holt historische Kursdaten"""
        interval_map = {
            '1min': '1min', '5min': '5min', '15min': '15min',
            '1hour': '1hour', '4hour': '4hour', '1day': '1day', '1week': '1week'
        }
        
        kucoin_interval = interval_map.get(interval, '1hour')
        params = {'symbol': symbol, 'type': kucoin_interval}
        
        if start_time:
            params['startAt'] = int(start_time.timestamp())
        if end_time:
            params['endAt'] = int(end_time.timestamp())
            
        data = self._make_request('GET', '/api/v1/market/candles', params=params)
        
        if data and data['code'] == '200000' and data['data']:
            klines = data['data']
            klines.reverse()
            
            processed_data = []
            for kline in klines:
                try:
                    processed_data.append({
                        'timestamp': datetime.fromtimestamp(int(kline[0])),
                        'open': float(kline[1]),
                        'close': float(kline[2]),
                        'high': float(kline[3]),
                        'low': float(kline[4]),
                        'volume': float(kline[5]),
                        'turnover': float(kline[6])
                    })
                except (ValueError, IndexError):
                    continue
            
            return processed_data
        else:
            return None
    
    def get_account_balance(self):
        """Holt Kontostand"""
        data = self._make_request('GET', '/api/v1/accounts')
        
        if data and data['code'] == '200000':
            return data['data']
        else:
            return None

    def get_account_balances_detailed(self):
        """Holt detaillierte Kontostände aller Assets"""
        data = self._make_request('GET', '/api/v1/accounts')
        
        if data and data['code'] == '200000':
            balances = []
            for account in data['data']:
                if account['type'] == 'trade' and float(account['balance']) > 0:
                    balances.append({
                        'currency': account['currency'],
                        'balance': float(account['balance']),
                        'available': float(account['available']),
                        'holds': float(account['holds']),
                        'type': account['type']
                    })
            return balances
        else:
            return None
    
    def place_order(self, symbol, side, order_type, size, price=None):
        """Platziert eine Order mit automatischer Validierung"""
        is_valid, validation_msg = self.validate_order_size(symbol, size, price)
        if not is_valid:
            corrected_size = self.calculate_valid_size(symbol, size)
            if corrected_size != size:
                size = corrected_size
            else:
                return None
        
        body = {
            'clientOid': str(int(time.time() * 1000)),
            'side': side,
            'symbol': symbol,
            'type': order_type,
            'size': str(size)
        }
        
        if price and order_type == 'limit':
            body['price'] = str(price)
        
        body_str = json.dumps(body)
        data = self._make_request('POST', '/api/v1/orders', body_str)
        
        if data and data['code'] == '200000':
            return data['data']
        else:
            return None

    def get_ticker(self, symbol):
        """Holt aktuellen Preis"""
        params = {'symbol': symbol}
        data = self._make_request('GET', '/api/v1/market/orderbook/level1', params=params)
        
        if data and data['code'] == '200000':
            return float(data['data']['price'])
        else:
            return None

    def test_connection(self):
        """Testet die API-Verbindung"""
        self.get_symbols_info()
        balance = self.get_account_balance()
        return balance is not None

    def get_api_stats(self):
        """Gibt API-Statistiken zurück"""
        return {
            'request_count': self.request_count,
            'last_request_time': self.last_request_time.strftime('%H:%M:%S') if self.last_request_time else '-'
        }
    
    def _get_kucoin_timestamp(self):
        """Holt den aktuellen Timestamp von KuCoin Server für Synchronisation"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/timestamp", timeout=10)
            if response.status_code == 200:
                return int(response.json()['data'])
            else:
                return int(time.time() * 1000)
        except Exception:
            return int(time.time() * 1000)

# =============================================================================
# TAX LOGGER
# =============================================================================

class TaxLogger:
    """Klasse für Finanzamt-konforme Protokollierung aller Trades"""
    
    def __init__(self, log_directory="trade_logs"):
        self.log_directory = log_directory
        self.csv_log_path = os.path.join(self.log_directory, "trades_finanzamt.csv")
        self.json_log_path = os.path.join(self.log_directory, "trading_history.json")
        self.setup_logging()
        
    def setup_logging(self):
        """Erstellt Log-Verzeichnis und Dateien"""
        if not os.path.exists(self.log_directory):
            os.makedirs(self.log_directory)
            
        # CSV Datei erstellen falls nicht vorhanden
        if not os.path.exists(self.csv_log_path):
            with open(self.csv_log_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow([
                    'Datum_Uhrzeit', 'Typ', 'Symbol', 'Menge', 'Preis_pro_Einheit',
                    'Gesamtbetrag', 'Gebuehren', 'Netto_Betrag', 'Gewinn_Verlust',
                    'Gewinn_Verlust_prozent', 'Handelsgrund', 'Order_ID', 'Portfolio_Wert'
                ])
                
        # JSON Datei erstellen falls nicht vorhanden
        if not os.path.exists(self.json_log_path):
            with open(self.json_log_path, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=2, ensure_ascii=False)
            print("✅ Neue Trading-History JSON Datei erstellt")
    
    def log_trade(self, trade_data):
        """Protokolliert einen Trade für das Finanzamt"""
        timestamp = datetime.now().strftime('%d.%m.%Y %H:%M:%S')
        
        # Logge in alle Formate
        self._log_to_csv(timestamp, trade_data)
        trade_record = self._log_to_json(timestamp, trade_data)
        
        return trade_record
    
    def _log_to_csv(self, timestamp, trade_data):
        """Loggt Trade in CSV Format für Finanzamt"""
        try:
            with open(self.csv_log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=';')
                
                # Berechne Werte
                total_value = trade_data['amount'] * trade_data['price']
                fees = total_value * 0.001  # 0.1% Gebühren
                net_amount = total_value - fees
                profit_loss = trade_data.get('profit_loss', 0)
                profit_loss_percent = trade_data.get('profit_loss_percent', 0)
                portfolio_value = trade_data.get('portfolio_value', 0)
                
                writer.writerow([
                    timestamp,
                    trade_data['side'],
                    trade_data['symbol'],
                    f"{trade_data['amount']:.8f}",
                    f"{trade_data['price']:.8f}",
                    f"{total_value:.2f}",
                    f"{fees:.2f}",
                    f"{net_amount:.2f}",
                    f"{profit_loss:.2f}",
                    f"{profit_loss_percent:.2f}",
                    trade_data.get('reason', ''),
                    trade_data.get('order_id', ''),
                    f"{portfolio_value:.2f}"
                ])
                
            print(f"✅ Trade in CSV geloggt: {trade_data['symbol']} {trade_data['side']}")
            
        except Exception as e:
            print(f"❌ Fehler beim CSV-Logging: {e}")
    
    def _log_to_json(self, timestamp, trade_data):
        """Loggt Trade in JSON Format"""
        try:
            # Lade bestehende History
            if os.path.exists(self.json_log_path):
                with open(self.json_log_path, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            else:
                history = []
                
            # Berechne Werte
            total_value = trade_data['amount'] * trade_data['price']
            fees = total_value * 0.001  # 0.1% Gebühren
            
            # Erstelle Trade-Record
            trade_record = {
                'timestamp': timestamp,
                'timestamp_iso': datetime.now().isoformat(),
                'trade_id': f"TRADE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{trade_data['symbol']}",
                'side': trade_data['side'],
                'symbol': trade_data['symbol'],
                'amount': float(trade_data['amount']),
                'price': float(trade_data['price']),
                'total_value': float(total_value),
                'fees': float(fees),
                'reason': trade_data.get('reason', ''),
                'profit_loss': trade_data.get('profit_loss', 0),
                'profit_loss_percent': trade_data.get('profit_loss_percent', 0),
                'order_id': trade_data.get('order_id', ''),
                'portfolio_value': trade_data.get('portfolio_value', 0)
            }
            
            # Füge neuen Trade hinzu
            history.append(trade_record)
            
            # Speichere zurück
            with open(self.json_log_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Trade in JSON geloggt: {trade_data['symbol']} {trade_data['side']}")
            return trade_record
            
        except Exception as e:
            print(f"❌ Fehler beim JSON-Logging: {e}")
            return None
    
    def get_recent_trades(self, limit=100):
        """Holt die neuesten Trades aus der JSON-Historie"""
        try:
            # Prüfe ob Datei existiert
            if not os.path.exists(self.json_log_path):
                print("ℹ️  Keine JSON-History Datei gefunden")
                return []
            
            # Lade History
            with open(self.json_log_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            if not history:
                print("ℹ️  JSON-History ist leer")
                return []
            
            # Sortiere nach Zeitstempel (neueste zuerst)
            recent_trades = sorted(history, key=lambda x: x.get('timestamp_iso', ''), reverse=True)
            
            print(f"✅ {len(recent_trades)} Trades aus JSON-History geladen")
            return recent_trades[:limit]
            
        except json.JSONDecodeError:
            print("❌ Fehler: JSON-History Datei ist korrupt")
            return []
        except Exception as e:
            print(f"❌ Fehler beim Laden der JSON-History: {e}")
            return []

# =============================================================================
# TRADING BOT
# =============================================================================

class KuCoinTradingBot:
    def __init__(self, api_key, api_secret, api_passphrase, sandbox=False):
        self.api = KuCoinAPI(api_key, api_secret, api_passphrase, sandbox)
        self.tax_logger = TaxLogger()
        self.active_trades = {}
        self.trade_history = []  # Wird gleich geladen
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
        self.trading_pairs = ['BTC-USDT', 'ETH-USDT', 'XRP-USDT', 'DOGE-USDT', 'SOL-USDT', 'SHIB-USDT', 'BNB-USDT']
        
        # Cache
        self.price_cache = {}
        self.balance_cache = None
        self.last_update = None
        self.next_scheduled_update = None
        self.last_trade_time = None
        
        # Performance Tracking
        self.gui_reference = None
        self.headless_reference = None
        
        # Lade Trade-History SOFORT beim Start
        print("🔄 Lade Trade-History beim Start...")
        self.load_trade_history()
        
        print(f"✅ KuCoin Trading Bot initialisiert - Sandbox: {sandbox}")
        print(f"📊 Trade-History: {len(self.trade_history)} Trades geladen")
        
    def load_trade_history(self):
        """Lädt Trade-History aus dem Tax-Logger - KORRIGIERT"""
        try:
            recent_trades = self.tax_logger.get_recent_trades(1000)  # Letzte 1000 Trades
            
            # DEBUG: Ausgabe zur Überprüfung
            print(f"🔍 Lade Trade-History: {len(recent_trades)} Trades gefunden")
            if recent_trades:
                for i, trade in enumerate(recent_trades[:3]):  # Zeige erste 3 Trades
                    print(f"  Trade {i+1}: {trade.get('symbol')} {trade.get('side')} {trade.get('timestamp')}")
            
            self.trade_history = recent_trades
            print(f"✅ {len(self.trade_history)} Trades aus History geladen")
            
        except Exception as e:
            print(f"❌ Fehler beim Laden der Trade-History: {e}")
            self.trade_history = []
        
    def set_gui_reference(self, gui):
        self.gui_reference = gui
        
    def set_headless_reference(self, headless):
        self.headless_reference = headless
        
    def update_bot_activity(self, message):
        """Aktualisiert Bot-Aktivität ohne übermäßige Debug-Ausgaben"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        
        # Nur wichtige Nachrichten in der Konsole ausgeben
        if any(keyword in message for keyword in ['✅', '❌', '⚡', '📊', '🎯']):
            print(log_entry)
        
        if self.gui_reference:
            self.gui_reference.update_bot_activity(log_entry)
        elif self.headless_reference:
            self.headless_reference.update_bot_activity(log_entry)
    
    def set_trading_pairs(self, pairs):
        """Setzt die zu handelnden Kryptowährungen"""
        self.trading_pairs = pairs
        self.update_bot_activity(f"📊 Trading-Pairs aktualisiert: {', '.join(pairs)}")
    
    def get_available_pairs(self):
        """Gibt verfügbare Trading-Pairs von KuCoin zurück"""
        try:
            if not self.api.symbols_info:
                self.api.get_symbols_info()
            symbols_info = self.api.symbols_info
            usdt_pairs = [symbol for symbol in symbols_info.keys() if symbol.endswith('-USDT')]
            return sorted(usdt_pairs)
        except Exception as e:
            return  ['BTC-USDT', 'ETH-USDT', 'XRP-USDT', 'DOGE-USDT', 'SOL-USDT', 'SHIB-USDT', 'BNB-USDT']
    
    def calculate_rsi(self, prices, period=14):
        """Berechnet RSI ohne pandas"""
        if len(prices) < period + 1:
            return 50
            
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        if len(gains) < period:
            return 50
            
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100 if avg_gain > 0 else 50
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return round(rsi, 2)
    
    def calculate_moving_average(self, prices, period):
        """Berechnet gleitenden Durchschnitt ohne pandas"""
        if len(prices) < period:
            return sum(prices) / len(prices) if prices else 0
        return sum(prices[-period:]) / period
    
    def get_historical_data(self, symbol, interval='15min', limit=100):
        """Holt echte historische Daten von KuCoin"""
        try:
            klines_data = self.api.get_klines(symbol, interval)
            
            if klines_data:
                prices = [kline['close'] for kline in klines_data]
                return prices[-limit:]
            else:
                return None
                
        except Exception:
            return None
    
    def analyze_crypto(self, symbol):
        """Analysiert Kryptowährung mit technischen Indikatoren"""
        try:
            current_price = self.get_current_price(symbol)
            if not current_price:
                return None
                
            historical_data = self.get_historical_data(symbol, self.backtest_interval, 50)
            if not historical_data:
                return None
                
            rsi = self.calculate_rsi(historical_data)
            ma_short = self.calculate_moving_average(historical_data, 10)
            ma_long = self.calculate_moving_average(historical_data, 20)
            
            signals = []
            confidence = 50
            
            # RSI Signale
            if rsi < self.rsi_oversold:
                signals.append("RSI Oversold")
                confidence += 25
            elif rsi > self.rsi_overbought:
                signals.append("RSI Overbought") 
                confidence -= 25
                
            # Moving Average Signale
            if ma_short > ma_long:
                signals.append("MA Bullish")
                confidence += 15
            else:
                signals.append("MA Bearish")
                confidence -= 15
                
            # Trend-Analyse
            price_trend = current_price > historical_data[-5] if len(historical_data) >= 5 else True
            if price_trend:
                signals.append("Aufwärtstrend")
                confidence += 10
            else:
                signals.append("Abwärtstrend")
                confidence -= 10
                
            # Bestimme endgültiges Signal
            if confidence >= 70:
                current_signal = "STRONG_BUY"
            elif confidence >= 60:
                current_signal = "BUY"
            elif confidence <= 30:
                current_signal = "STRONG_SELL"
            elif confidence <= 40:
                current_signal = "SELL"
            else:
                current_signal = "HOLD"
                
            return {
                'symbol': symbol,
                'current_price': current_price,
                'current_signal': current_signal,
                'confidence': max(0, min(100, confidence)),
                'signals': signals,
                'rsi': rsi,
                'ma_short': ma_short,
                'ma_long': ma_long,
                'total_return': ((current_price - historical_data[0]) / historical_data[0]) * 100 if historical_data else 0
            }
            
        except Exception:
            return None
    
    def quick_signal_check(self):
        """Schnelle Signalprüfung für konfigurierte Kryptos"""
        try:
            self.update_bot_activity("⚡ Starte schnelle Signalprüfung...")
            
            results = {}
            for crypto in self.trading_pairs:
                analysis = self.analyze_crypto(crypto)
                if analysis:
                    results[crypto] = analysis
                    
            self.current_recommendations = results
            return results
            
        except Exception as e:
            self.update_bot_activity(f"❌ Fehler bei schneller Signalprüfung: {e}")
            return {}
    
    def run_complete_backtest(self, pairs=None, execute_trades=False):
        """Führt vollständigen Backtest durch und optional auch Trades"""
        try:
            self.update_bot_activity("📊 Starte Backtest...")
            
            cryptos = pairs if pairs else self.trading_pairs
            
            results = {}
            for crypto in cryptos:
                analysis = self.analyze_crypto(crypto)
                if analysis:
                    results[crypto] = analysis
                    
            self.current_recommendations = results
            self.last_update = datetime.now()
            self.next_scheduled_update = self.last_update + timedelta(minutes=15)
            
            if execute_trades and self.auto_trading:
                self.execute_trades_based_on_signals(results)
            
            self.update_bot_activity(f"✅ Backtest abgeschlossen - {len(results)} Kryptos analysiert")
            return results
            
        except Exception as e:
            self.update_bot_activity(f"❌ Backtest Fehler: {e}")
            return {}
        
    def execute_trades_based_on_signals(self, results):
        """Führt automatisch Trades basierend auf Backtest-Ergebnissen aus"""
        if not self.auto_trading:
            return
            
        if not results:
            return
            
        executed_trades = 0
        for crypto, data in results.items():
            if "BUY" in data['current_signal'] and data['confidence'] >= 70:
                success = self.execute_trade(crypto, data['current_signal'])
                if success:
                    executed_trades += 1
        
        self.update_bot_activity(f"📊 Insgesamt {executed_trades} Trades ausgeführt")
    
    def get_balance_summary(self):
        """Gibt echte Kontostand-Übersicht zurück"""
        try:
            balances = self.api.get_account_balances_detailed()
            
            if not balances:
                return None
            
            assets = []
            total_value = 0
            
            for balance in balances:
                currency = balance['currency']
                if currency == 'USDT':
                    price = 1.0
                    value_usd = balance['available']
                else:
                    symbol = f"{currency}-USDT"
                    price = self.get_current_price(symbol)
                    value_usd = balance['available'] * price if price else 0
                
                if value_usd > 0.01:
                    assets.append({
                        'currency': currency,
                        'balance': balance['balance'],
                        'available': balance['available'],
                        'price': price,
                        'value_usd': value_usd,
                        'percentage': 0
                    })
                    total_value += value_usd
            
            for asset in assets:
                asset['percentage'] = (asset['value_usd'] / total_value) * 100 if total_value > 0 else 0
            
            return {
                'total_portfolio_value': total_value,
                'assets': sorted(assets, key=lambda x: x['value_usd'], reverse=True),
                'last_updated': datetime.now()
            }
            
        except Exception:
            return None
    
    def get_current_price(self, symbol):
        """Holt aktuellen Preis von der echten API"""
        try:
            # Cache für 10 Sekunden
            if symbol in self.price_cache:
                cached_price, timestamp = self.price_cache[symbol]
                if (datetime.now() - timestamp).total_seconds() < 10:
                    return cached_price
            
            price = self.api.get_ticker(symbol)
            if price:
                self.price_cache[symbol] = (price, datetime.now())
                return price
            else:
                return None
        except Exception:
            return None
    
    def update_caches(self):
        """Aktualisiert alle Caches mit echten Daten"""
        try:
            for symbol in self.trading_pairs:
                self.get_current_price(symbol)
            
            self.balance_cache = self.get_balance_summary()
            
        except Exception:
            pass
    
    def calculate_portfolio_value(self):
        """Berechnet Gesamtwert des Portfolios"""
        balance = self.get_balance_summary()
        return balance['total_portfolio_value'] if balance else 0.0
    
    def set_trading_settings(self, stop_loss=None, trade_size=None, rsi_oversold=None, rsi_overbought=None):
        """Aktualisiert Trading-Einstellungen"""
        if stop_loss is not None:
            self.stop_loss_percent = stop_loss
        if trade_size is not None:
            self.trade_size_percent = trade_size
        if rsi_oversold is not None:
            self.rsi_oversold = rsi_oversold
        if rsi_overbought is not None:
            self.rsi_overbought = rsi_overbought
    
    def set_interval(self, interval):
        """Setzt Analyse-Interval"""
        self.backtest_interval = interval
    
    def check_stop_loss(self):
        """Prüft Stop-Loss für aktive Trades"""
        if not self.active_trades:
            return
            
        for symbol, trade in self.active_trades.items():
            current_price = self.get_current_price(symbol)
            if not current_price:
                continue
                
            buy_price = trade['buy_price']
            stop_loss_price = buy_price * (1 - self.stop_loss_percent / 100)
            
            if current_price <= stop_loss_price:
                self.close_trade(symbol, f"Stop-Loss erreicht ({self.stop_loss_percent}%)")
    
    def close_trade(self, symbol, reason):
        """Schließt einen aktiven Trade mit echter API"""
        if symbol in self.active_trades:
            trade = self.active_trades.pop(symbol)
            current_price = self.get_current_price(symbol)
            if not current_price:
                return
                    
            profit_loss = (current_price - trade['buy_price']) * trade['amount']
                
            if self.auto_trading:
                order_result = self.api.place_order(
                    symbol=symbol,
                    side='sell',
                    order_type='market',
                    size=trade['amount']
                )
                    
                if order_result:
                    order_id = order_result.get('orderId', 'unknown')
                else:
                    order_id = 'failed'
            else:
                order_id = 'simulated'
            
            # Logge den Trade
            trade_data = {
                'symbol': symbol,
                'side': 'SELL',
                'amount': trade['amount'],
                'price': current_price,
                'profit_loss': profit_loss,
                'profit_loss_percent': (profit_loss / (trade['buy_price'] * trade['amount'])) * 100,
                'reason': reason,
                'order_id': order_id,
                'portfolio_value': self.calculate_portfolio_value()
            }
            self.tax_logger.log_trade(trade_data)
            
            # Aktualisiere Trade-History
            self.load_trade_history()
                
            self.update_bot_activity(f"🔒 Trade geschlossen: {symbol} - {reason} - P/L: ${profit_loss:.2f}")
    
    def execute_trade(self, symbol, signal):
        """Führt einen Trade mit echter API aus - KORRIGIERT"""
        if not self.auto_trading:
            print(f"❌ Auto-Trading ist deaktiviert, kann Trade nicht ausführen")
            return False
            
        if symbol in self.active_trades:
            print(f"❌ Trade für {symbol} bereits aktiv")
            return False
            
        if len(self.active_trades) >= self.max_open_trades:
            print(f"❌ Maximale Anzahl offener Trades erreicht")
            return False
            
        current_price = self.get_current_price(symbol)
        if not current_price:
            print(f"❌ Kein aktueller Preis für {symbol} verfügbar")
            return False
        
        portfolio_value = self.calculate_portfolio_value()
        if portfolio_value <= 0:
            print(f"❌ Portfolio-Wert ist 0 oder negativ")
            return False
            
        trade_value = portfolio_value * (self.trade_size_percent / 100)
        trade_amount = trade_value / current_price
        
        valid_amount = self.api.calculate_valid_size(symbol, trade_amount)
        
        print(f"🔄 Versuche Trade: {symbol} {signal} - Menge: {valid_amount:.6f} - Preis: ${current_price:.6f}")
            
        if "BUY" in signal:
            order_result = self.api.place_order(
                symbol=symbol,
                side='buy',
                order_type='market',
                size=valid_amount
            )
                
            if order_result:
                print(f"✅ API Order erfolgreich: {order_result}")
                
                self.active_trades[symbol] = {
                    'buy_price': current_price,
                    'amount': valid_amount,
                    'timestamp': datetime.now(),
                    'order_id': order_result.get('orderId', 'unknown')
                }
                    
                trade_data = {
                    'symbol': symbol,
                    'side': 'BUY',
                    'amount': valid_amount,
                    'price': current_price,
                    'profit_loss': 0,
                    'profit_loss_percent': 0,
                    'reason': f"Auto-Trade: {signal}",
                    'order_id': order_result.get('orderId', 'unknown'),
                    'portfolio_value': portfolio_value
                }
                
                # Trade protokollieren
                logged_trade = self.tax_logger.log_trade(trade_data)
                if logged_trade:
                    print(f"✅ Trade erfolgreich protokolliert: {symbol}")
                else:
                    print(f"❌ Trade konnte nicht protokolliert werden: {symbol}")
                
                # Trade-History SOFORT aktualisieren
                self.load_trade_history()
                
                self.last_trade_time = datetime.now()
                self.update_bot_activity(f"🟢 Trade eröffnet: {symbol} - {valid_amount:.4f} @ ${current_price:.2f}")
                return True
            else:
                print(f"❌ API Order fehlgeschlagen für {symbol}")
                return False
        return False
    
    def get_trade_history_for_gui(self, limit=50):
        """Gibt Trade-History für die GUI zurück - VOLLSTÄNDIG KORRIGIERT"""
        try:
            print(f"🔍 Lade Trade-History für GUI... ({len(self.trade_history)} Trades verfügbar)")
            
            if not self.trade_history:
                print("ℹ️  Keine Trade-History verfügbar")
                return []
            
            # Verwende die interne History und konvertiere sie für die GUI
            gui_trades = []
            
            for trade in self.trade_history[-limit:]:  # Neueste zuerst
                try:
                    # Erstelle ein GUI-kompatibles Trade-Objekt
                    gui_trade = {
                        'timestamp': trade.get('timestamp', 'Unbekannt'),
                        'symbol': trade.get('symbol', 'Unknown'),
                        'side': trade.get('side', 'UNKNOWN'),
                        'price': float(trade.get('price', 0)),
                        'amount': float(trade.get('amount', 0)),
                        'profit_loss': float(trade.get('profit_loss', 0)),
                        'profit_loss_percent': float(trade.get('profit_loss_percent', 0)),
                        'reason': trade.get('reason', '')
                    }
                    gui_trades.append(gui_trade)
                    
                    # Debug-Ausgabe für die ersten paar Trades
                    if len(gui_trades) <= 3:
                        print(f"  📋 GUI Trade: {gui_trade['symbol']} {gui_trade['side']} {gui_trade['timestamp']}")
                        
                except Exception as e:
                    print(f"⚠️  Fehler beim Konvertieren des Trades: {e}")
                    continue
            
            print(f"✅ {len(gui_trades)} Trades für GUI vorbereitet")
            return gui_trades
            
        except Exception as e:
            print(f"❌ Fehler in get_trade_history_for_gui: {e}")
            return []

# =============================================================================
# GUI
# =============================================================================

class TradingBotGUI:
    def __init__(self, bot):
        self.bot = bot
        self.bot.set_gui_reference(self)
        
        self.root = tk.Tk()
        self.root.title("KuCoin Trading Bot")
        
        # Status Variable FRÜH initialisieren
        self.status_var = tk.StringVar(value="Bot initialisiert - Bereit")
        
        # Ermittle Bildschirmauflösung
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        
        print(f"🖥️  Bildschirmauflösung: {self.screen_width}x{self.screen_height}")
        
        # Aktivitätslog
        self.bot_activity_log = []
        self.activity_log = None
        
        # Entscheide welche GUI basierend auf Auflösung
        if self.screen_width >= 1280 and self.screen_height >= 720:
            self.setup_large_gui()
            print("✅ Lade große GUI für High-Res Display")
        else:
            self.setup_small_gui()
            print("✅ Lade kleine GUI für Low-Res Display")
        
        self.start_auto_updates()

        # INITIALE AKTUALISIERUNGEN - WICHTIG
        print("🔄 Starte initiale GUI-Aktualisierungen...")
        self.root.after(1000, self.update_balance_display)  # Verzögert starten
        self.root.after(2000, self.update_trade_history)    # Trade-History nach 2 Sekunden
        self.root.after(3000, self.update_tax_log)  # Finanzamt-Log nach 3 Sekunden
        self.root.after(3000, self.update_recommendations)  # Empfehlungen nach 3 Sekunden
        
    def setup_large_gui(self):
        """Große GUI für High-Res Displays (ab 1280x720)"""
        self.root.geometry("1400x900")
        self.root.configure(bg='#2c3e50')
        
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab-Wechsel Event hinzufügen (wichtig!)
        notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        
        # Mehr Tabs für große Displays
        trading_tab = ttk.Frame(notebook)
        config_tab = ttk.Frame(notebook)
        tax_tab = ttk.Frame(notebook)
        monitoring_tab = ttk.Frame(notebook)
        
        notebook.add(trading_tab, text="Trading")
        notebook.add(config_tab, text="Konfiguration")
        notebook.add(tax_tab, text="Finanzamt")
        notebook.add(monitoring_tab, text="Bot Monitoring")
        
        self.setup_trading_tab_large(trading_tab)
        self.setup_config_tab_large(config_tab)
        self.setup_tax_tab_large(tax_tab)
        self.setup_monitoring_tab_large(monitoring_tab)
        
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def setup_small_gui(self):
        """Kleine GUI für Low-Res Displays (unter 1280x720)"""
        self.root.geometry("780x460")
        self.root.configure(bg='#2c3e50')
        
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Weniger Tabs für kleine Displays
        trading_tab = ttk.Frame(notebook)
        status_tab = ttk.Frame(notebook)
        config_tab = ttk.Frame(notebook)
        
        notebook.add(trading_tab, text="Trading")
        notebook.add(status_tab, text="Status")
        notebook.add(config_tab, text="Einstellungen")
        
        self.setup_trading_tab_small(trading_tab)
        self.setup_status_tab_small(status_tab)
        self.setup_config_tab_small(config_tab)
        
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # === GROSSE GUI KOMPONENTEN ===
    
    def setup_trading_tab_large(self, parent):
        """Trading Tab für große Displays"""
        left_frame = ttk.Frame(parent)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        right_frame = ttk.Frame(parent)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.setup_balance_panel_large(left_frame)
        self.setup_control_panel_large(left_frame)
        self.setup_recommendations_with_trading(left_frame)  # NEU: Mit Trade-Buttons
        self.setup_active_trades_panel_large(right_frame)
        self.setup_trade_history_panel_large(right_frame)
        
    def setup_balance_panel_large(self, parent):
        """Balance Panel für große Displays"""
        balance_frame = ttk.LabelFrame(parent, text="Kontostand & Bestände", padding=10)
        balance_frame.pack(fill=tk.X, pady=5)
        
        refresh_frame = ttk.Frame(balance_frame)
        refresh_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(refresh_frame, text="Aktualisieren", 
                  command=self.update_balance_display).pack(side=tk.LEFT)
        
        self.balance_info_var = tk.StringVar(value="Lade Kontostand...")
        balance_label = ttk.Label(balance_frame, textvariable=self.balance_info_var)
        balance_label.pack(anchor=tk.W)
        
        columns = ('Asset', 'Bestand', 'Verfügbar', 'Preis', 'Wert (USD)', 'Anteil')
        self.balance_tree = ttk.Treeview(balance_frame, columns=columns, show='headings', height=6)
        
        for col in columns:
            self.balance_tree.heading(col, text=col)
            self.balance_tree.column(col, width=100)
        
        self.balance_tree.column('Asset', width=80)
        self.balance_tree.column('Bestand', width=100)
        self.balance_tree.column('Verfügbar', width=100)
        self.balance_tree.column('Preis', width=100)
        self.balance_tree.column('Wert (USD)', width=100)
        self.balance_tree.column('Anteil', width=80)
        
        self.balance_tree.pack(fill=tk.BOTH, expand=True)
        
        self.update_balance_display()
        
    def setup_control_panel_large(self, parent):
        """Control Panel für große Displays - ERWEITERT"""
        control_frame = ttk.LabelFrame(parent, text="Bot Steuerung", padding=10)
        control_frame.pack(fill=tk.X, pady=5)
        
        self.auto_trading_var = tk.BooleanVar(value=self.bot.auto_trading)
        auto_switch = ttk.Checkbutton(control_frame, text="Auto-Trading", 
                                    variable=self.auto_trading_var,
                                    command=self.toggle_auto_trading)
        auto_switch.pack(anchor=tk.W)
        
        settings_frame = ttk.Frame(control_frame)
        settings_frame.pack(fill=tk.X, pady=5)
        
        # Erste Zeile
        ttk.Label(settings_frame, text="Stop-Loss %:").grid(row=0, column=0, sticky=tk.W)
        self.stop_loss_var = tk.StringVar(value=str(self.bot.stop_loss_percent))
        ttk.Entry(settings_frame, textvariable=self.stop_loss_var, width=8).grid(row=0, column=1, padx=5)
        
        ttk.Label(settings_frame, text="Trade Größe %:").grid(row=0, column=2, sticky=tk.W, padx=(20,0))
        self.trade_size_var = tk.StringVar(value=str(self.bot.trade_size_percent))
        ttk.Entry(settings_frame, textvariable=self.trade_size_var, width=8).grid(row=0, column=3, padx=5)
        
        ttk.Label(settings_frame, text="RSI Oversold:").grid(row=0, column=4, sticky=tk.W, padx=(20,0))
        self.rsi_oversold_var = tk.StringVar(value=str(self.bot.rsi_oversold))
        ttk.Entry(settings_frame, textvariable=self.rsi_oversold_var, width=8).grid(row=0, column=5, padx=5)
        
        # Zweite Zeile
        ttk.Label(settings_frame, text="RSI Overbought:").grid(row=1, column=0, sticky=tk.W, pady=(10,0))
        self.rsi_overbought_var = tk.StringVar(value=str(self.bot.rsi_overbought))
        ttk.Entry(settings_frame, textvariable=self.rsi_overbought_var, width=8).grid(row=1, column=1, padx=5, pady=(10,0))
        
        ttk.Label(settings_frame, text="Intervall:").grid(row=1, column=2, sticky=tk.W, pady=(10,0))
        self.interval_var = tk.StringVar(value=self.bot.backtest_interval)
        interval_combo = ttk.Combobox(settings_frame, textvariable=self.interval_var, 
                                    values=['1min', '5min', '15min', '1hour', '4hour', '1day'], width=8)
        interval_combo.grid(row=1, column=3, padx=5, pady=(10,0))
        
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, text="Einstellungen Speichern", 
                command=self.save_settings).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Schnell-Check", 
                command=self.quick_signal_check).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Backtest Starten", 
                command=self.start_backtest).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="History Aktualisieren", 
                command=self.force_history_update).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Alle Trades Schliessen", 
                command=self.close_all_trades).pack(side=tk.LEFT, padx=2)
        
    def setup_recommendations_with_trading(self, parent):
        """Erweitert das Recommendations Panel mit Trade-Buttons"""
        rec_frame = ttk.LabelFrame(parent, text="Trading Empfehlungen", padding=10)
        rec_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Toolbar für Empfehlungen
        toolbar = ttk.Frame(rec_frame)
        toolbar.pack(fill=tk.X, pady=5)
        
        ttk.Button(toolbar, text="Alle Kauf-Signale ausführen", 
                  command=self.execute_all_buy_signals).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Ausgewählten Trade ausführen", 
                  command=self.execute_selected_trade).pack(side=tk.LEFT, padx=2)
        
        # Treeview mit Action-Spalte
        columns = ('Symbol', 'Preis', 'Signal', 'Confidence', 'Performance', 'Aktion')
        self.rec_tree = ttk.Treeview(rec_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.rec_tree.heading(col, text=col)
            self.rec_tree.column(col, width=90)
        
        self.rec_tree.column('Symbol', width=100)
        self.rec_tree.column('Aktion', width=120)
        self.rec_tree.pack(fill=tk.BOTH, expand=True)
        
        # Double-Click Event
        self.rec_tree.bind('<Double-1>', self.on_recommendation_double_click)
        
    def execute_selected_trade(self):
        """Führt Trade für ausgewählte Empfehlung aus"""
        selection = self.rec_tree.selection()
        if not selection:
            messagebox.showwarning("Warnung", "Bitte wählen Sie eine Empfehlung aus!")
            return
            
        item = self.rec_tree.item(selection[0])
        values = item['values']
        symbol = values[0]
        signal = values[2]
        
        if "BUY" not in signal:
            messagebox.showwarning("Warnung", "Nur KAUF-Signale können ausgeführt werden!")
            return
            
        self.execute_manual_trade(symbol, signal)

    def execute_all_buy_signals(self):
        """Führt alle KAUF-Signale aus"""
        buy_signals = []
        for item in self.rec_tree.get_children():
            values = self.rec_tree.item(item)['values']
            if "BUY" in values[2] and float(values[3].replace('%', '')) >= 70:
                buy_signals.append(values[0])
        
        if not buy_signals:
            messagebox.showinfo("Info", "Keine KAUF-Signale mit Confidence >= 70% gefunden")
            return
            
        result = messagebox.askyesno(
            "Bestätigung", 
            f"Möchten Sie {len(buy_signals)} KAUF-Signale ausführen?\n\n"
            f"Symbole: {', '.join(buy_signals)}"
        )
        
        if result:
            for symbol in buy_signals:
                self.execute_manual_trade(symbol, "STRONG_BUY")

    def execute_manual_trade(self, symbol, signal):
        """Führt einen manuellen Trade aus"""
        def execute():
            try:
                self.update_status(f"🎯 Führe manuellen Trade aus: {symbol} {signal}")
                
                # Temporär Auto-Trading aktivieren falls nötig
                was_auto_trading = self.bot.auto_trading
                if not was_auto_trading:
                    self.bot.auto_trading = True
                    
                # Trade ausführen
                success = self.bot.execute_trade(symbol, signal)
                
                # Auto-Trading Status zurücksetzen
                if not was_auto_trading:
                    self.bot.auto_trading = False
                    
                if success:
                    messagebox.showinfo("Erfolg", f"Trade für {symbol} erfolgreich ausgeführt!")
                    
                    # WICHTIG: Nach Trade alle relevanten Teile aktualisieren
                    self.update_active_trades()
                    self.update_balance_display()
                    self.update_recommendations_with_actions()
                    
                    # Finanzamt-Tab nur wenn sichtbar
                    self.update_finanzamt_on_demand()
                    
                else:
                    messagebox.showerror("Fehler", f"Trade für {symbol} fehlgeschlagen!")
                    
            except Exception as e:
                messagebox.showerror("Fehler", f"Trade-Fehler: {str(e)}")
        
        # Bestätigungsdialog
        current_price = self.bot.get_current_price(symbol)
        if current_price:
            confirmation = messagebox.askyesno(
                "Trade bestätigen",
                f"Möchten Sie diesen Trade ausführen?\n\n"
                f"Symbol: {symbol}\n"
                f"Signal: {signal}\n"
                f"Aktueller Preis: ${current_price:.6f}\n"
                f"Trade-Größe: {self.bot.trade_size_percent}% des Portfolios"
            )
            
            if confirmation:
                threading.Thread(target=execute, daemon=True).start()

    def on_recommendation_double_click(self, event):
        """Handle Double-Click auf Empfehlung"""
        self.execute_selected_trade()

    def update_recommendations_with_actions(self):
        """Aktualisiert Empfehlungen mit Action-Buttons (vereinfacht)"""
        if not hasattr(self, 'rec_tree'):
            return
            
        for item in self.rec_tree.get_children():
            self.rec_tree.delete(item)
            
        if not self.bot.current_recommendations:
            self.rec_tree.insert('', tk.END, values=(
                "Keine", "Daten", "verfügbar", "", "", ""
            ))
            return
            
        for crypto, data in self.bot.current_recommendations.items():
            try:
                signal = data.get('current_signal', 'HOLD')
                confidence = data.get('confidence', 0)
                price = data.get('current_price', 0)
                performance = f"{data.get('total_return', 0):+.1f}%"
                
                # Action-Button Text basierend auf Signal
                action_text = ""
                if "BUY" in signal and confidence >= 70:
                    action_text = "🟢 HANDELN"
                elif "SELL" in signal:
                    action_text = "🔴 VERKAUFEN"
                else:
                    action_text = "🟡 WARTEN"
                
                tags = ()
                if "BUY" in signal:
                    tags = ('buy',)
                elif "SELL" in signal:
                    tags = ('sell',)
                else:
                    tags = ('hold',)
                    
                self.rec_tree.insert('', tk.END, values=(
                    crypto, 
                    f"${price:.6f}", 
                    signal, 
                    f"{confidence:.0f}%", 
                    performance, 
                    action_text
                ), tags=tags)
                
            except Exception as e:
                continue
                
        if hasattr(self, 'rec_tree'):
            self.rec_tree.tag_configure('buy', background='#d4edda')
            self.rec_tree.tag_configure('sell', background='#f8d7da')
            self.rec_tree.tag_configure('hold', background='#fff3cd')
        
    def setup_active_trades_panel_large(self, parent):
        """Active Trades Panel für große Displays"""
        trades_frame = ttk.LabelFrame(parent, text="Aktive Trades", padding=10)
        trades_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        columns = ('Symbol', 'Kaufpreis', 'Aktuell', 'Menge', 'P/L %', 'P/L €', 'Seit')
        self.trades_tree = ttk.Treeview(trades_frame, columns=columns, show='headings', height=6)
        
        for col in columns:
            self.trades_tree.heading(col, text=col)
            self.trades_tree.column(col, width=80)
        
        self.trades_tree.pack(fill=tk.BOTH, expand=True)
        
        action_frame = ttk.Frame(trades_frame)
        action_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(action_frame, text="Trade Schliessen", 
                  command=self.close_selected_trade).pack(side=tk.LEFT, padx=2)
        
    def setup_trade_history_panel_large(self, parent):
        """Trade History Panel für große Displays"""
        history_frame = ttk.LabelFrame(parent, text="Trade History", padding=10)
        history_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        columns = ('Datum', 'Symbol', 'Side', 'Preis', 'Menge', 'P/L %', 'P/L €', 'Grund')
        self.history_tree = ttk.Treeview(history_frame, columns=columns, show='headings', height=6)
        
        for col in columns:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=80)
        
        self.history_tree.column('Datum', width=120)
        self.history_tree.column('Grund', width=120)
        self.history_tree.pack(fill=tk.BOTH, expand=True)
        
    def setup_config_tab_large(self, parent):
        """Config Tab für große Displays"""
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        left_frame = ttk.LabelFrame(main_frame, text="Verfügbare Trading-Pairs", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        right_frame = ttk.LabelFrame(main_frame, text="Ausgewählte Trading-Pairs", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.setup_pairs_selection_large(left_frame, right_frame)
        
    def setup_pairs_selection_large(self, left_frame, right_frame):
        """Pairs Selection für große Displays"""
        # Verfügbare Pairs
        available_frame = ttk.Frame(left_frame)
        available_frame.pack(fill=tk.BOTH, expand=True)
        
        search_frame = ttk.Frame(available_frame)
        search_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(search_frame, text="Suchen:").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=20)
        search_entry.pack(side=tk.LEFT, padx=5)
        search_entry.bind('<KeyRelease>', self.filter_available_pairs)
        
        ttk.Button(search_frame, text="Alle laden", 
                  command=self.load_available_pairs).pack(side=tk.RIGHT, padx=5)
        
        self.available_listbox = tk.Listbox(available_frame, selectmode=tk.MULTIPLE)
        self.available_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        
        available_scrollbar = ttk.Scrollbar(self.available_listbox)
        available_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.available_listbox.config(yscrollcommand=available_scrollbar.set)
        available_scrollbar.config(command=self.available_listbox.yview)
        
        available_buttons = ttk.Frame(available_frame)
        available_buttons.pack(fill=tk.X, pady=5)
        
        ttk.Button(available_buttons, text="Auswählen", 
                  command=self.add_selected_pairs).pack(side=tk.LEFT, padx=2)
        ttk.Button(available_buttons, text="Alle auswählen", 
                  command=self.add_all_pairs).pack(side=tk.LEFT, padx=2)
        
        # Ausgewählte Pairs
        selected_frame = ttk.Frame(right_frame)
        selected_frame.pack(fill=tk.BOTH, expand=True)
        
        self.selected_listbox = tk.Listbox(selected_frame, selectmode=tk.MULTIPLE)
        self.selected_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        
        selected_scrollbar = ttk.Scrollbar(self.selected_listbox)
        selected_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.selected_listbox.config(yscrollcommand=selected_scrollbar.set)
        selected_scrollbar.config(command=self.selected_listbox.yview)
        
        selected_buttons = ttk.Frame(selected_frame)
        selected_buttons.pack(fill=tk.X, pady=5)
        
        ttk.Button(selected_buttons, text="Entfernen", 
                  command=self.remove_selected_pairs).pack(side=tk.LEFT, padx=2)
        ttk.Button(selected_buttons, text="Alle entfernen", 
                  command=self.remove_all_pairs).pack(side=tk.LEFT, padx=2)
        ttk.Button(selected_buttons, text="Speichern", 
                  command=self.save_trading_pairs).pack(side=tk.RIGHT, padx=2)
        
        default_pairs_frame = ttk.LabelFrame(right_frame, text="Schnellauswahl", padding=10)
        default_pairs_frame.pack(fill=tk.X, pady=10)
        
        default_pairs = [
            "BTC-USDT", "ETH-USDT", "ADA-USDT", "DOT-USDT", "LINK-USDT",
            "BNB-USDT", "XRP-USDT", "SOL-USDT", "DOGE-USDT", "MATIC-USDT"
        ]
        
        for i in range(0, len(default_pairs), 5):
            row_frame = ttk.Frame(default_pairs_frame)
            row_frame.pack(fill=tk.X, pady=2)
            for pair in default_pairs[i:i+5]:
                ttk.Button(row_frame, text=pair, width=10,
                          command=lambda p=pair: self.add_single_pair(p)).pack(side=tk.LEFT, padx=2)
        
        self.load_available_pairs()
        self.load_current_pairs()
        
    def setup_tax_tab_large(self, parent):
        """Tax Tab für große Displays - OPTIMIERT"""
        tax_frame = ttk.LabelFrame(parent, text="Steuerliche Handelsaufzeichnungen", padding=10)
        tax_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        report_frame = ttk.Frame(tax_frame)
        report_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(report_frame, text="Steuerreport Generieren", 
                  command=self.generate_tax_report).pack(side=tk.LEFT, padx=2)
        ttk.Button(report_frame, text="Logs Exportieren", 
                  command=self.export_logs).pack(side=tk.LEFT, padx=2)
        ttk.Button(report_frame, text="Daten Aktualisieren", 
                  command=self.force_tax_update).pack(side=tk.LEFT, padx=2)  # NEU
        ttk.Button(report_frame, text="Debug History", 
                  command=self.debug_trade_history).pack(side=tk.LEFT, padx=2)
        
        columns = ('Datum', 'Typ', 'Symbol', 'Menge', 'Preis', 'Gesamt', 'Gebühren', 'Gewinn/Verlust', 'Grund')
        self.tax_tree = ttk.Treeview(tax_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.tax_tree.heading(col, text=col)
            self.tax_tree.column(col, width=100)
        
        self.tax_tree.column('Datum', width=120)
        self.tax_tree.column('Grund', width=150)
        self.tax_tree.pack(fill=tk.BOTH, expand=True)
        
        info_frame = ttk.Frame(tax_frame)
        info_frame.pack(fill=tk.X, pady=5)
        
        self.portfolio_var = tk.StringVar(value="Portfolio Wert: €0.00")
        ttk.Label(info_frame, textvariable=self.portfolio_var).pack(side=tk.LEFT)
        
        self.total_profit_var = tk.StringVar(value="Gesamtgewinn: €0.00")
        ttk.Label(info_frame, textvariable=self.total_profit_var).pack(side=tk.LEFT, padx=20)
        
    def setup_monitoring_tab_large(self, parent):
        """Monitoring Tab für große Displays"""
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        left_frame = ttk.LabelFrame(main_frame, text="Bot Status", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        right_frame = ttk.LabelFrame(main_frame, text="Aktivitätslog", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.bot_status_vars = {}
        
        status_fields = [
            ("Auto-Trading", "auto_trading"),
            ("Aktive Trades", "active_trades"),
            ("Trading-Pairs", "trading_pairs"),
            ("Letzte Analyse", "last_analysis"),
            ("Nächste Analyse", "next_analysis"),
            ("API Requests", "api_requests"),
            ("Letzter Trade", "last_trade"),
            ("Signal-Modus", "signal_mode")
        ]
        
        for i, (label, key) in enumerate(status_fields):
            ttk.Label(left_frame, text=f"{label}:").grid(row=i, column=0, sticky=tk.W, pady=2)
            self.bot_status_vars[key] = tk.StringVar(value="-")
            ttk.Label(left_frame, textvariable=self.bot_status_vars[key]).grid(row=i, column=1, sticky=tk.W, pady=2, padx=(10, 0))
        
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).grid(row=len(status_fields), column=0, 
                                                           columnspan=2, sticky=tk.EW, pady=10)
        
        ttk.Label(left_frame, text="API Statistiken").grid(
            row=len(status_fields)+1, column=0, columnspan=2, sticky=tk.W, pady=(10, 5))
        
        api_fields = [
            ("Total Requests", "total_requests"),
            ("Letzter Request", "last_request")
        ]
        
        for i, (label, key) in enumerate(api_fields):
            ttk.Label(left_frame, text=f"{label}:").grid(
                row=len(status_fields)+2+i, column=0, sticky=tk.W, pady=2)
            self.bot_status_vars[key] = tk.StringVar(value="-")
            ttk.Label(left_frame, textvariable=self.bot_status_vars[key]).grid(
                row=len(status_fields)+2+i, column=1, sticky=tk.W, pady=2, padx=(10, 0))
        
        button_frame = ttk.Frame(left_frame)
        button_frame.grid(row=len(status_fields)+4, column=0, columnspan=2, sticky=tk.EW, pady=20)
        
        ttk.Button(button_frame, text="Sofort Analyse", 
                  command=self.quick_signal_check).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="API Stats Aktualisieren", 
                  command=self.force_api_stats_update).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Log Leeren", 
                  command=self.clear_activity_log).pack(side=tk.LEFT, padx=2)
        
        self.activity_log = scrolledtext.ScrolledText(
            right_frame, 
            wrap=tk.WORD, 
            width=60, 
            height=20
        )
        self.activity_log.pack(fill=tk.BOTH, expand=True)
        self.activity_log.config(state=tk.DISABLED)
        
        self.update_bot_status()

    # === KLEINE GUI KOMPONENTEN ===
    
    def setup_trading_tab_small(self, parent):
        """Optimierte Trading-Tab für kleines Display"""
        # Obere Reihe: Balance und Kontrolle
        top_frame = ttk.Frame(parent)
        top_frame.pack(fill=tk.X, pady=2)
        
        # Balance Frame
        balance_frame = ttk.LabelFrame(top_frame, text="Kontostand")
        balance_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        self.balance_info_var = tk.StringVar(value="Lade Kontostand...")
        ttk.Label(balance_frame, textvariable=self.balance_info_var).pack()
        
        ttk.Button(balance_frame, text="Aktualisieren", 
                  command=self.update_balance_display, width=12).pack(pady=2)
        
        # Control Frame
        control_frame = ttk.LabelFrame(top_frame, text="Steuerung")
        control_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)
        
        self.auto_trading_var = tk.BooleanVar(value=self.bot.auto_trading)
        ttk.Checkbutton(control_frame, text="Auto-Trading", 
                       variable=self.auto_trading_var,
                       command=self.toggle_auto_trading).pack()
        
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(pady=2)
        
        ttk.Button(btn_frame, text="Schnell-Check", 
                  command=self.quick_signal_check, width=10).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_frame, text="Backtest", 
                  command=self.start_backtest, width=8).pack(side=tk.LEFT, padx=1)
        
        # Mittlere Reihe: Empfehlungen
        rec_frame = ttk.LabelFrame(parent, text="Trading Empfehlungen")
        rec_frame.pack(fill=tk.BOTH, expand=True, pady=2)
        
        tree_frame = ttk.Frame(rec_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ('Symbol', 'Signal', 'Confidence', 'Preis')
        self.rec_tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=6)
        
        self.rec_tree.column('Symbol', width=80, minwidth=80)
        self.rec_tree.column('Signal', width=80, minwidth=80)
        self.rec_tree.column('Confidence', width=70, minwidth=70)
        self.rec_tree.column('Preis', width=90, minwidth=90)
        
        for col in columns:
            self.rec_tree.heading(col, text=col)
        
        tree_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.rec_tree.yview)
        self.rec_tree.configure(yscrollcommand=tree_scroll.set)
        
        self.rec_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Untere Reihe: Aktive Trades
        trades_frame = ttk.LabelFrame(parent, text="Aktive Trades")
        trades_frame.pack(fill=tk.X, pady=2)
        
        trades_columns = ('Symbol', 'Kaufpreis', 'Aktuell', 'P/L %')
        self.trades_tree = ttk.Treeview(trades_frame, columns=trades_columns, show='headings', height=3)
        
        self.trades_tree.column('Symbol', width=70, minwidth=70)
        self.trades_tree.column('Kaufpreis', width=80, minwidth=80)
        self.trades_tree.column('Aktuell', width=80, minwidth=80)
        self.trades_tree.column('P/L %', width=60, minwidth=60)
        
        for col in trades_columns:
            self.trades_tree.heading(col, text=col)
        
        self.trades_tree.pack(fill=tk.X)
        
        action_frame = ttk.Frame(trades_frame)
        action_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(action_frame, text="Trade Schliessen", 
                  command=self.close_selected_trade, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="Alle Schliessen", 
                  command=self.close_all_trades, width=12).pack(side=tk.LEFT, padx=2)
        
        self.update_balance_display()
        
    def setup_status_tab_small(self, parent):
        """Optimierte Status-Tab"""
        left_frame = ttk.LabelFrame(parent, text="Bot Status")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        self.bot_status_vars = {}
        
        status_fields = [
            ("Auto-Trading", "auto_trading"),
            ("Aktive Trades", "active_trades"),
            ("Trading-Pairs", "trading_pairs"),
            ("Letzte Analyse", "last_analysis"),
            ("Nächste Analyse", "next_analysis"),
            ("Letzter Trade", "last_trade"),
            ("API Requests", "api_requests")
        ]
        
        for i, (label, key) in enumerate(status_fields):
            row_frame = ttk.Frame(left_frame)
            row_frame.pack(fill=tk.X, pady=1)
            
            ttk.Label(row_frame, text=f"{label}:", width=15, anchor=tk.W).pack(side=tk.LEFT)
            self.bot_status_vars[key] = tk.StringVar(value="-")
            ttk.Label(row_frame, textvariable=self.bot_status_vars[key]).pack(side=tk.LEFT)
        
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Cache Aktualisieren", 
                  command=self.force_cache_update).pack(pady=2)
        ttk.Button(button_frame, text="Log Leeren", 
                  command=self.clear_activity_log).pack(pady=2)
        
        right_frame = ttk.LabelFrame(parent, text="Aktivitätslog")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        self.activity_log = scrolledtext.ScrolledText(
            right_frame, 
            wrap=tk.WORD, 
            width=40, 
            height=15
        )
        self.activity_log.pack(fill=tk.BOTH, expand=True)
        self.activity_log.config(state=tk.DISABLED)
        
        self.update_bot_status()
        
    def setup_config_tab_small(self, parent):
        """Optimierte Konfigurations-Tab"""
        pairs_frame = ttk.LabelFrame(parent, text="Trading-Pairs")
        pairs_frame.pack(fill=tk.BOTH, expand=True, pady=2)
        
        available_frame = ttk.Frame(pairs_frame)
        available_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(available_frame, text="Verfügbare Pairs:").pack(side=tk.LEFT)
        ttk.Button(available_frame, text="Laden", 
                  command=self.load_available_pairs, width=8).pack(side=tk.RIGHT, padx=2)
        
        self.available_listbox = tk.Listbox(pairs_frame, height=4)
        self.available_listbox.pack(fill=tk.X, pady=2)
        
        selected_frame = ttk.Frame(pairs_frame)
        selected_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(selected_frame, text="Ausgewählte Pairs:").pack(side=tk.LEFT)
        
        btn_frame = ttk.Frame(selected_frame)
        btn_frame.pack(side=tk.RIGHT)
        
        ttk.Button(btn_frame, text="Hinzufügen", 
                  command=self.add_selected_pairs, width=8).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_frame, text="Entfernen", 
                  command=self.remove_selected_pairs, width=8).pack(side=tk.LEFT, padx=1)
        
        self.selected_listbox = tk.Listbox(pairs_frame, height=4)
        self.selected_listbox.pack(fill=tk.X, pady=2)
        
        ttk.Button(pairs_frame, text="Pairs Speichern", 
                  command=self.save_trading_pairs).pack(pady=2)
        
        quick_frame = ttk.LabelFrame(parent, text="Schnellauswahl")
        quick_frame.pack(fill=tk.X, pady=2)
        
        popular_pairs = ['BTC-USDT', 'ETH-USDT', 'ADA-USDT', 'DOT-USDT', 'LINK-USDT']
        
        for i in range(0, len(popular_pairs), 3):
            row_frame = ttk.Frame(quick_frame)
            row_frame.pack(fill=tk.X, pady=1)
            for pair in popular_pairs[i:i+3]:
                ttk.Button(row_frame, text=pair, 
                          command=lambda p=pair: self.add_single_pair(p),
                          width=10).pack(side=tk.LEFT, padx=1)
        
        settings_frame = ttk.LabelFrame(parent, text="Trading Einstellungen")
        settings_frame.pack(fill=tk.X, pady=2)
        
        row1 = ttk.Frame(settings_frame)
        row1.pack(fill=tk.X, pady=1)
        
        ttk.Label(row1, text="Stop-Loss %:", width=12).pack(side=tk.LEFT)
        self.stop_loss_var = tk.StringVar(value=str(self.bot.stop_loss_percent))
        ttk.Entry(row1, textvariable=self.stop_loss_var, width=8).pack(side=tk.LEFT)
        
        ttk.Label(row1, text="Trade Größe %:", width=12).pack(side=tk.LEFT, padx=(10,0))
        self.trade_size_var = tk.StringVar(value=str(self.bot.trade_size_percent))
        ttk.Entry(row1, textvariable=self.trade_size_var, width=8).pack(side=tk.LEFT)
        
        row2 = ttk.Frame(settings_frame)
        row2.pack(fill=tk.X, pady=1)
        
        ttk.Label(row2, text="RSI Oversold:", width=12).pack(side=tk.LEFT)
        self.rsi_oversold_var = tk.StringVar(value=str(self.bot.rsi_oversold))
        ttk.Entry(row2, textvariable=self.rsi_oversold_var, width=8).pack(side=tk.LEFT)
        
        ttk.Label(row2, text="RSI Overbought:", width=12).pack(side=tk.LEFT, padx=(10,0))
        self.rsi_overbought_var = tk.StringVar(value=str(self.bot.rsi_overbought))
        ttk.Entry(row2, textvariable=self.rsi_overbought_var, width=8).pack(side=tk.LEFT)
        
        ttk.Button(settings_frame, text="Einstellungen Speichern", 
                  command=self.save_settings).pack(pady=5)

    # === GEMEINSAME METHODEN ===
    
    def update_bot_status(self):
        """Aktualisiert den Bot-Status"""
        try:
            if hasattr(self, 'bot_status_vars'):
                self.bot_status_vars['auto_trading'].set(
                    "AKTIV" if self.bot.auto_trading else "INAKTIV"
                )
                self.bot_status_vars['active_trades'].set(
                    f"{len(self.bot.active_trades)}"
                )
                self.bot_status_vars['trading_pairs'].set(
                    f"{len(self.bot.trading_pairs)}"
                )
                
                if self.bot.last_update:
                    self.bot_status_vars['last_analysis'].set(
                        self.bot.last_update.strftime('%H:%M')
                    )
                else:
                    self.bot_status_vars['last_analysis'].set("--:--")
                    
                if self.bot.next_scheduled_update:
                    time_diff = self.bot.next_scheduled_update - datetime.now()
                    minutes = max(0, int(time_diff.total_seconds() / 60))
                    self.bot_status_vars['next_analysis'].set(f"{minutes}m")
                else:
                    self.bot_status_vars['next_analysis'].set("-")
                
                if self.bot.last_trade_time:
                    self.bot_status_vars['last_trade'].set(
                        self.bot.last_trade_time.strftime('%H:%M')
                    )
                else:
                    self.bot_status_vars['last_trade'].set("--:--")
                
                api_stats = self.bot.api.get_api_stats()
                if api_stats:
                    if 'total_requests' in self.bot_status_vars:
                        self.bot_status_vars['total_requests'].set(str(api_stats['request_count']))
                    if 'last_request' in self.bot_status_vars:
                        self.bot_status_vars['last_request'].set(api_stats['last_request_time'])
                        
        except Exception as e:
            print(f"Status update error: {e}")
            
        self.root.after(5000, self.update_bot_status)

    def update_recommendations(self):
        """Aktualisiert die Trading-Empfehlungen (für Kompatibilität)"""
        self.update_recommendations_with_actions()

    def update_active_trades(self):
        """Aktualisiert aktive Trades"""
        if hasattr(self, 'trades_tree'):
            for item in self.trades_tree.get_children():
                self.trades_tree.delete(item)
                
        for symbol, trade in self.bot.active_trades.items():
            current_price = self.bot.get_current_price(symbol)
            if current_price:
                pl_percent = ((current_price - trade['buy_price']) / trade['buy_price']) * 100
                
                if hasattr(self, 'screen_width') and self.screen_width >= 1280:
                    # Große GUI
                    pl_amount = (current_price - trade['buy_price']) * trade['amount']
                    time_since = datetime.now() - trade['timestamp']
                    hours = int(time_since.total_seconds() / 3600)
                    minutes = int((time_since.total_seconds() % 3600) / 60)
                    
                    tags = ('profit',) if pl_percent >= 0 else ('loss',)
                    
                    if hasattr(self, 'trades_tree'):
                        self.trades_tree.insert('', tk.END, values=(
                            symbol,
                            f"${trade['buy_price']:.6f}",
                            f"${current_price:.6f}",
                            f"{trade['amount']:.4f}",
                            f"{pl_percent:+.2f}%",
                            f"${pl_amount:+.2f}",
                            f"{hours:02d}:{minutes:02d}"
                        ), tags=tags)
                else:
                    # Kleine GUI
                    tags = ('profit',) if pl_percent >= 0 else ('loss',)
                    
                    if hasattr(self, 'trades_tree'):
                        self.trades_tree.insert('', tk.END, values=(
                            symbol,
                            f"${trade['buy_price']:.4f}",
                            f"${current_price:.4f}",
                            f"{pl_percent:+.1f}%"
                        ), tags=tags)
                
        if hasattr(self, 'trades_tree'):
            self.trades_tree.tag_configure('profit', background='#d4edda')
            self.trades_tree.tag_configure('loss', background='#f8d7da')

    def load_available_pairs(self):
        """Lädt verfügbare Trading-Pairs von KuCoin"""
        def load_pairs():
            self.update_status("Lade verfügbare Trading-Pairs...")
            available_pairs = self.bot.get_available_pairs()
            self.root.after(0, self._update_available_pairs, available_pairs)
        
        threading.Thread(target=load_pairs, daemon=True).start()
    
    def _update_available_pairs(self, pairs):
        """Aktualisiert die Liste der verfügbaren Pairs in der GUI"""
        if hasattr(self, 'available_listbox'):
            self.available_listbox.delete(0, tk.END)
            self.available_pairs_list = pairs
            
            for pair in pairs[:50]:
                self.available_listbox.insert(tk.END, pair)
            
            self.update_status(f"{len(pairs)} verfügbare Pairs geladen")
            self.load_current_pairs()
    
    def load_current_pairs(self):
        """Lädt aktuell ausgewählte Trading-Pairs"""
        if hasattr(self, 'selected_listbox'):
            self.selected_listbox.delete(0, tk.END)
            for pair in self.bot.trading_pairs:
                self.selected_listbox.insert(tk.END, pair)
    
    def filter_available_pairs(self, event=None):
        """Filtert die verfügbaren Pairs basierend auf der Suche"""
        if not hasattr(self, 'available_pairs_list') or not hasattr(self, 'available_listbox'):
            return
            
        search_term = self.search_var.get().upper()
        self.available_listbox.delete(0, tk.END)
        
        for pair in self.available_pairs_list:
            if search_term in pair:
                self.available_listbox.insert(tk.END, pair)
    
    def add_selected_pairs(self):
        """Fügt ausgewählte Pairs zur Auswahlliste hinzu"""
        if hasattr(self, 'available_listbox') and hasattr(self, 'selected_listbox'):
            selected_indices = self.available_listbox.curselection()
            current_pairs = list(self.selected_listbox.get(0, tk.END))
            
            for index in selected_indices:
                pair = self.available_listbox.get(index)
                if pair not in current_pairs:
                    self.selected_listbox.insert(tk.END, pair)
    
    def add_all_pairs(self):
        """Fügt alle verfügbaren Pairs zur Auswahlliste hinzu"""
        if hasattr(self, 'available_pairs_list') and hasattr(self, 'selected_listbox'):
            self.selected_listbox.delete(0, tk.END)
            for pair in self.available_pairs_list[:20]:  # Limit für Performance
                self.selected_listbox.insert(tk.END, pair)
    
    def add_single_pair(self, pair):
        """Fügt ein einzelnes Pair zur Auswahlliste hinzu"""
        if hasattr(self, 'selected_listbox'):
            current_pairs = list(self.selected_listbox.get(0, tk.END))
            if pair not in current_pairs:
                self.selected_listbox.insert(tk.END, pair)
    
    def remove_selected_pairs(self):
        """Entfernt ausgewählte Pairs aus der Auswahlliste"""
        if hasattr(self, 'selected_listbox'):
            selected_indices = self.selected_listbox.curselection()
            for index in reversed(selected_indices):
                self.selected_listbox.delete(index)
    
    def remove_all_pairs(self):
        """Entfernt alle Pairs aus der Auswahlliste"""
        if hasattr(self, 'selected_listbox'):
            self.selected_listbox.delete(0, tk.END)
    
    def get_selected_pairs(self):
        """Gibt alle ausgewählten Pairs zurück"""
        if hasattr(self, 'selected_listbox'):
            return list(self.selected_listbox.get(0, tk.END))
        return []
    
    def save_trading_pairs(self):
        """Speichert die ausgewählte Trading-Pairs"""
        selected_pairs = self.get_selected_pairs()
        
        if not selected_pairs:
            messagebox.showwarning("Warnung", "Bitte wählen Sie mindestens ein Trading-Pair aus!")
            return
        
        self.bot.set_trading_pairs(selected_pairs)
        messagebox.showinfo("Erfolg", f"{len(selected_pairs)} Trading-Pairs gespeichert!")
        
        # Backtest mit neuen Pairs starten
        self.start_backtest_with_pairs(selected_pairs)

    def start_backtest_with_pairs(self, pairs):
        """Startet Backtest mit spezifischen Pairs"""
        def run_backtest():
            self.update_status(f"Starte Backtest mit {len(pairs)} Pairs...")
            results = self.bot.run_complete_backtest(pairs)
            self.root.after(0, self._update_after_backtest, results)
        
        threading.Thread(target=run_backtest, daemon=True).start()

    def _update_after_backtest(self, results):
        """Aktualisiert die GUI nach Backtest-Abschluss"""
        try:
            if results:
                self.update_recommendations_with_actions()
                self.update_status(f"Backtest abgeschlossen - {len(results)} Kryptos analysiert")
            else:
                self.update_status("Backtest fehlgeschlagen - Keine Ergebnisse")
        except Exception as e:
            self.update_status(f"Update Fehler: {str(e)}")

    def toggle_auto_trading(self):
        """Schaltet Auto-Trading um - mit Bestätigung"""
        new_state = self.auto_trading_var.get()
        
        if new_state:
            result = messagebox.askyesno(
                "Auto-Trading aktivieren", 
                "WARNUNG: Auto-Trading wird echte Trades ausführen!\n\n"
                "Möchten Sie wirklich fortfahren?"
            )
            if not result:
                self.auto_trading_var.set(False)
                return
        
        self.bot.auto_trading = self.auto_trading_var.get()
        status = "AKTIV" if self.bot.auto_trading else "INAKTIV"
        self.update_status(f"Auto-Trading: {status}")

    def save_settings(self):
        """Speichert die Einstellungen"""
        try:
            stop_loss = float(self.stop_loss_var.get())
            trade_size = float(self.trade_size_var.get())
            rsi_oversold = float(self.rsi_oversold_var.get())
            rsi_overbought = float(self.rsi_overbought_var.get())
            
            self.bot.set_trading_settings(
                stop_loss=stop_loss,
                trade_size=trade_size,
                rsi_oversold=rsi_oversold,
                rsi_overbought=rsi_overbought
            )
            
            if hasattr(self, 'interval_var'):
                self.bot.set_interval(self.interval_var.get())
            
            self.update_status("Einstellungen gespeichert")
            messagebox.showinfo("Erfolg", "Einstellungen wurden gespeichert!")
            
        except ValueError:
            messagebox.showerror("Fehler", "Bitte gültige Zahlen eingeben!")

    def quick_signal_check(self):
        """Startet einen schnellen Signal-Check"""
        def run_quick_check():
            self.bot.quick_signal_check()
            self.root.after(0, self.update_recommendations_with_actions)
        
        threading.Thread(target=run_quick_check, daemon=True).start()
        self.update_status("Schnelle Signalprüfung gestartet...")

    def start_backtest(self):
        """Startet einen Backtest"""
        def run_backtest():
            try:
                self.update_status("Starte Backtest...")
                
                # Deaktiviere Auto-Trading während Backtest um Konflikte zu vermeiden
                was_auto_trading = self.bot.auto_trading
                if was_auto_trading:
                    self.bot.auto_trading = False
                    self.auto_trading_var.set(False)
                
                # Backtest ausführen
                results = self.bot.run_complete_backtest()
                
                # Auto-Trading wieder aktivieren falls es vorher aktiv war
                if was_auto_trading:
                    self.bot.auto_trading = True
                    self.auto_trading_var.set(True)
                
                # GUI im Haupt-Thread aktualisieren
                self.root.after(0, lambda: self._update_after_backtest_complete(results, was_auto_trading))
                
            except Exception as e:
                error_msg = f"Backtest Fehler: {str(e)}"
                print(error_msg)
                self.root.after(0, lambda: self.update_status(error_msg))
        
        threading.Thread(target=run_backtest, daemon=True).start()

    def _update_after_backtest_complete(self, results, was_auto_trading):
        """Aktualisiert die GUI nach Backtest-Abschluss"""
        try:
            if results:
                self.update_recommendations_with_actions()
                
                if was_auto_trading:
                    self.update_status(f"Backtest abgeschlossen - Auto-Trading läuft weiter")
                    messagebox.showinfo("Backtest Abgeschlossen", 
                                      f"Backtest erfolgreich für {len(results)} Kryptowährungen durchgeführt!\n\n"
                                      "Auto-Trading wurde während des Backtests pausiert und ist jetzt wieder aktiv.")
                else:
                    self.update_status(f"Backtest abgeschlossen - {len(results)} Kryptos analysiert")
                    messagebox.showinfo("Backtest Abgeschlossen", 
                                      f"Backtest erfolgreich für {len(results)} Kryptowährungen durchgeführt!")
            else:
                self.update_status("Backtest fehlgeschlagen - Keine Ergebnisse")
                messagebox.showerror("Fehler", "Backtest konnte keine Daten abrufen!")
                
        except Exception as e:
            self.update_status(f"Update Fehler: {str(e)}")

    def close_all_trades(self):
        """Schließt alle aktiven Trades"""
        if not self.bot.active_trades:
            messagebox.showinfo("Info", "Keine aktiven Trades")
            return
            
        result = messagebox.askyesno(
            "Alle Trades schliessen", 
            "Möchten Sie wirklich alle aktiven Trades schliessen?"
        )
        
        if result:
            for symbol in list(self.bot.active_trades.keys()):
                self.bot.close_trade(symbol, "MANUELL GESCHLOSSEN")
                
            self.update_active_trades()
            messagebox.showinfo("Erfolg", "Alle Trades geschlossen")
        
    def close_selected_trade(self):
        """Schließt den ausgewählten Trade"""
        if not hasattr(self, 'trades_tree'):
            return
            
        selection = self.trades_tree.selection()
        if not selection:
            messagebox.showwarning("Warnung", "Bitte wählen Sie einen Trade aus")
            return
            
        item = self.trades_tree.item(selection[0])
        symbol = item['values'][0]
        
        self.bot.close_trade(symbol, "MANUELL GESCHLOSSEN")
        self.update_active_trades()

    def update_balance_display(self):
        """Aktualisiert die Kontostand-Anzeige mit echten Daten"""
        def update():
            try:
                balance_summary = self.bot.get_balance_summary()
                
                if balance_summary:
                    # Gesamtportfolio Wert
                    total_value = balance_summary['total_portfolio_value']
                    last_updated = balance_summary['last_updated'].strftime('%H:%M:%S')
                    
                    if hasattr(self, 'screen_width') and self.screen_width >= 1280:
                        self.balance_info_var.set(
                            f"Gesamtportfolio: ${total_value:,.2f} (Stand: {last_updated})"
                        )
                        
                        # Detaillierte Bestände für große GUI
                        if hasattr(self, 'balance_tree'):
                            for item in self.balance_tree.get_children():
                                self.balance_tree.delete(item)
                            
                            for asset in balance_summary['assets']:
                                tags = ()
                                if asset['currency'] == 'USDT':
                                    tags = ('usdt',)
                                
                                self.balance_tree.insert('', tk.END, values=(
                                    asset['currency'],
                                    f"{asset['balance']:.6f}",
                                    f"{asset['available']:.6f}",
                                    f"${asset['price']:.6f}" if asset['currency'] != 'USDT' else "$1.000000",
                                    f"${asset['value_usd']:,.2f}",
                                    f"{asset['percentage']:.1f}%"
                                ), tags=tags)
                            
                            self.balance_tree.tag_configure('usdt', background='#e8f4fd')
                    else:
                        # Kleine GUI
                        self.balance_info_var.set(
                            f"Portfolio: ${total_value:,.2f} ({last_updated})"
                        )
                    
                else:
                    self.balance_info_var.set("Keine Kontostandsdaten verfügbar")
                    
            except Exception as e:
                self.balance_info_var.set(f"Fehler: {str(e)}")
                print(f"Balance update error: {e}")
        
        threading.Thread(target=update, daemon=True).start()

    def force_cache_update(self):
        """Erzwingt eine Cache-Aktualisierung"""
        def update():
            self.bot.update_caches()
            self.update_balance_display()
            self.update_recommendations_with_actions()
            self.update_active_trades()
            
        threading.Thread(target=update, daemon=True).start()
        self.update_status("Cache wird aktualisiert...")

    def clear_activity_log(self):
        """Leert den Aktivitätslog"""
        if hasattr(self, 'activity_log') and self.activity_log is not None:
            self.activity_log.config(state=tk.NORMAL)
            self.activity_log.delete(1.0, tk.END)
            self.activity_log.config(state=tk.DISABLED)
        self.bot_activity_log = []
        self.update_status("Aktivitätslog geleert")

    def update_bot_activity(self, message):
        """Fügt eine neue Aktivitätsnachricht hinzu"""
        if hasattr(self, 'activity_log') and self.activity_log is not None:
            try:
                self.activity_log.config(state=tk.NORMAL)
                self.activity_log.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {message}\n")
                self.activity_log.see(tk.END)
                self.activity_log.config(state=tk.DISABLED)
            except Exception as e:
                print(f"Activity log update error: {e}")
        
        # Immer zur internen Liste hinzufügen
        self.bot_activity_log.append(f"{datetime.now().strftime('%H:%M:%S')} - {message}")

    def update_status(self, message):
        """Aktualisiert die Status-Anzeige"""
        if hasattr(self, 'status_var'):
            self.status_var.set(message)
        print(f"Status: {message}")

    def start_auto_updates(self):
        """Startet automatische Updates - OPTIMIERT"""
        def auto_update_loop():
            """Interne Update-Schleife"""
            while True:
                try:
                    # NUR wichtige Updates im Hintergrund
                    self.root.after(0, self.update_balance_display)
                    self.root.after(0, self.update_recommendations_with_actions)
                    self.root.after(0, self.update_active_trades)
                    if hasattr(self, 'bot_status_vars'):
                        self.root.after(0, self.update_bot_status)
                except Exception as e:
                    print(f"Auto-update error: {e}")
                time.sleep(30)  # Alle 30 Sekunden updaten
                
        threading.Thread(target=auto_update_loop, daemon=True).start()
        print("✅ Auto-Updates gestartet (ohne Finanzamt-Log)")

    def on_tab_changed(self, event):
        """Wird aufgerufen wenn Tab gewechselt wird"""
        try:
            notebook = event.widget
            current_tab = notebook.select()
            tab_text = notebook.tab(current_tab, "text")
            
            if tab_text == "Finanzamt":
                # Finanzamt-Tab wurde aktiviert -> einmalig aktualisieren
                print("🔄 Aktualisiere Finanzamt-Tab...")
                self.update_tax_log()
                self.update_trade_history()
        except Exception as e:
            print(f"Tab change error: {e}")

    def force_tax_update(self):
        """Aktualisiert Finanzamt-Daten manuell"""
        self.update_status("Aktualisiere Finanzamt-Daten...")
        self.update_tax_log()
        self.update_trade_history()
        self.update_status("Finanzamt-Daten aktualisiert")

    def update_finanzamt_on_demand(self):
        """Aktualisiert Finanzamt-Tab nur bei Bedarf"""
        if hasattr(self, 'tax_tree') and self.tax_tree.winfo_ismapped():
            # Nur aktualisieren wenn Tab sichtbar ist
            self.update_tax_log()
            self.update_trade_history()

    # Tax und Debug Methoden
    def generate_tax_report(self):
        """Generiert einen detaillierten Steuerreport"""
        report_window = tk.Toplevel(self.root)
        report_window.title("Steuerreport Generieren")
        report_window.geometry("400x200")
        
        ttk.Label(report_window, text="Startdatum (YYYY-MM-DD):").pack(pady=5)
        start_entry = ttk.Entry(report_window, width=20)
        start_entry.pack(pady=5)
        start_entry.insert(0, (datetime.now().replace(day=1)).strftime('%Y-%m-%d'))  # Erster des Monats
        
        ttk.Label(report_window, text="Enddatum (YYYY-MM-DD):").pack(pady=5)
        end_entry = ttk.Entry(report_window, width=20)
        end_entry.pack(pady=5)
        end_entry.insert(0, datetime.now().strftime('%Y-%m-%d'))
        
        def generate():
            start_date = start_entry.get()
            end_date = end_entry.get()
            
            try:
                # Vereinfachter Report für Demo
                recent_trades = self.bot.tax_logger.get_recent_trades(1000)
                
                if not recent_trades:
                    messagebox.showinfo("Info", "Keine Trades im ausgewählten Zeitraum")
                    return
                    
                trades_in_period = []
                for trade in recent_trades:
                    try:
                        trade_date = datetime.fromisoformat(trade['timestamp_iso']).date()
                        start = datetime.strptime(start_date, '%Y-%m-%d').date()
                        end = datetime.strptime(end_date, '%Y-%m-%d').date()
                        
                        if start <= trade_date <= end:
                            trades_in_period.append(trade)
                    except:
                        continue
                
                if not trades_in_period:
                    messagebox.showinfo("Info", "Keine Trades im ausgewählten Zeitraum")
                    return
                    
                total_trades = len(trades_in_period)
                buy_trades = len([t for t in trades_in_period if t['side'] == 'BUY'])
                sell_trades = len([t for t in trades_in_period if t['side'] == 'SELL'])
                total_volume = sum(t['total_value'] for t in trades_in_period)
                total_profit = sum(t['profit_loss'] for t in trades_in_period if t['profit_loss'] > 0)
                total_loss = abs(sum(t['profit_loss'] for t in trades_in_period if t['profit_loss'] < 0))
                net_profit = total_profit - total_loss
                
                report_text = f"""📊 Steuerreport für {start_date} bis {end_date}
                    
📈 Handelsaktivität:
• Gesamte Trades: {total_trades}
• Kauf-Trades: {buy_trades}
• Verkauf-Trades: {sell_trades}

💰 Finanzielle Übersicht:
• Handelsvolumen: ${total_volume:,.2f}
• Gesamtgewinn: ${total_profit:,.2f}
• Gesamtverlust: ${total_loss:,.2f}
• Netto Gewinn/Verlust: ${net_profit:+,.2f}
"""
                
                messagebox.showinfo("Steuerreport", report_text)
                    
            except Exception as e:
                messagebox.showerror("Fehler", f"Fehler beim Generieren: {e}")
                
            report_window.destroy()
        
        ttk.Button(report_window, text="Report Generieren", command=generate).pack(pady=10)

    def export_logs(self):
        """Exportiert die Logs"""
        try:
            import shutil
            export_dir = f"finanzamt_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(export_dir)
            
            # Kopiere alle relevanten Dateien
            if os.path.exists("trade_logs/trades_finanzamt.csv"):
                shutil.copy2("trade_logs/trades_finanzamt.csv", export_dir)
            if os.path.exists("trade_logs/trading_history.json"):
                shutil.copy2("trade_logs/trading_history.json", export_dir)
            
            messagebox.showinfo("Erfolg", f"Logs exportiert nach: {export_dir}")
        except Exception as e:
            messagebox.showerror("Fehler", f"Export fehlgeschlagen: {e}")

    def show_tax_logs(self):
        """Zeigt die Steuerlogs an"""
        self.update_tax_log()

    def update_tax_log(self):
        """Aktualisiert die Finanzamt-Log Anzeige"""
        if not hasattr(self, 'tax_tree'):
            return
            
        try:
            # Lösche vorhandene Einträge
            for item in self.tax_tree.get_children():
                self.tax_tree.delete(item)
            
            print("🔄 Aktualisiere Finanzamt-Log...")
            
            # Hole aktuelle Trades vom Tax-Logger
            recent_trades = self.bot.tax_logger.get_recent_trades(100)
            
            if not recent_trades:
                print("ℹ️  Keine Trades im Finanzamt-Log verfügbar")
                self.tax_tree.insert('', tk.END, values=(
                    "Keine", "Trades", "verfügbar", "-", "-", "-", "-", "-", "-"
                ))
                return
                
            print(f"📋 Zeige {len(recent_trades)} Trades im Finanzamt-Tab an")
            
            # Zeige Trades in umgekehrter Reihenfolge (älteste zuerst)
            for trade in reversed(recent_trades):
                try:
                    timestamp = trade.get('timestamp', 'Unbekannt')
                    side = trade.get('side', 'UNKNOWN')
                    symbol = trade.get('symbol', 'Unknown')
                    amount = trade.get('amount', 0)
                    price = trade.get('price', 0)
                    total_value = trade.get('total_value', 0)
                    fees = trade.get('fees', 0)
                    profit_loss = trade.get('profit_loss', 0)
                    reason = trade.get('reason', '')
                    
                    # Debug-Ausgabe für ersten Trade
                    if trade == recent_trades[0]:
                        print(f"🔍 Erster Finanzamt-Trade: {timestamp} - {symbol} - {side} - ${profit_loss:.2f}")
                    
                    # Berechne Gesamtbetrag falls nicht vorhanden
                    if total_value == 0 and amount > 0 and price > 0:
                        total_value = amount * price
                    
                    # Berechne Gebühren falls nicht vorhanden
                    if fees == 0 and total_value > 0:
                        fees = total_value * 0.001
                    
                    # Bestimme Tags für Farbgebung
                    tags = ()
                    if profit_loss > 0:
                        tags = ('profit',)
                    elif profit_loss < 0:
                        tags = ('loss',)
                    elif side == 'BUY':
                        tags = ('buy',)
                    else:
                        tags = ('neutral',)
                    
                    # Füge Trade zur Treeview hinzu
                    self.tax_tree.insert('', tk.END, values=(
                        str(timestamp)[:16],  # Kürze auf 16 Zeichen
                        str(side),
                        str(symbol),
                        f"{float(amount):.6f}",
                        f"${float(price):.6f}",
                        f"${float(total_value):.2f}",
                        f"${float(fees):.2f}",
                        f"${float(profit_loss):+.2f}",
                        str(reason)[:30]  # Kürze Grund auf 30 Zeichen
                    ), tags=tags)
                    
                except Exception as e:
                    print(f"⚠️  Fehler beim Anzeigen des Finanzamt-Trades: {e}")
                    continue
                    
            # Konfiguriere Tags für Farben
            self.tax_tree.tag_configure('buy', background='#e8f4fd')
            self.tax_tree.tag_configure('profit', background='#d4edda')
            self.tax_tree.tag_configure('loss', background='#f8d7da')
            self.tax_tree.tag_configure('neutral', background='#fff3cd')
            
            # Aktualisiere Portfolio-Informationen
            self.update_portfolio_info(recent_trades)
            
            print("✅ Finanzamt-Log erfolgreich aktualisiert")
            
        except Exception as e:
            print(f"❌ Fehler beim Aktualisieren des Finanzamt-Logs: {e}")

    def update_portfolio_info(self, recent_trades=None):
        """Aktualisiert Portfolio-Informationen"""
        try:
            if recent_trades is None:
                recent_trades = self.bot.tax_logger.get_recent_trades(1000)  # Alle Trades
            
            # Berechne Portfolio-Wert
            portfolio_value = self.bot.calculate_portfolio_value()
            
            # Berechne Gesamtgewinn (nur aus verkauften Trades)
            total_profit = 0
            for trade in recent_trades:
                if trade.get('side') == 'SELL':
                    profit_loss = trade.get('profit_loss', 0)
                    if profit_loss is not None:
                        total_profit += profit_loss
            
            # Aktualisiere Anzeige
            if hasattr(self, 'portfolio_var'):
                self.portfolio_var.set(f"Portfolio Wert: ${portfolio_value:,.2f}")
            
            if hasattr(self, 'total_profit_var'):
                self.total_profit_var.set(f"Gesamtgewinn: ${total_profit:+.2f}")
                
            print(f"💰 Portfolio: ${portfolio_value:,.2f}, Gewinn: ${total_profit:+.2f}")
            
        except Exception as e:
            print(f"❌ Fehler beim Aktualisieren der Portfolio-Info: {e}")

    def debug_trade_history(self):
        """Debug-Funktion zur Überprüfung der Trade-History"""
        try:
            print("🔍 DEBUG: Trade-History Überprüfung")
            print(f"Bot Trade-History: {len(self.bot.trade_history)} Trades")
            
            # Zeige alle Trades an
            for i, trade in enumerate(self.bot.trade_history):
                print(f"  {i+1}: {trade.get('symbol')} {trade.get('side')} {trade.get('timestamp')}")
                
            # Überprüfe JSON-Datei
            json_path = "trade_logs/trading_history.json"
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_trades = json.load(f)
                print(f"JSON Datei: {len(json_trades)} Trades")
                
            messagebox.showinfo("Debug", f"Trade-History: {len(self.bot.trade_history)} Trades geladen")
            
        except Exception as e:
            messagebox.showerror("Debug Fehler", f"Fehler: {e}")

    def debug_status(self):
        """Zeigt Debug-Informationen an"""
        debug_info = f"""Debug Informationen:
Auto-Trading: {self.bot.auto_trading}
Aktive Trades: {len(self.bot.active_trades)}
Trading-Pairs: {self.bot.trading_pairs}
Empfehlungen: {len(self.bot.current_recommendations)}
Letztes Update: {self.bot.last_update}
Display Auflösung: {self.screen_width}x{self.screen_height}
"""
        messagebox.showinfo("Debug Info", debug_info)
        
    def run(self):
        """Startet die GUI-Hauptschleife"""
        # Initiale Aktualisierungen
        self.update_balance_display()
        if hasattr(self, 'selected_listbox'):
            self.load_current_pairs()
            self.load_available_pairs()
        
        self.root.mainloop()

    def update_trade_history(self):
        """Aktualisiert die Trade-History Anzeige - VOLLSTÄNDIG KORRIGIERT"""
        if not hasattr(self, 'history_tree'):
            print("❌ history_tree nicht verfügbar")
            return
            
        try:
            # Lösche vorhandene Einträge
            for item in self.history_tree.get_children():
                self.history_tree.delete(item)
            
            print("🔄 Aktualisiere Trade-History in GUI...")
            
            # Hole Trade-History vom Bot
            trade_history = self.bot.get_trade_history_for_gui(50)
            
            if not trade_history:
                print("ℹ️  Keine Trades für GUI verfügbar")
                self.history_tree.insert('', tk.END, values=(
                    "Keine", "Trades", "verfügbar", "-", "-", "-", "-", "-"
                ))
                return
                
            print(f"📋 Zeige {len(trade_history)} Trades in GUI an")
            
            # Zeige Trades in umgekehrter Reihenfolge (älteste zuerst)
            for trade in reversed(trade_history):
                try:
                    timestamp = trade.get('timestamp', 'Unbekannt')
                    symbol = trade.get('symbol', 'Unknown')
                    side = trade.get('side', 'UNKNOWN')
                    price = trade.get('price', 0)
                    amount = trade.get('amount', 0)
                    profit_loss = trade.get('profit_loss', 0)
                    profit_loss_percent = trade.get('profit_loss_percent', 0)
                    reason = trade.get('reason', '')
                    
                    # Debug-Ausgabe für ersten Trade
                    if trade == trade_history[0]:
                        print(f"🔍 Erster Trade: {timestamp} - {symbol} - {side} - ${price}")
                    
                    # Bestimme Tags für Farbgebung
                    tags = ()
                    if side == 'BUY':
                        tags = ('buy',)
                    elif side == 'SELL':
                        if profit_loss > 0:
                            tags = ('profit',)
                        else:
                            tags = ('loss',)
                    else:
                        tags = ('neutral',)
                    
                    # Füge Trade zur Treeview hinzu
                    self.history_tree.insert('', tk.END, values=(
                        str(timestamp)[:16],  # Kürze auf 16 Zeichen
                        str(symbol),
                        str(side),
                        f"${float(price):.6f}",
                        f"{float(amount):.6f}",
                        f"{float(profit_loss_percent):+.2f}%",
                        f"${float(profit_loss):+.2f}",
                        str(reason)[:30]  # Kürze Grund auf 30 Zeichen
                    ), tags=tags)
                    
                except Exception as e:
                    print(f"⚠️  Fehler beim Anzeigen des Trades: {e}")
                    continue
                    
            # Konfiguriere Tags für Farben
            self.history_tree.tag_configure('buy', background='#d4edda')
            self.history_tree.tag_configure('profit', background='#e8f5e8')
            self.history_tree.tag_configure('loss', background='#f8d7da')
            self.history_tree.tag_configure('neutral', background='#fff3cd')
            
            print("✅ Trade-History erfolgreich aktualisiert")
            
        except Exception as e:
            print(f"❌ Fehler beim Aktualisieren der Trade-History: {e}")

    def force_history_update(self):
        """Erzwingt eine Aktualisierung der Trade-History"""
        print("🔄 Erzwinge Trade-History Update...")
        
        def update():
            try:
                # Lade History neu vom Bot
                self.bot.load_trade_history()
                
                # Aktualisiere GUI
                self.root.after(0, self.update_trade_history)
                self.root.after(0, self.update_tax_log)
                
                self.update_status("Trade-History aktualisiert")
                print("✅ Trade-History erzwungenes Update abgeschlossen")
                
            except Exception as e:
                error_msg = f"Fehler beim History-Update: {e}"
                print(f"❌ {error_msg}")
                self.root.after(0, lambda: self.update_status(error_msg))
        
        threading.Thread(target=update, daemon=True).start()

    def force_api_stats_update(self):
        """Erzwingt Aktualisierung der API-Statistiken"""
        try:
            # Test-API Aufruf
            self.bot.api.get_ticker("BTC-USDT")
            
            # Statistiken aktualisieren
            api_stats = self.bot.api.get_api_stats()
            if api_stats and hasattr(self, 'bot_status_vars'):
                if 'total_requests' in self.bot_status_vars:
                    self.bot_status_vars['total_requests'].set(str(api_stats['request_count']))
                if 'last_request' in self.bot_status_vars:
                    self.bot_status_vars['last_request'].set(
                        api_stats['last_request_time'] if api_stats['last_request_time'] != '-' 
                        else datetime.now().strftime('%H:%M:%S')
                    )
                    
            self.update_status("API-Statistiken aktualisiert")
            
        except Exception as e:
            self.update_status(f"API-Statistiken Fehler: {e}")

# =============================================================================
# HEADLESS MODUS
# =============================================================================

class HeadlessTradingBot:
    def __init__(self, bot):
        self.bot = bot
        self.bot.set_headless_reference(self)
        self.setup_schedule()
        
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

# =============================================================================
# HAUPTPROGRAMM
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

        # Debug: Prüfe ob Trades geladen wurden
        print(f"📊 Bot initialisiert mit {len(bot.trade_history)} Trades in History")
        
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

def main_headless():
    """Headless Modus für Server-Betrieb"""
    print("🚀 Starte KuCoin Trading Bot - Headless Modus...")
    
    # Lade API-Daten aus .env Datei
    env_vars = load_env_file()
    
    API_KEY = env_vars.get('KUCOIN_API_KEY')
    API_SECRET = env_vars.get('KUCOIN_API_SECRET')
    API_PASSPHRASE = env_vars.get('KUCOIN_API_PASSPHRASE')
    SANDBOX = env_vars.get('KUCOIN_SANDBOX', 'False').lower() == 'true'
    
    if not all([API_KEY, API_SECRET, API_PASSPHRASE]):
        print("❌ Fehler: API-Daten nicht gefunden!")
        return
    
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
    # Entscheide ob GUI oder Headless Modus
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--headless":
        main_headless()
    else:
        main()