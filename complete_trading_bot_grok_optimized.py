# complete_trading_bot.py
import os
import warnings
warnings.filterwarnings('ignore')

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
from datetime import datetime, timedelta
import hmac
import hashlib
import base64
import json
import requests
from urllib.parse import urlencode
import csv
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Try to import TA-Lib
try:
    import talib
    TA_LIB_AVAILABLE = True
    print("✅ TA-Lib verfügbar - Verwende optimierte technische Indikatoren")
except ImportError:
    TA_LIB_AVAILABLE = False
    print("⚠️  TA-Lib nicht verfügbar - Verwende manuelle Berechnungen")

# =============================================================================
# TECHNICAL ANALYSIS CACHE
# =============================================================================

class TechnicalAnalysisCache:
    """Intelligenter Cache für technische Indikatoren"""
    
    def __init__(self, ttl_minutes=10):
        self.cache = {}
        self.ttl = timedelta(minutes=ttl_minutes)
        self.analysis_cache = {}  # Spezieller Cache für Analyse-Ergebnisse
        
    def get_analysis(self, symbol, interval):
        """Holt Analyse-Ergebnisse aus Cache"""
        cache_key = f"analysis_{symbol}_{interval}"
        if cache_key in self.analysis_cache:
            data, timestamp = self.analysis_cache[cache_key]
            if datetime.now() - timestamp < self.ttl:
                return data
        return None
    
    def set_analysis(self, symbol, interval, data):
        """Speichert Analyse-Ergebnisse im Cache"""
        cache_key = f"analysis_{symbol}_{interval}"
        self.analysis_cache[cache_key] = (data, datetime.now())
    
    def get_indicator(self, symbol, indicator_name, period=None):
        """Holt spezifischen Indikator aus Cache"""
        cache_key = f"indicator_{symbol}_{indicator_name}_{period}" if period else f"indicator_{symbol}_{indicator_name}"
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < self.ttl:
                return data
        return None
    
    def set_indicator(self, symbol, indicator_name, data, period=None):
        """Speichert spezifischen Indikator im Cache"""
        cache_key = f"indicator_{symbol}_{indicator_name}_{period}" if period else f"indicator_{symbol}_{indicator_name}"
        self.cache[cache_key] = (data, datetime.now())
    
    def clear_expired(self):
        """Räumt abgelaufene Cache-Einträge auf"""
        current_time = datetime.now()
        
        # Lösche abgelaufene Analyse-Cache-Einträge
        expired_analysis = [
            key for key, (data, timestamp) in self.analysis_cache.items() 
            if current_time - timestamp > self.ttl
        ]
        for key in expired_analysis:
            del self.analysis_cache[key]
        
        # Lösche abgelaufene Indikator-Cache-Einträge
        expired_indicators = [
            key for key, (data, timestamp) in self.cache.items() 
            if current_time - timestamp > self.ttl
        ]
        for key in expired_indicators:
            del self.cache[key]

# =============================================================================
# ACTIVE TRADES STORAGE
# =============================================================================

class ActiveTradesStorage:
    """Persistente Speicherung aktiver Trades"""
    
    def __init__(self, storage_file="active_trades.json"):
        self.storage_file = storage_file
        self.ensure_storage_file()
    
    def ensure_storage_file(self):
        """Stellt sicher, dass die Storage-Datei existiert"""
        if not os.path.exists(self.storage_file):
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump({}, f, indent=2, ensure_ascii=False)
            print(f"✅ Neue Active-Trades-Storage Datei erstellt: {self.storage_file}")
    
    def save_active_trades(self, active_trades):
        """Speichert aktive Trades persistent"""
        try:
            # Konvertiere datetime-Objekte zu Strings für JSON
            serializable_trades = {}
            for symbol, trade in active_trades.items():
                serializable_trade = trade.copy()
                # Konvertiere datetime zu ISO-String
                if 'timestamp' in serializable_trade and isinstance(serializable_trade['timestamp'], datetime):
                    serializable_trade['timestamp'] = serializable_trade['timestamp'].isoformat()
                serializable_trades[symbol] = serializable_trade
            
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_trades, f, indent=2, ensure_ascii=False)
            
            print(f"✅ {len(serializable_trades)} aktive Trades gespeichert")
            return True
        except Exception as e:
            print(f"❌ Fehler beim Speichern der aktiven Trades: {e}")
            return False
    
    def load_active_trades(self):
        """Lädt aktive Trades aus persistentem Storage"""
        try:
            if not os.path.exists(self.storage_file):
                return {}
            
            with open(self.storage_file, 'r', encoding='utf-8') as f:
                loaded_trades = json.load(f)
            
            # Konvertiere ISO-Strings zurück zu datetime-Objekten
            active_trades = {}
            for symbol, trade in loaded_trades.items():
                restored_trade = trade.copy()
                if 'timestamp' in restored_trade and isinstance(restored_trade['timestamp'], str):
                    restored_trade['timestamp'] = datetime.fromisoformat(restored_trade['timestamp'])
                active_trades[symbol] = restored_trade
            
            print(f"✅ {len(active_trades)} aktive Trades geladen")
            return active_trades
        except Exception as e:
            print(f"❌ Fehler beim Laden der aktiven Trades: {e}")
            return {}
    
    def clear_active_trades(self):
        """Löscht alle gespeicherten aktiven Trades"""
        try:
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump({}, f, indent=2, ensure_ascii=False)
            print("✅ Aktive Trades Storage geleert")
            return True
        except Exception as e:
            print(f"❌ Fehler beim Leeren der aktiven Trades: {e}")
            return False

# =============================================================================
# CACHE SYSTEM
# =============================================================================

class DataCache:
    """Intelligenter Cache für Kursdaten und Analysen"""
    
    def __init__(self, ttl_minutes=5):
        self.cache = {}
        self.ttl = timedelta(minutes=ttl_minutes)
        
    def get(self, key):
        """Holt Daten aus Cache wenn gültig"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                return data
            else:
                del self.cache[key]
        return None
    
    def set(self, key, data):
        """Speichert Daten im Cache"""
        self.cache[key] = (data, datetime.now())
    
    def clear_expired(self):
        """Räumt abgelaufene Cache-Einträge auf"""
        current_time = datetime.now()
        expired_keys = [
            key for key, (data, timestamp) in self.cache.items() 
            if current_time - timestamp > self.ttl
        ]
        for key in expired_keys:
            del self.cache[key]

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
        """Validiert die Order-Größe gemäß KuCoin's Anforderungen mit detaillierter Fehlerausgabe"""
        # Lade Symbol-Informationen falls nicht vorhanden
        if symbol not in self.symbols_info:
            if not self.get_symbols_info():
                return False, "Konnte Symbol-Informationen nicht laden"
        
        symbol_info = self.symbols_info.get(symbol)
        if not symbol_info:
            return False, f"Unbekanntes Symbol: {symbol}"
        
        error_messages = []
        
        # Prüfe Mindestgröße
        min_size = symbol_info.get('baseMinSize', 0)
        if size < min_size:
            error_messages.append(f"Größe zu klein. Minimum: {min_size}")
        
        # Prüfe Maximale Größe
        max_size = symbol_info.get('baseMaxSize', float('inf'))
        if size > max_size:
            error_messages.append(f"Größe zu groß. Maximum: {max_size}")
        
        # Prüfe Inkrement (Step-Größe)
        base_increment = symbol_info.get('baseIncrement', 0)
        if base_increment > 0:
            # Vermeide Division durch Null
            if base_increment > 0:
                steps = size / base_increment
                if not (abs(steps - round(steps)) < 1e-10):  # Toleranz für Floating-Point
                    valid_steps = round(steps)
                    valid_size = round(valid_steps * base_increment, 8)
                    error_messages.append(f"Ungültige Schrittgröße. Verwende: {valid_size}")
        
        # Prüfe Mindestbetrag wenn Preis gegeben
        if price is not None:
            min_funds = symbol_info.get('minFunds', 0)
            order_value = size * price
            if min_funds > 0 and order_value < min_funds:
                error_messages.append(f"Order-Wert zu klein. Minimum: {min_funds} USDT (aktuell: {order_value:.2f} USDT)")
        
        if error_messages:
            return False, " | ".join(error_messages)
        
        return True, "Validierung erfolgreich"
    
    def calculate_valid_size(self, symbol, desired_size):
        """Berechnet eine gültige Order-Größe basierend auf den Symbol-Regeln"""
        if symbol not in self.symbols_info:
            self.get_symbols_info()
        
        symbol_info = self.symbols_info.get(symbol)
        if not symbol_info:
            return desired_size
        
        base_increment = symbol_info.get('baseIncrement', 0)
        base_min_size = symbol_info.get('baseMinSize', 0)
        base_max_size = symbol_info.get('baseMaxSize', float('inf'))
        
        # Stelle sicher, dass die Größe innerhalb der Grenzen liegt
        if desired_size < base_min_size:
            desired_size = base_min_size
        elif desired_size > base_max_size:
            desired_size = base_max_size
        
        # Runde auf das nächste gültige Inkrement
        if base_increment > 0:
            steps = desired_size / base_increment
            valid_steps = round(steps)
            valid_size = round(valid_steps * base_increment, 8)
            
            # Stelle sicher, dass die gerundete Größe innerhalb der Grenzen bleibt
            if valid_size < base_min_size:
                valid_size = base_min_size
            elif valid_size > base_max_size:
                valid_size = base_max_size
                
            return valid_size
        
        return desired_size
    
    def print_symbol_info(self, symbol):
        """Gibt detaillierte Symbol-Informationen aus (für Debugging)"""
        if symbol not in self.symbols_info:
            self.get_symbols_info()
        
        symbol_info = self.symbols_info.get(symbol)
        if symbol_info:
            print(f"📋 Symbol Info für {symbol}:")
            for key, value in symbol_info.items():
                print(f"   {key}: {value}")
        else:
            print(f"❌ Keine Informationen für {symbol} verfügbar")

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

    def get_multiple_tickers(self, symbols):
        """Holt Preise für mehrere Symbole parallel"""
        def fetch_single_ticker(symbol):
            return symbol, self.get_ticker(symbol)
        
        results = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(fetch_single_ticker, symbol) for symbol in symbols]
            
            for future in as_completed(futures):
                try:
                    symbol, price = future.result()
                    if price:
                        results[symbol] = price
                except Exception as e:
                    continue
                
        return results

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
# TECHNICAL ANALYSIS ENGINE - OPTIMIERT MIT CACHING
# =============================================================================

class TechnicalAnalysis:
    """Optimierte technische Analyse mit NumPy, TA-Lib und intelligentem Caching"""
    
    def __init__(self):
        self.cache = TechnicalAnalysisCache(ttl_minutes=15)  # Längeres TTL für Analysen
        self.last_prices = {}  # Cache für letzte Preisdaten pro Symbol
        
    def calculate_rsi(self, prices, period=14, symbol=None):
        """Berechnet RSI mit Caching und inkrementeller Aktualisierung"""
        if symbol:
            # Versuche aus Cache zu laden
            cached_rsi = self.cache.get_indicator(symbol, 'rsi', period)
            if cached_rsi is not None:
                return cached_rsi
        
        if len(prices) < period + 1:
            return 50
            
        prices_array = np.array(prices, dtype=np.float64)
        
        if TA_LIB_AVAILABLE:
            try:
                rsi = talib.RSI(prices_array, timeperiod=period)
                result = float(rsi[-1]) if not np.isnan(rsi[-1]) else 50
            except:
                result = self._calculate_rsi_manual(prices, period)
        else:
            result = self._calculate_rsi_manual(prices, period)
        
        # Speichere im Cache falls Symbol gegeben
        if symbol:
            self.cache.set_indicator(symbol, 'rsi', result, period)
        
        return result
    
    def _calculate_rsi_manual(self, prices, period=14):
        """Manuelle RSI-Berechnung als Fallback"""
        if len(prices) < period + 1:
            return 50
            
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        if len(gains) < period or len(losses) < period:
            return 50
            
        avg_gains = np.convolve(gains, np.ones(period)/period, mode='valid')
        avg_losses = np.convolve(losses, np.ones(period)/period, mode='valid')
        
        if len(avg_gains) == 0 or len(avg_losses) == 0:
            return 50
            
        if avg_losses[-1] == 0:
            return 100 if avg_gains[-1] > 0 else 50
            
        rs = avg_gains[-1] / avg_losses[-1]
        rsi = 100 - (100 / (1 + rs))
        
        return min(max(rsi, 0), 100)
    
    def calculate_moving_averages(self, prices, periods=[10, 20, 50], symbol=None):
        """Berechnet mehrere gleitende Durchschnitte gleichzeitig mit Caching"""
        if symbol:
            # Versuche aus Cache zu laden
            cache_key = f"ma_{min(periods)}_{max(periods)}"
            cached_ma = self.cache.get_indicator(symbol, cache_key)
            if cached_ma is not None:
                return cached_ma
        
        if len(prices) < max(periods):
            result = {f"ma_{period}": np.mean(prices) if prices else 0 for period in periods}
        else:
            prices_array = np.array(prices, dtype=np.float64)
            results = {}
            
            for period in periods:
                if TA_LIB_AVAILABLE:
                    try:
                        ma = talib.SMA(prices_array, timeperiod=period)
                        results[f"ma_{period}"] = float(ma[-1]) if not np.isnan(ma[-1]) else np.mean(prices_array[-period:])
                    except:
                        results[f"ma_{period}"] = np.mean(prices_array[-period:])
                else:
                    results[f"ma_{period}"] = np.mean(prices_array[-period:])
                    
            result = results
        
        # Speichere im Cache falls Symbol gegeben
        if symbol:
            cache_key = f"ma_{min(periods)}_{max(periods)}"
            self.cache.set_indicator(symbol, cache_key, result)
        
        return result
    
    def calculate_macd(self, prices, fastperiod=12, slowperiod=26, signalperiod=9, symbol=None):
        """Berechnet MACD Indikator mit Caching"""
        if symbol:
            # Versuche aus Cache zu laden
            cached_macd = self.cache.get_indicator(symbol, 'macd')
            if cached_macd is not None:
                return cached_macd
        
        if len(prices) < slowperiod + signalperiod:
            result = {'macd': 0, 'signal': 0, 'histogram': 0}
        else:
            prices_array = np.array(prices, dtype=np.float64)
            
            if TA_LIB_AVAILABLE:
                try:
                    macd, signal, histogram = talib.MACD(prices_array, 
                                                       fastperiod=fastperiod, 
                                                       slowperiod=slowperiod, 
                                                       signalperiod=signalperiod)
                    result = {
                        'macd': float(macd[-1]) if not np.isnan(macd[-1]) else 0,
                        'signal': float(signal[-1]) if not np.isnan(signal[-1]) else 0,
                        'histogram': float(histogram[-1]) if not np.isnan(histogram[-1]) else 0
                    }
                except:
                    result = self._calculate_macd_manual(prices, fastperiod, slowperiod, signalperiod)
            else:
                result = self._calculate_macd_manual(prices, fastperiod, slowperiod, signalperiod)
        
        # Speichere im Cache falls Symbol gegeben
        if symbol:
            self.cache.set_indicator(symbol, 'macd', result)
        
        return result
    
    def _calculate_macd_manual(self, prices, fastperiod=12, slowperiod=26, signalperiod=9):
        """Manuelle MACD Berechnung als Fallback"""
        if len(prices) < slowperiod + signalperiod:
            return {'macd': 0, 'signal': 0, 'histogram': 0}
            
        # Vereinfachte EMA-Berechnung
        def calculate_ema(data, period):
            if len(data) < period:
                return np.array([])
            weights = np.exp(np.linspace(-1., 0., period))
            weights /= weights.sum()
            ema = np.convolve(data, weights, mode='valid')
            return ema
        
        ema_fast = calculate_ema(prices, fastperiod)
        ema_slow = calculate_ema(prices, slowperiod)
        
        if len(ema_fast) == 0 or len(ema_slow) == 0:
            return {'macd': 0, 'signal': 0, 'histogram': 0}
            
        macd_line = ema_fast[-1] - ema_slow[-1]
        
        # Für Signal-Line verwenden wir den letzten MACD Wert
        macd_signal = calculate_ema(np.array([macd_line]), signalperiod)
        if len(macd_signal) > 0:
            signal_line = macd_signal[-1]
        else:
            signal_line = macd_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': macd_line - signal_line
        }
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2, symbol=None):
        """Berechnet Bollinger Bands mit Caching"""
        if symbol:
            # Versuche aus Cache zu laden
            cached_bb = self.cache.get_indicator(symbol, 'bollinger_bands', period)
            if cached_bb is not None:
                return cached_bb
        
        if len(prices) < period:
            result = {'upper': prices[-1] if prices else 0, 'middle': prices[-1] if prices else 0, 'lower': prices[-1] if prices else 0}
        else:
            prices_array = np.array(prices[-period:], dtype=np.float64)
            middle = np.mean(prices_array)
            std = np.std(prices_array)
            
            result = {
                'upper': middle + (std * std_dev),
                'middle': middle,
                'lower': middle - (std * std_dev)
            }
        
        # Speichere im Cache falls Symbol gegeben
        if symbol:
            self.cache.set_indicator(symbol, 'bollinger_bands', result, period)
        
        return result
    
    def update_price_cache(self, symbol, new_price):
        """Aktualisiert den Preis-Cache für inkrementelle Berechnungen"""
        if symbol not in self.last_prices:
            self.last_prices[symbol] = []
        
        # Füge neuen Preis hinzu (begrenzt auf 100 Einträge für Speichereffizienz)
        self.last_prices[symbol].append(new_price)
        if len(self.last_prices[symbol]) > 100:
            self.last_prices[symbol] = self.last_prices[symbol][-100:]
    
    def get_cached_prices(self, symbol, required_length):
        """Gibt gecachte Preise zurück, ergänzt falls nötig"""
        if symbol in self.last_prices and len(self.last_prices[symbol]) >= required_length:
            return self.last_prices[symbol][-required_length:]
        return None
    
    def clear_expired_cache(self):
        """Räumt abgelaufenen Cache auf"""
        self.cache.clear_expired()

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
# TRADING BOT - ERWEITERTE VERSION MIT OPTIMIERTER ANALYSE
# =============================================================================

class KuCoinTradingBot:
    def __init__(self, api_key, api_secret, api_passphrase, sandbox=False):
        self.api = KuCoinAPI(api_key, api_secret, api_passphrase, sandbox)
        self.tax_logger = TaxLogger()
        self.trade_storage = ActiveTradesStorage()  # NEU: Persistent Storage
        self.tech_analysis = TechnicalAnalysis()  # OPTIMIERT: Mit Caching
        self.data_cache = DataCache(ttl_minutes=10)
        
        # Lade aktive Trades aus persistentem Storage
        self.active_trades = self.trade_storage.load_active_trades()
        self.trade_history = []
        self.current_recommendations = {}
        
        # Trading Einstellungen
        self.auto_trading = False
        self.stop_loss_percent = 2.0
        self.take_profit_percent = 5.0  # NEU: Take-Profit
        self.trailing_stop_percent = 3.0  # NEU: Trailing Stop
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
        self.last_balance_check = None  # NEU: Letzte Kontostandsprüfung
        
        # Performance Tracking
        self.gui_reference = None
        
        # Lade Trade-History
        print("🔄 Lade Trade-History beim Start...")
        self.load_trade_history()
        
        # Starte automatische Trade-Überwachung
        self.start_auto_monitoring()
        
        print(f"✅ KuCoin Trading Bot initialisiert - Sandbox: {sandbox}")
        print(f"📊 Trade-History: {len(self.trade_history)} Trades geladen")
        print(f"📈 Aktive Trades: {len(self.active_trades)} Trades geladen")
        print(f"🔧 Technische Analyse: {'TA-Lib + NumPy' if TA_LIB_AVAILABLE else 'NumPy'} mit Caching")
        print(f"🎯 Verkaufsstrategie: Take-Profit {self.take_profit_percent}%, Trailing Stop {self.trailing_stop_percent}%")
        
    def start_auto_monitoring(self):
        """Startet automatische Trade-Überwachung mit Trailing Stop und technischen Signalen"""
        def monitoring_loop():
            while True:
                try:
                    self.check_stop_loss()
                    self.check_take_profit()
                    self.check_trailing_stop()
                    self.check_technical_sell_signals()
                    time.sleep(30)  # Alle 30 Sekunden prüfen
                except Exception as e:
                    print(f"❌ Fehler in Monitoring-Loop: {e}")
                    time.sleep(60)
        
        threading.Thread(target=monitoring_loop, daemon=True).start()
        print("✅ Automatische Trade-Überwachung gestartet")
        
    def load_trade_history(self):
        """Lädt Trade-History aus dem Tax-Logger"""
        try:
            recent_trades = self.tax_logger.get_recent_trades(1000)
            
            print(f"🔍 Lade Trade-History: {len(recent_trades)} Trades gefunden")
            if recent_trades:
                for i, trade in enumerate(recent_trades[:3]):
                    print(f"  Trade {i+1}: {trade.get('symbol')} {trade.get('side')} {trade.get('timestamp')}")
            
            self.trade_history = recent_trades
            print(f"✅ {len(self.trade_history)} Trades aus History geladen")
            
        except Exception as e:
            print(f"❌ Fehler beim Laden der Trade-History: {e}")
            self.trade_history = []
    
    def save_active_trades(self):
        """Speichert aktive Trades persistent - NEUE METHODE"""
        return self.trade_storage.save_active_trades(self.active_trades)
    
    def get_active_trades_with_current_value(self):
        """Gibt aktive Trades mit aktuellem Wert zurück - NEUE METHODE"""
        active_trades_with_value = {}
        
        if not self.active_trades:
            return active_trades_with_value
        
        # Hole aktuelle Preise für alle aktiven Trades
        symbols = list(self.active_trades.keys())
        current_prices = self.api.get_multiple_tickers(symbols)
        
        for symbol, trade in self.active_trades.items():
            current_price = current_prices.get(symbol)
            if current_price:
                # Berechne aktuellen Wert
                current_value = trade['amount'] * current_price
                
                # Berechne Gewinn/Verlust
                entry_price = trade['buy_price']
                profit_loss = (current_price - entry_price) * trade['amount']
                profit_loss_percent = ((current_price - entry_price) / entry_price) * 100
                
                # Füge Werte zum Trade hinzu
                trade_with_value = trade.copy()
                trade_with_value['current_price'] = current_price
                trade_with_value['current_value'] = current_value
                trade_with_value['profit_loss'] = profit_loss
                trade_with_value['profit_loss_percent'] = profit_loss_percent
                
                active_trades_with_value[symbol] = trade_with_value
        
        return active_trades_with_value
    
    def set_gui_reference(self, gui):
        self.gui_reference = gui
        
    def update_bot_activity(self, message):
        """Aktualisiert Bot-Aktivität"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        
        if any(keyword in message for keyword in ['✅', '❌', '⚡', '📊', '🎯']):
            print(log_entry)
        
        if self.gui_reference:
            self.gui_reference.update_bot_activity(log_entry)
    
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
        """Berechnet RSI - Kompatibilitätsfunktion"""
        return self.tech_analysis.calculate_rsi(prices, period)
    
    def calculate_moving_average(self, prices, period):
        """Berechnet gleitenden Durchschnitt - Kompatibilitätsfunktion"""
        if len(prices) < period:
            return sum(prices) / len(prices) if prices else 0
        return sum(prices[-period:]) / period
    
    def get_historical_data(self, symbol, interval='15min', limit=100):
        """Holt echte historische Daten von KuCoin mit Caching"""
        cache_key = f"klines_{symbol}_{interval}_{limit}"
        cached_data = self.data_cache.get(cache_key)
        
        if cached_data is not None:
            return cached_data
            
        try:
            klines_data = self.api.get_klines(symbol, interval)
            
            if klines_data:
                prices = [kline['close'] for kline in klines_data]
                result = prices[-limit:]
                
                self.data_cache.set(cache_key, result)
                return result
            else:
                return None
                
        except Exception as e:
            print(f"❌ Fehler beim Abrufen historischer Daten für {symbol}: {e}")
            return None
    
    def analyze_crypto(self, symbol):
        """Analysiert Kryptowährung mit OPTIMIERTEN technischen Indikatoren und Caching"""
        try:
            current_price = self.get_current_price(symbol)
            if not current_price:
                return None
                
            # OPTIMIERT: Verwende Analyse-Cache
            cache_key = f"analysis_{symbol}_{self.backtest_interval}"
            cached_analysis = self.tech_analysis.cache.get_analysis(symbol, self.backtest_interval)
            
            if cached_analysis:
                print(f"✅ Verwende gecachte Analyse für {symbol}")
                return cached_analysis
                
            historical_data = self.get_historical_data(symbol, self.backtest_interval, 100)
            if not historical_data:
                return None
                
            # OPTIMIERT: Verwende gecachte technische Indikatoren
            rsi = self.tech_analysis.calculate_rsi(historical_data, symbol=symbol)
            ma_data = self.tech_analysis.calculate_moving_averages(historical_data, [10, 20], symbol=symbol)
            macd_data = self.tech_analysis.calculate_macd(historical_data, symbol=symbol)
            bollinger_data = self.tech_analysis.calculate_bollinger_bands(historical_data, symbol=symbol)
            
            ma_short = ma_data.get('ma_10', current_price)
            ma_long = ma_data.get('ma_20', current_price)
            
            # Signal-Berechnung
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
                
            # MACD Signal
            if macd_data['macd'] > macd_data['signal']:
                signals.append("MACD Bullish")
                confidence += 10
            else:
                signals.append("MACD Bearish")
                confidence -= 10
                
            # Bollinger Bands Signal
            if current_price < bollinger_data['lower']:
                signals.append("Untere BB - Überverkauft")
                confidence += 10
            elif current_price > bollinger_data['upper']:
                signals.append("Obere BB - Überkauft")
                confidence -= 10
                
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
                
            analysis_result = {
                'symbol': symbol,
                'current_price': current_price,
                'current_signal': current_signal,
                'confidence': max(0, min(100, confidence)),
                'signals': signals,
                'rsi': rsi,
                'ma_short': ma_short,
                'ma_long': ma_long,
                'macd': macd_data,
                'bollinger_bands': bollinger_data,
                'total_return': ((current_price - historical_data[0]) / historical_data[0]) * 100 if historical_data else 0
            }
            
            # OPTIMIERT: Speichere Analyse im Cache
            self.tech_analysis.cache.set_analysis(symbol, self.backtest_interval, analysis_result)
            
            return analysis_result
            
        except Exception as e:
            print(f"❌ Fehler bei Analyse von {symbol}: {e}")
            return None
    
    def analyze_all_cryptos_parallel(self):
        """Analysiert alle Kryptos parallel mit optimiertem Caching"""
        self.update_bot_activity("🔄 Starte parallele Analyse aller Kryptowährungen...")
        
        results = {}
        with ThreadPoolExecutor(max_workers=min(len(self.trading_pairs), 8)) as executor:
            futures = {executor.submit(self.analyze_crypto, symbol): symbol for symbol in self.trading_pairs}
            
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()
                    if result:
                        results[symbol] = result
                except Exception as e:
                    print(f"❌ Fehler bei Analyse von {symbol}: {e}")
        
        self.current_recommendations = results
        self.update_bot_activity(f"✅ Parallele Analyse abgeschlossen: {len(results)}/{len(self.trading_pairs)} Kryptos analysiert")
        return results
    
    def quick_signal_check(self):
        """Schnelle Signalprüfung für konfigurierte Kryptos"""
        try:
            self.update_bot_activity("⚡ Starte schnelle Signalprüfung...")
            
            results = self.analyze_all_cryptos_parallel()
                    
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
            
            results = self.analyze_all_cryptos_parallel()
                    
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
        # Prüfe ob letzte Abfrage weniger als 2 Minuten her ist
        if self.last_balance_check and (datetime.now() - self.last_balance_check).total_seconds() < 120:
            return self.balance_cache
            
        try:
            balances = self.api.get_account_balances_detailed()
            
            if not balances:
                return None
            
            assets = []
            total_value = 0
            
            # Hole alle Preise parallel
            symbols_to_fetch = []
            for balance in balances:
                currency = balance['currency']
                if currency != 'USDT':
                    symbols_to_fetch.append(f"{currency}-USDT")
            
            current_prices = self.api.get_multiple_tickers(symbols_to_fetch)
            
            for balance in balances:
                currency = balance['currency']
                if currency == 'USDT':
                    price = 1.0
                    value_usd = balance['available']
                else:
                    symbol = f"{currency}-USDT"
                    price = current_prices.get(symbol, 0)
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
            
            balance_summary = {
                'total_portfolio_value': total_value,
                'assets': sorted(assets, key=lambda x: x['value_usd'], reverse=True),
                'last_updated': datetime.now()
            }
            
            # Cache die Balance-Daten
            self.balance_cache = balance_summary
            self.last_balance_check = datetime.now()
            
            return balance_summary
            
        except Exception as e:
            print(f"❌ Fehler bei Balance Summary: {e}")
            return None
    
    def get_current_price(self, symbol):
        """Holt aktuellen Preis mit Caching"""
        cache_key = f"price_{symbol}"
        cached_price = self.data_cache.get(cache_key)
        
        if cached_price is not None:
            return cached_price
            
        price = self.api.get_ticker(symbol)
        if price:
            self.data_cache.set(cache_key, price)
            # OPTIMIERT: Aktualisiere Preis-Cache für technische Analyse
            self.tech_analysis.update_price_cache(symbol, price)
        return price
    
    def get_current_prices(self):
        """Holt aktuelle Preise aller Trading-Pairs parallel"""
        prices = self.api.get_multiple_tickers(self.trading_pairs)
        for symbol, price in prices.items():
            self.data_cache.set(f"price_{symbol}", price)
            # OPTIMIERT: Aktualisiere Preis-Cache für technische Analyse
            self.tech_analysis.update_price_cache(symbol, price)
        return prices
    
    def update_caches(self):
        """Aktualisiert alle Caches mit echten Daten"""
        try:
            self.get_current_prices()
            self.balance_cache = None  # Zurücksetzen um erneute Abfrage zu erzwingen
            self.balance_cache = self.get_balance_summary()
            
            # OPTIMIERT: Räume abgelaufenen Cache auf
            self.tech_analysis.clear_expired_cache()
            self.data_cache.clear_expired()
            
        except Exception as e:
            print(f"❌ Fehler beim Cache-Update: {e}")
    
    def calculate_portfolio_value(self):
        """Berechnet Gesamtwert des Portfolios"""
        balance = self.get_balance_summary()
        return balance['total_portfolio_value'] if balance else 0.0
    
    def set_trading_settings(self, stop_loss=None, trade_size=None, rsi_oversold=None, rsi_overbought=None, take_profit=None, trailing_stop=None):
        """Aktualisiert Trading-Einstellungen"""
        if stop_loss is not None:
            self.stop_loss_percent = stop_loss
        if trade_size is not None:
            self.trade_size_percent = trade_size
        if rsi_oversold is not None:
            self.rsi_oversold = rsi_oversold
        if rsi_overbought is not None:
            self.rsi_overbought = rsi_overbought
        if take_profit is not None:
            self.take_profit_percent = take_profit
        if trailing_stop is not None:
            self.trailing_stop_percent = trailing_stop
    
    def set_interval(self, interval):
        """Setzt Analyse-Interval"""
        self.backtest_interval = interval
    
    def check_stop_loss(self):
        """Prüft Stop-Loss für aktive Trades"""
        if not self.active_trades:
            return
            
        for symbol, trade in list(self.active_trades.items()):
            current_price = self.get_current_price(symbol)
            if not current_price:
                continue
                
            buy_price = trade['buy_price']
            stop_loss_price = buy_price * (1 - self.stop_loss_percent / 100)
            
            if current_price <= stop_loss_price:
                self.close_trade(symbol, f"Stop-Loss erreicht ({self.stop_loss_percent}%)")
    
    def check_take_profit(self):
        """Prüft Take-Profit für aktive Trades"""
        if not self.active_trades:
            return
            
        for symbol, trade in list(self.active_trades.items()):
            current_price = self.get_current_price(symbol)
            if not current_price:
                continue
                
            buy_price = trade['buy_price']
            profit_percent = ((current_price - trade['buy_price']) / trade['buy_price']) * 100
            
            if profit_percent >= self.take_profit_percent:
                self.close_trade(symbol, f"Take-Profit erreicht ({profit_percent:.1f}%)")
    
    def check_trailing_stop(self):
        """Prüft Trailing-Stop für aktive Trades"""
        if not self.active_trades:
            return
            
        for symbol, trade in list(self.active_trades.items()):
            current_price = self.get_current_price(symbol)
            if not current_price:
                continue
            
            # Initialisiere highest_price falls nicht vorhanden
            if 'highest_price' not in trade:
                trade['highest_price'] = trade['buy_price']
            
            # Update highest_price
            if current_price > trade['highest_price']:
                trade['highest_price'] = current_price
                self.update_bot_activity(f"📈 Neues Hoch für {symbol}: ${current_price:.4f}")
            
            # Berechne Trailing Stop Preis
            trailing_stop_price = trade['highest_price'] * (1 - self.trailing_stop_percent / 100)
            
            if current_price <= trailing_stop_price:
                profit_percent = ((current_price - trade['buy_price']) / trade['buy_price']) * 100
                self.close_trade(symbol, f"Trailing Stop erreicht ({profit_percent:+.1f}% Gewinn)")
    
    def check_technical_sell_signals(self):
        """Prüft technische Verkaufssignale für aktive Trades"""
        if not self.active_trades:
            return
            
        for symbol, trade in list(self.active_trades.items()):
            try:
                analysis = self.analyze_crypto(symbol)
                if not analysis:
                    continue
                
                current_signal = analysis.get('current_signal', 'HOLD')
                rsi = analysis.get('rsi', 50)
                ma_short = analysis.get('ma_short', 0)
                ma_long = analysis.get('ma_long', 0)
                macd_data = analysis.get('macd', {})
                macd_line = macd_data.get('macd', 0)
                macd_signal = macd_data.get('signal', 0)
                
                # Verkaufsbedingungen
                sell_conditions = [
                    current_signal in ["STRONG_SELL", "SELL"],
                    rsi > 75,  # Stark überkauft
                    ma_short < ma_long,  # MA Death Cross
                    macd_line < macd_signal,  # MACD bearish
                    (ma_short < ma_long) and (rsi > 70),  # Kombiniertes Signal
                ]
                
                # Zähle erfüllte Bedingungen
                fulfilled_conditions = sum(sell_conditions)
                
                if fulfilled_conditions >= 2:  # Mindestens 2 Bedingungen erfüllt
                    current_price = self.get_current_price(symbol)
                    if current_price:
                        profit_percent = ((current_price - trade['buy_price']) / trade['buy_price']) * 100
                        reason = f"Technisches Verkaufssignal ({fulfilled_conditions}/5 Bedingungen, {profit_percent:+.1f}%)"
                        self.close_trade(symbol, reason)
                        
            except Exception as e:
                print(f"❌ Fehler bei technischer Signalprüfung für {symbol}: {e}")
    
    def close_all_trades(self):
        """Schließt alle aktiven Trades manuell"""
        if not self.active_trades:
            self.update_bot_activity("ℹ️ Keine aktiven Trades zum Schließen")
            return False
            
        closed_trades = 0
        for symbol in list(self.active_trades.keys()):
            success = self.close_trade(symbol, "Manuell geschlossen")
            if success:
                closed_trades += 1
                
        self.update_bot_activity(f"🔴 {closed_trades} aktive Trades manuell geschlossen")
        return closed_trades > 0
    
    def close_trade(self, symbol, reason):
        """Schließt einen aktiven Trade mit echter API"""
        if symbol not in self.active_trades:
            return False
            
        trade = self.active_trades.pop(symbol)
        current_price = self.get_current_price(symbol)
        if not current_price:
            return False
                
        profit_loss = (current_price - trade['buy_price']) * trade['amount']
        profit_loss_percent = ((current_price - trade['buy_price']) / trade['buy_price']) * 100
            
        if self.auto_trading:
            order_result = self.api.place_order(
                symbol=symbol,
                side='sell',
                order_type='market',
                size=trade['amount']
            )
                
            if order_result:
                order_id = order_result.get('orderId', 'unknown')
                self.update_bot_activity(f"🔴 Verkauf: {symbol} - {profit_loss_percent:+.1f}% - {reason}")
            else:
                order_id = 'failed'
                self.update_bot_activity(f"❌ Verkauf fehlgeschlagen: {symbol}")
                return False
        else:
            order_id = 'simulated'
            self.update_bot_activity(f"🔴 Simulierter Verkauf: {symbol} - {profit_loss_percent:+.1f}% - {reason}")
        
        # Logge den Trade
        trade_data = {
            'symbol': symbol,
            'side': 'SELL',
            'amount': trade['amount'],
            'price': current_price,
            'profit_loss': profit_loss,
            'profit_loss_percent': profit_loss_percent,
            'reason': reason,
            'order_id': order_id,
            'portfolio_value': self.calculate_portfolio_value()
        }
        self.tax_logger.log_trade(trade_data)
        
        # Speichere aktive Trades nach Schließen
        self.save_active_trades()
        
        # Aktualisiere Trade-History
        self.load_trade_history()
        return True
    
    def execute_trade(self, symbol, signal):
        """Führt einen Trade mit echter API aus mit verbesserter Validierung"""
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
            
        # Berechne Trade-Wert (mindestens $10 um Gebühren zu decken)
        trade_value = portfolio_value * (self.trade_size_percent / 100)
        min_trade_value = 10.0  # Mindesthandelswert in USDT
        max_trade_value = portfolio_value * 0.2  # Maximal 20% des Portfolios pro Trade
        
        if trade_value < min_trade_value:
            print(f"⚠️ Trade-Wert zu klein (${trade_value:.2f}), erhöhe auf Minimum ${min_trade_value}")
            trade_value = min_trade_value
        elif trade_value > max_trade_value:
            print(f"⚠️ Trade-Wert zu groß (${trade_value:.2f}), reduziere auf Maximum ${max_trade_value:.2f}")
            trade_value = max_trade_value
        
        # Berechne Menge basierend auf Preis
        trade_amount = trade_value / current_price
        
        print(f"🔍 Validiere Order für {symbol}: Menge={trade_amount:.6f}, Wert=${trade_value:.2f}")
        
        # Verwende KuCoin's eigene Validierung
        is_valid, validation_msg = self.api.validate_order_size(symbol, trade_amount, current_price)
        
        if not is_valid:
            print(f"❌ Order-Validierung fehlgeschlagen: {validation_msg}")
            
            # Versuche korrigierte Größe
            corrected_amount = self.api.calculate_valid_size(symbol, trade_amount)
            if corrected_amount != trade_amount:
                print(f"🔄 Verwende korrigierte Menge: {corrected_amount:.6f}")
                trade_amount = corrected_amount
            else:
                print(f"❌ Keine gültige Menge für {symbol} gefunden")
                return False
        
        # Finale Validierung
        is_valid_final, final_msg = self.api.validate_order_size(symbol, trade_amount, current_price)
        if not is_valid_final:
            print(f"❌ Finale Validierung fehlgeschlagen: {final_msg}")
            return False
        
        final_trade_value = trade_amount * current_price
        print(f"🔄 Versuche Trade: {symbol} {signal} - Menge: {trade_amount:.6f} - Preis: ${current_price:.6f} - Wert: ${final_trade_value:.2f}")
            
        if "BUY" in signal:
            try:
                order_result = self.api.place_order(
                    symbol=symbol,
                    side='buy',
                    order_type='market',
                    size=trade_amount
                )
                    
                if order_result:
                    print(f"✅ API Order erfolgreich: {order_result}")
                    
                    self.active_trades[symbol] = {
                        'buy_price': current_price,
                        'amount': trade_amount,
                        'timestamp': datetime.now(),
                        'highest_price': current_price,  # Für Trailing Stop
                        'order_id': order_result.get('orderId', 'unknown')
                    }
                        
                    trade_data = {
                        'symbol': symbol,
                        'side': 'BUY',
                        'amount': trade_amount,
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
                    
                    # Speichere aktive Trades nach Kauf
                    self.save_active_trades()
                    
                    # Trade-History SOFORT aktualisieren
                    self.load_trade_history()
                    
                    self.last_trade_time = datetime.now()
                    self.update_bot_activity(f"🟢 Trade eröffnet: {symbol} - {trade_amount:.6f} @ ${current_price:.2f}")
                    return True
                else:
                    print(f"❌ API Order fehlgeschlagen für {symbol}")
                    # Debug-Informationen
                    self.debug_order_issues(symbol, trade_amount, current_price)
                    return False
            except Exception as e:
                print(f"❌ Exception während Trade-Ausführung: {e}")
                return False
        return False

    def debug_order_issues(self, symbol, amount, price):
        """Debug-Methode zur Untersuchung von Order-Problemen"""
        print(f"🔍 Debug Order-Problem für {symbol}:")
        print(f"   - Menge: {amount}")
        print(f"   - Preis: ${price}")
        print(f"   - Wert: ${amount * price:.2f}")
        
        # Prüfe Symbol-Informationen
        if symbol in self.api.symbols_info:
            symbol_info = self.api.symbols_info[symbol]
            print(f"   - Symbol Info: {symbol_info}")
            
            # Prüfe spezifische Anforderungen
            min_size = symbol_info.get('baseMinSize', 0)
            max_size = symbol_info.get('baseMaxSize', float('inf'))
            base_increment = symbol_info.get('baseIncrement', 0)
            min_funds = symbol_info.get('minFunds', 0)
            
            print(f"   - Min Size: {min_size}")
            print(f"   - Max Size: {max_size}")
            print(f"   - Base Increment: {base_increment}")
            print(f"   - Min Funds: {min_funds}")
            
            # Prüfe ob Menge den Anforderungen entspricht
            if amount < min_size:
                print(f"   ❌ Menge zu klein! Minimum: {min_size}")
            if amount > max_size:
                print(f"   ❌ Menge zu groß! Maximum: {max_size}")
            if base_increment > 0:
                steps = amount / base_increment
                if not steps.is_integer():
                    valid_steps = round(steps)
                    valid_amount = round(valid_steps * base_increment, 8)
                    print(f"   ❌ Ungültige Schrittgröße! Nächstes Valid: {valid_amount}")
            
            order_value = amount * price
            if order_value < min_funds:
                print(f"   ❌ Order-Wert zu klein! Minimum: {min_funds}")
        else:
            print(f"   ❌ Keine Symbol-Informationen verfügbar")
    
    def get_trade_history_for_gui(self, limit=50):
        """Gibt Trade-History für die GUI zurück mit korrekten Berechnungen"""
        try:
            print(f"🔍 Lade Trade-History für GUI... ({len(self.trade_history)} Trades verfügbar)")
            
            if not self.trade_history:
                print("ℹ️ Keine Trade-History verfügbar")
                return []
            
            gui_trades = []
            
            for trade in self.trade_history[-limit:]:
                try:
                    # Stelle sicher, dass profit_loss korrekt berechnet wird
                    profit_loss = trade.get('profit_loss', 0)
                    if profit_loss is None:
                        profit_loss = 0
                    
                    # Berechne profit_loss_percent falls nicht vorhanden
                    profit_loss_percent = trade.get('profit_loss_percent', 0)
                    if profit_loss_percent is None:
                        price = trade.get('price', 0)
                        amount = trade.get('amount', 0)
                        if price and amount and price > 0:
                            # Vereinfachte Berechnung für Prozent
                            profit_loss_percent = (profit_loss / (price * amount)) * 100
                        else:
                            profit_loss_percent = 0
                    
                    gui_trade = {
                        'timestamp': trade.get('timestamp', 'Unbekannt'),
                        'symbol': trade.get('symbol', 'Unknown'),
                        'side': trade.get('side', 'UNKNOWN'),
                        'price': float(trade.get('price', 0)),
                        'amount': float(trade.get('amount', 0)),
                        'profit_loss': float(profit_loss),
                        'profit_loss_percent': float(profit_loss_percent),
                        'reason': trade.get('reason', '')
                    }
                    gui_trades.append(gui_trade)
                    
                except Exception as e:
                    print(f"⚠️ Fehler beim Konvertieren des Trades: {e}")
                    continue
            
            print(f"✅ {len(gui_trades)} Trades für GUI vorbereitet")
            return gui_trades
            
        except Exception as e:
            print(f"❌ Fehler in get_trade_history_for_gui: {e}")
            return []

# =============================================================================
# MODERN FULLHD GUI - UNVERÄNDERT (wie in der ursprünglichen Datei)
# =============================================================================

# [Hier folgt der komplette GUI-Code aus der ursprünglichen Datei...]
# Aus Platzgründen hier nur die wichtigsten Teile:


class ModernTradingGUI:
    def __init__(self, bot):
        self.bot = bot
        self.bot.set_gui_reference(self)
        
        self.root = tk.Tk()
        self.setup_gui()
        self.start_auto_updates()
        
        
    def setup_gui(self):
        """Erstellt das moderne FullHD GUI"""
        # Hauptfenster für FullHD
        self.root.title("🚀 KuCoin Trading Bot - FullHD Optimiert")
        self.root.geometry("1920x1080")
        self.root.configure(bg='#1e1e1e')  # Dunkler Hintergrund
        
        # Styling für moderne Optik
        self.setup_styles()
        
        # Haupt-Layout
        self.setup_main_layout()
        
        # Starte initiale Updates
        self.root.after(1000, self.initial_updates)
        
    def setup_styles(self):
        """Konfiguriert moderne Styles mit besseren Kontrasten"""
        style = ttk.Style()
        
        # Verwende 'clam' Theme für bessere Anpassbarkeit
        style.theme_use('clam')
        
        # Configure Styles
        style.configure('Modern.TFrame', background='#2d2d2d')
        style.configure('Modern.TLabelframe', background='#2d2d2d', foreground='white')
        style.configure('Modern.TLabelframe.Label', background='#2d2d2d', foreground='white')
        style.configure('Modern.TLabel', background='#2d2d2d', foreground='white')
        style.configure('Modern.TButton', background='#404040', foreground='white')
        style.configure('Success.TButton', background='#28a745', foreground='white')
        style.configure('Danger.TButton', background='#dc3545', foreground='white')
        style.configure('Warning.TButton', background='#ffc107', foreground='black')
        
        # Treeview Styles mit besserem Kontrast
        style.configure('Modern.Treeview', 
                       background='#1e1e1e',
                       foreground='#ffffff',  # WEIẞER Text für besseren Kontrast
                       fieldbackground='#1e1e1e',
                       borderwidth=1,
                       relief='solid')
        style.configure('Modern.Treeview.Heading', 
                       background='#404040',
                       foreground='#ffffff',  # WEIẞER Text in Headers
                       relief='raised',
                       borderwidth=1)
        
        # Treeview Zeilen mit alternierenden Farben
        style.map('Modern.Treeview', 
                  background=[('selected', '#4a6984')],  # Blaue Auswahl
                  foreground=[('selected', '#ffffff')])   # Weißer Text bei Auswahl
        
        # Scrollbar Style
        style.configure('Modern.Vertical.TScrollbar', 
                       background='#404040',
                       darkcolor='#404040',
                       lightcolor='#404040',
                       troughcolor='#2d2d2d',
                       bordercolor='#404040',
                       arrowcolor='#ffffff')
        
        # Entry und Combobox Styles
        style.configure('Modern.TEntry', 
                       fieldbackground='#2d2d2d',
                       foreground='#ffffff',
                       insertcolor='#ffffff')  # Cursor Farbe
        
        style.configure('Modern.TCombobox',
                       fieldbackground='#2d2d2d',
                       background='#2d2d2d',
                       foreground='#ffffff',
                       arrowcolor='#ffffff')
        
        style.map('Modern.TCombobox',
                 fieldbackground=[('readonly', '#2d2d2d')],
                 selectbackground=[('readonly', '#4a6984')],
                 selectforeground=[('readonly', '#ffffff')])
        
    def setup_main_layout(self):
        """Setup des Hauptlayouts für FullHD"""
        # Header mit Status und Kontrollen
        self.setup_header()
        
        # Hauptbereich mit Tabs
        self.setup_main_tabs()
        
        # Status Bar
        self.setup_status_bar()
        
    def setup_header(self):
        """Setup des Headers mit verbesserten Farben"""
        header_frame = ttk.Frame(self.root, style='Modern.TFrame', height=80)
        header_frame.pack(fill=tk.X, padx=10, pady=5)
        header_frame.pack_propagate(False)
        
        # Linke Seite: Titel und Status
        left_header = ttk.Frame(header_frame, style='Modern.TFrame')
        left_header.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        
        title_label = ttk.Label(left_header, 
                              text="🚀 KuCoin Trading Bot", 
                              font=('Arial', 20, 'bold'),
                              style='Modern.TLabel',
                              foreground='#ffffff')  # Explizit weiße Schrift
        title_label.pack(anchor=tk.W)
        
        self.status_label = ttk.Label(left_header, 
                                    text="🔴 Nicht verbunden", 
                                    font=('Arial', 12),
                                    style='Modern.TLabel',
                                    foreground='#ff6b6b')  # Rote Schrift für "Nicht verbunden"
        self.status_label.pack(anchor=tk.W)
        
        # Rechte Seite: API Controls
        right_header = ttk.Frame(header_frame, style='Modern.TFrame')
        right_header.pack(side=tk.RIGHT, fill=tk.Y)
        
        # API Connection Frame
        api_frame = ttk.LabelFrame(right_header, text="API Verbindung", style='Modern.TLabelframe')
        api_frame.pack(side=tk.LEFT, padx=5)
        
        self.connect_button = ttk.Button(api_frame, 
                                       text="Verbinden", 
                                       command=self.toggle_connection,
                                       style='Modern.TButton')
        self.connect_button.pack(padx=10, pady=5)
        
        # Auto Trading Frame
        trading_frame = ttk.LabelFrame(right_header, text="Trading", style='Modern.TLabelframe')
        trading_frame.pack(side=tk.LEFT, padx=5)
        
        self.auto_trading_var = tk.BooleanVar(value=self.bot.auto_trading)
        auto_trading_btn = ttk.Checkbutton(trading_frame, 
                                         text="Auto Trading", 
                                         variable=self.auto_trading_var,
                                         command=self.toggle_auto_trading,
                                         style='Modern.TLabel')
        auto_trading_btn.pack(padx=10, pady=5)
        
    def setup_main_tabs(self):
        """Setup der Haupt-Tabs"""
        # Notebook für Tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Tabs erstellen
        self.setup_dashboard_tab()
        self.setup_trading_tab()
        self.setup_analysis_tab()
        self.setup_portfolio_tab()
        self.setup_history_tab()
        self.setup_tax_tab()
        
        # Tab Change Event
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        
    def setup_dashboard_tab(self):
        """Dashboard Tab mit Übersicht - VERBESSERTE FARBEN"""
        dashboard_frame = ttk.Frame(self.notebook, style='Modern.TFrame')
        self.notebook.add(dashboard_frame, text="📊 Dashboard")
        
        # Obere Reihe: Balance und Schnellaktionen
        top_frame = ttk.Frame(dashboard_frame, style='Modern.TFrame')
        top_frame.pack(fill=tk.X, pady=10)
        
        # Balance Panel
        balance_frame = ttk.LabelFrame(top_frame, text="Portfolio Übersicht", style='Modern.TLabelframe', width=400)
        balance_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        balance_frame.pack_propagate(False)
        
        self.balance_info_var = tk.StringVar(value="Lade Kontostand...")
        balance_label = ttk.Label(balance_frame, textvariable=self.balance_info_var, style='Modern.TLabel')
        balance_label.pack(pady=10)
        
        ttk.Button(balance_frame, text="Aktualisieren", 
                  command=self.update_balance_display, 
                  style='Modern.TButton').pack(pady=5)
        
        # Quick Actions
        actions_frame = ttk.LabelFrame(top_frame, text="Schnellaktionen", style='Modern.TLabelframe')
        actions_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        action_buttons = [
            ("🔍 Schnellanalyse", self.quick_signal_check, 'Modern.TButton'),
            ("📊 Vollständiger Backtest", self.start_backtest, 'Modern.TButton'),
            ("🔄 Cache Aktualisieren", self.force_cache_update, 'Modern.TButton'),
            ("🤖 Auto Trade Starten", self.toggle_auto_trade, 'Warning.TButton'),
            ("🔴 Alle Trades schließen", self.close_all_trades, 'Danger.TButton')
        ]
        
        for text, command, style_name in action_buttons:
            ttk.Button(actions_frame, text=text, command=command, style=style_name).pack(fill=tk.X, padx=20, pady=5)
        
        # Untere Reihe: Aktive Trades und Empfehlungen
        bottom_frame = ttk.Frame(dashboard_frame, style='Modern.TFrame')
        bottom_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Aktive Trades - ERWEITERT mit aktuellem Wert
        trades_frame = ttk.LabelFrame(bottom_frame, text="Aktive Trades", style='Modern.TLabelframe')
        trades_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Frame für Trades Treeview und Schließen-Button
        trades_content_frame = ttk.Frame(trades_frame, style='Modern.TFrame')
        trades_content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # ERWEITERTE SPALTEN mit aktuellem Wert
        columns = ('Symbol', 'Kaufpreis', 'Aktuell', 'Menge', 'Wert €', 'P/L %', 'P/L €', 'Seit', 'Höchstkurs')
        self.trades_tree = ttk.Treeview(trades_content_frame, columns=columns, show='headings', style='Modern.Treeview')
        
        column_widths = {
            'Symbol': 100, 'Kaufpreis': 100, 'Aktuell': 100, 'Menge': 100,
            'Wert €': 100, 'P/L %': 80, 'P/L €': 100, 'Seit': 80, 'Höchstkurs': 100
        }
        
        for col in columns:
            self.trades_tree.heading(col, text=col)
            self.trades_tree.column(col, width=column_widths[col])
        
        self.trades_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbar für Trades Treeview
        trades_scroll = ttk.Scrollbar(trades_content_frame, orient=tk.VERTICAL, command=self.trades_tree.yview, style='Modern.Vertical.TScrollbar')
        self.trades_tree.configure(yscrollcommand=trades_scroll.set)
        trades_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Button Frame unterhalb der Trades
        trades_button_frame = ttk.Frame(trades_frame, style='Modern.TFrame')
        trades_button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(trades_button_frame, text="Ausgewählten Trade schließen", 
                  command=self.close_selected_trade,
                  style='Danger.TButton').pack(side=tk.LEFT, padx=5)
        
        # Top Empfehlungen
        rec_frame = ttk.LabelFrame(bottom_frame, text="Top Empfehlungen", style='Modern.TLabelframe')
        rec_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        columns = ('Symbol', 'Preis', 'Signal', 'Confidence', 'Aktion')
        self.dashboard_rec_tree = ttk.Treeview(rec_frame, columns=columns, show='headings', style='Modern.Treeview', height=8)
        
        for col in columns:
            self.dashboard_rec_tree.heading(col, text=col)
            self.dashboard_rec_tree.column(col, width=120)
        
        self.dashboard_rec_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrollbar für Empfehlungen Treeview
        rec_scroll = ttk.Scrollbar(rec_frame, orient=tk.VERTICAL, command=self.dashboard_rec_tree.yview, style='Modern.Vertical.TScrollbar')
        self.dashboard_rec_tree.configure(yscrollcommand=rec_scroll.set)
        rec_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Tags für farbige Zeilen
        self.trades_tree.tag_configure('profit', background='#d4edda', foreground='#000000')
        self.trades_tree.tag_configure('loss', background='#f8d7da', foreground='#000000')
        self.dashboard_rec_tree.tag_configure('buy', background='#d4edda', foreground='#000000')
        self.dashboard_rec_tree.tag_configure('sell', background='#f8d7da', foreground='#000000')
        self.dashboard_rec_tree.tag_configure('hold', background='#fff3cd', foreground='#000000')
        
        # Double-Click für schnellen Trade
        self.dashboard_rec_tree.bind('<Double-1>', self.on_dashboard_recommendation_click)
        
    def setup_trading_tab(self):
        """Trading Tab mit Pair-Auswahl und Einstellungen - VERBESSERTE FARBEN"""
        trading_frame = ttk.Frame(self.notebook, style='Modern.TFrame')
        self.notebook.add(trading_frame, text="💎 Trading")
        
        # Linke Seite: Pair Auswahl
        left_frame = ttk.Frame(trading_frame, style='Modern.TFrame')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Verfügbare Pairs
        available_frame = ttk.LabelFrame(left_frame, text="Verfügbare Trading Pairs", style='Modern.TLabelframe')
        available_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Suchleiste
        search_frame = ttk.Frame(available_frame, style='Modern.TFrame')
        search_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(search_frame, text="Suchen:", style='Modern.TLabel').pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=20, style='Modern.TEntry')
        search_entry.pack(side=tk.LEFT, padx=5)
        search_entry.bind('<KeyRelease>', self.filter_available_pairs)
        
        ttk.Button(search_frame, text="Alle laden", 
                  command=self.load_available_pairs,
                  style='Modern.TButton').pack(side=tk.RIGHT)
        
        # Available Pairs Listbox - MIT VERBESSERTEN FARBEN
        listbox_frame = ttk.Frame(available_frame, style='Modern.TFrame')
        listbox_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.available_listbox = tk.Listbox(
            listbox_frame, 
            bg='#1e1e1e',
            fg='#ffffff',  # WEIẞER Text für besseren Kontrast
            selectbackground='#4a6984',
            selectforeground='#ffffff',
            font=('Arial', 10),
            relief='solid',
            borderwidth=1,
            highlightthickness=1,
            highlightcolor='#404040'
        )
        self.available_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        available_scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=self.available_listbox.yview, style='Modern.Vertical.TScrollbar')
        available_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.available_listbox.config(yscrollcommand=available_scrollbar.set)
        
        # Ausgewählte Pairs - MIT VERBESSERTEN FARBEN
        selected_frame = ttk.LabelFrame(left_frame, text="Ausgewählte Trading Pairs", style='Modern.TLabelframe')
        selected_frame.pack(fill=tk.X, pady=5)
        
        self.selected_listbox = tk.Listbox(
            selected_frame, 
            bg='#1e1e1e',
            fg='#ffffff',  # WEIẞER Text für besseren Kontrast
            selectbackground='#4a6984',
            selectforeground='#ffffff',
            font=('Arial', 10),
            relief='solid',
            borderwidth=1,
            height=6,
            highlightthickness=1,
            highlightcolor='#404040'
        )
        self.selected_listbox.pack(fill=tk.X, padx=5, pady=5)
        
        # Action Buttons
        action_frame = ttk.Frame(left_frame, style='Modern.TFrame')
        action_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(action_frame, text="➡️ Auswählen", 
                  command=self.add_selected_pairs,
                  style='Modern.TButton').pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="❌ Entfernen", 
                  command=self.remove_selected_pairs,
                  style='Danger.TButton').pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="💾 Speichern", 
                  command=self.save_trading_pairs,
                  style='Success.TButton').pack(side=tk.RIGHT, padx=2)
        
        # Rechte Seite: Einstellungen
        right_frame = ttk.Frame(trading_frame, style='Modern.TFrame')
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # Trading Einstellungen
        settings_frame = ttk.LabelFrame(right_frame, text="Trading Einstellungen", style='Modern.TLabelframe')
        settings_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Stop-Loss und Trade Size
        risk_frame = ttk.Frame(settings_frame, style='Modern.TFrame')
        risk_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(risk_frame, text="Stop-Loss %:", style='Modern.TLabel', width=15).grid(row=0, column=0, sticky=tk.W)
        self.stop_loss_var = tk.StringVar(value=str(self.bot.stop_loss_percent))
        stop_loss_entry = ttk.Entry(risk_frame, textvariable=self.stop_loss_var, width=10, style='Modern.TEntry')
        stop_loss_entry.grid(row=0, column=1, padx=5)
        
        ttk.Label(risk_frame, text="Trade Größe %:", style='Modern.TLabel', width=15).grid(row=0, column=2, sticky=tk.W, padx=(20,0))
        self.trade_size_var = tk.StringVar(value=str(self.bot.trade_size_percent))
        trade_size_entry = ttk.Entry(risk_frame, textvariable=self.trade_size_var, width=10, style='Modern.TEntry')
        trade_size_entry.grid(row=0, column=3, padx=5)
        
        # Take-Profit und Trailing Stop
        profit_frame = ttk.Frame(settings_frame, style='Modern.TFrame')
        profit_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(profit_frame, text="Take-Profit %:", style='Modern.TLabel', width=15).grid(row=0, column=0, sticky=tk.W)
        self.take_profit_var = tk.StringVar(value=str(self.bot.take_profit_percent))
        take_profit_entry = ttk.Entry(profit_frame, textvariable=self.take_profit_var, width=10, style='Modern.TEntry')
        take_profit_entry.grid(row=0, column=1, padx=5)
        
        ttk.Label(profit_frame, text="Trailing Stop %:", style='Modern.TLabel', width=15).grid(row=0, column=2, sticky=tk.W, padx=(20,0))
        self.trailing_stop_var = tk.StringVar(value=str(self.bot.trailing_stop_percent))
        trailing_stop_entry = ttk.Entry(profit_frame, textvariable=self.trailing_stop_var, width=10, style='Modern.TEntry')
        trailing_stop_entry.grid(row=0, column=3, padx=5)
        
        # RSI Einstellungen
        rsi_frame = ttk.Frame(settings_frame, style='Modern.TFrame')
        rsi_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(rsi_frame, text="RSI Oversold:", style='Modern.TLabel', width=15).grid(row=0, column=0, sticky=tk.W)
        self.rsi_oversold_var = tk.StringVar(value=str(self.bot.rsi_oversold))
        rsi_oversold_entry = ttk.Entry(rsi_frame, textvariable=self.rsi_oversold_var, width=10, style='Modern.TEntry')
        rsi_oversold_entry.grid(row=0, column=1, padx=5)
        
        ttk.Label(rsi_frame, text="RSI Overbought:", style='Modern.TLabel', width=15).grid(row=0, column=2, sticky=tk.W, padx=(20,0))
        self.rsi_overbought_var = tk.StringVar(value=str(self.bot.rsi_overbought))
        rsi_overbought_entry = ttk.Entry(rsi_frame, textvariable=self.rsi_overbought_var, width=10, style='Modern.TEntry')
        rsi_overbought_entry.grid(row=0, column=3, padx=5)
        
        # Intervall
        interval_frame = ttk.Frame(settings_frame, style='Modern.TFrame')
        interval_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(interval_frame, text="Analyse Intervall:", style='Modern.TLabel', width=15).grid(row=0, column=0, sticky=tk.W)
        self.interval_var = tk.StringVar(value=self.bot.backtest_interval)
        interval_combo = ttk.Combobox(interval_frame, textvariable=self.interval_var, 
                                    values=['1min', '5min', '15min', '1hour', '4hour', '1day'], 
                                    width=10, style='Modern.TCombobox')
        interval_combo.grid(row=0, column=1, padx=5)
        
        # Save Button
        ttk.Button(settings_frame, text="Einstellungen Speichern", 
                  command=self.save_settings,
                  style='Success.TButton').pack(pady=10)
        
        # Manueller Trade
        manual_frame = ttk.LabelFrame(right_frame, text="Manueller Trade", style='Modern.TLabelframe')
        manual_frame.pack(fill=tk.X, pady=5)
        
        manual_input_frame = ttk.Frame(manual_frame, style='Modern.TFrame')
        manual_input_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(manual_input_frame, text="Symbol:", style='Modern.TLabel').grid(row=0, column=0, sticky=tk.W)
        self.trade_symbol_entry = ttk.Entry(manual_input_frame, width=15, style='Modern.TEntry')
        self.trade_symbol_entry.grid(row=0, column=1, padx=5)
        
        ttk.Label(manual_input_frame, text="Menge:", style='Modern.TLabel').grid(row=0, column=2, sticky=tk.W, padx=(10,0))
        self.trade_amount_entry = ttk.Entry(manual_input_frame, width=15, style='Modern.TEntry')
        self.trade_amount_entry.grid(row=0, column=3, padx=5)
        
        manual_button_frame = ttk.Frame(manual_frame, style='Modern.TFrame')
        manual_button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(manual_button_frame, text="🟢 BUY", 
                  command=lambda: self.execute_manual_trade("buy"),
                  style='Success.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(manual_button_frame, text="🔴 SELL", 
                  command=lambda: self.execute_manual_trade("sell"),
                  style='Danger.TButton').pack(side=tk.LEFT, padx=5)
        
    def setup_analysis_tab(self):
        """Analysis Tab mit detaillierten Empfehlungen"""
        analysis_frame = ttk.Frame(self.notebook, style='Modern.TFrame')
        self.notebook.add(analysis_frame, text="📈 Analyse")
        
        # Toolbar
        toolbar = ttk.Frame(analysis_frame, style='Modern.TFrame')
        toolbar.pack(fill=tk.X, pady=5)
        
        ttk.Button(toolbar, text="Analyse Starten", 
                  command=self.start_analysis,
                  style='Modern.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Alle Kauf-Signale ausführen", 
                  command=self.execute_all_buy_signals,
                  style='Success.TButton').pack(side=tk.LEFT, padx=5)
        
        # Analysis Treeview
        tree_frame = ttk.Frame(analysis_frame, style='Modern.TFrame')
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        columns = ('Symbol', 'Preis', 'Signal', 'Confidence', 'RSI', 'MA Signal', 'MACD', 'Performance', 'Aktion')
        self.analysis_tree = ttk.Treeview(tree_frame, columns=columns, show='headings', style='Modern.Treeview')
        
        column_widths = {
            'Symbol': 100, 'Preis': 100, 'Signal': 100, 'Confidence': 80, 
            'RSI': 80, 'MA Signal': 100, 'MACD': 100, 'Performance': 100, 'Aktion': 120
        }
        
        for col in columns:
            self.analysis_tree.heading(col, text=col)
            self.analysis_tree.column(col, width=column_widths[col])
        
        # Scrollbar
        tree_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.analysis_tree.yview, style='Modern.Vertical.TScrollbar')
        self.analysis_tree.configure(yscrollcommand=tree_scroll.set)
        
        self.analysis_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Tags für farbige Zeilen
        self.analysis_tree.tag_configure('buy', background='#d4edda', foreground='#000000')
        self.analysis_tree.tag_configure('sell', background='#f8d7da', foreground='#000000')
        self.analysis_tree.tag_configure('hold', background='#fff3cd', foreground='#000000')
        
        # Double-Click Event
        self.analysis_tree.bind('<Double-1>', self.on_analysis_recommendation_click)
        
    def setup_portfolio_tab(self):
        """Portfolio Tab mit detaillierter Aufstellung"""
        portfolio_frame = ttk.Frame(self.notebook, style='Modern.TFrame')
        self.notebook.add(portfolio_frame, text="💰 Portfolio")
        
        # Summary Frame
        summary_frame = ttk.LabelFrame(portfolio_frame, text="Portfolio Zusammenfassung", style='Modern.TLabelframe')
        summary_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.portfolio_summary_var = tk.StringVar(value="Lade Portfolio...")
        summary_label = ttk.Label(summary_frame, textvariable=self.portfolio_summary_var, 
                                 font=('Arial', 12, 'bold'), style='Modern.TLabel')
        summary_label.pack(pady=10)
        
        ttk.Button(summary_frame, text="Portfolio Aktualisieren", 
                  command=self.update_portfolio_display,
                  style='Modern.TButton').pack(pady=5)
        
        # Detailed Assets
        assets_frame = ttk.LabelFrame(portfolio_frame, text="Asset Details", style='Modern.TLabelframe')
        assets_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        columns = ('Asset', 'Menge', 'Verfügbar', 'Preis', 'Wert (USD)', 'Anteil %')
        self.portfolio_tree = ttk.Treeview(assets_frame, columns=columns, show='headings', style='Modern.Treeview')
        
        column_widths = {
            'Asset': 100, 'Menge': 120, 'Verfügbar': 120, 
            'Preis': 100, 'Wert (USD)': 120, 'Anteil %': 80
        }
        
        for col in columns:
            self.portfolio_tree.heading(col, text=col)
            self.portfolio_tree.column(col, width=column_widths[col])
        
        self.portfolio_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrollbar
        portfolio_scroll = ttk.Scrollbar(assets_frame, orient=tk.VERTICAL, command=self.portfolio_tree.yview, style='Modern.Vertical.TScrollbar')
        self.portfolio_tree.configure(yscrollcommand=portfolio_scroll.set)
        portfolio_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
    def setup_history_tab(self):
        """History Tab mit Trade-Verlauf"""
        history_frame = ttk.Frame(self.notebook, style='Modern.TFrame')
        self.notebook.add(history_frame, text="📋 History")
        
        # Toolbar
        toolbar = ttk.Frame(history_frame, style='Modern.TFrame')
        toolbar.pack(fill=tk.X, pady=5)
        
        ttk.Button(toolbar, text="History Aktualisieren", 
                  command=self.force_history_update,
                  style='Modern.TButton').pack(side=tk.LEFT, padx=5)
        
        # History Treeview
        tree_frame = ttk.Frame(history_frame, style='Modern.TFrame')
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        columns = ('Datum', 'Symbol', 'Side', 'Preis', 'Menge', 'P/L %', 'P/L €', 'Grund')
        self.history_tree = ttk.Treeview(tree_frame, columns=columns, show='headings', style='Modern.Treeview')
        
        column_widths = {
            'Datum': 150, 'Symbol': 100, 'Side': 80, 'Preis': 100,
            'Menge': 120, 'P/L %': 80, 'P/L €': 100, 'Grund': 200
        }
        
        for col in columns:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=column_widths[col])
        
        self.history_tree.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar
        history_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.history_tree.yview, style='Modern.Vertical.TScrollbar')
        self.history_tree.configure(yscrollcommand=history_scroll.set)
        history_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Tags für History
        self.history_tree.tag_configure('buy', background='#d4edda', foreground='#000000')
        self.history_tree.tag_configure('profit', background='#e8f5e8', foreground='#000000')
        self.history_tree.tag_configure('loss', background='#f8d7da', foreground='#000000')
        self.history_tree.tag_configure('neutral', background='#fff3cd', foreground='#000000')
        
    def setup_tax_tab(self):
        """Tax Tab mit Finanzamt-Protokollierung"""
        tax_frame = ttk.Frame(self.notebook, style='Modern.TFrame')
        self.notebook.add(tax_frame, text="🏛️ Finanzamt")
        
        # Toolbar
        toolbar = ttk.Frame(tax_frame, style='Modern.TFrame')
        toolbar.pack(fill=tk.X, pady=5)
        
        ttk.Button(toolbar, text="Steuerreport Generieren", 
                  command=self.generate_tax_report,
                  style='Modern.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Logs Exportieren", 
                  command=self.export_logs,
                  style='Modern.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Daten Aktualisieren", 
                  command=self.force_tax_update,
                  style='Modern.TButton').pack(side=tk.LEFT, padx=5)
        
        # Tax Treeview
        tree_frame = ttk.Frame(tax_frame, style='Modern.TFrame')
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        columns = ('Datum', 'Typ', 'Symbol', 'Menge', 'Preis', 'Gesamt', 'Gebühren', 'Gewinn/Verlust', 'Grund')
        self.tax_tree = ttk.Treeview(tree_frame, columns=columns, show='headings', style='Modern.Treeview')
        
        column_widths = {
            'Datum': 150, 'Typ': 80, 'Symbol': 100, 'Menge': 120,
            'Preis': 100, 'Gesamt': 120, 'Gebühren': 100, 'Gewinn/Verlust': 120, 'Grund': 200
        }
        
        for col in columns:
            self.tax_tree.heading(col, text=col)
            self.tax_tree.column(col, width=column_widths[col])
        
        self.tax_tree.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar
        tax_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tax_tree.yview, style='Modern.Vertical.TScrollbar')
        self.tax_tree.configure(yscrollcommand=tax_scroll.set)
        tax_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Tags für Tax
        self.tax_tree.tag_configure('buy', background='#e8f4fd', foreground='#000000')
        self.tax_tree.tag_configure('profit', background='#d4edda', foreground='#000000')
        self.tax_tree.tag_configure('loss', background='#f8d7da', foreground='#000000')
        self.tax_tree.tag_configure('neutral', background='#fff3cd', foreground='#000000')
        
        # Portfolio Info
        info_frame = ttk.Frame(tax_frame, style='Modern.TFrame')
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.portfolio_var = tk.StringVar(value="Portfolio Wert: €0.00")
        ttk.Label(info_frame, textvariable=self.portfolio_var, style='Modern.TLabel').pack(side=tk.LEFT)
        
        self.total_profit_var = tk.StringVar(value="Gesamtgewinn: €0.00")
        ttk.Label(info_frame, textvariable=self.total_profit_var, style='Modern.TLabel').pack(side=tk.LEFT, padx=20)
        
    def setup_status_bar(self):
        """Setup der Status Bar"""
        status_frame = ttk.Frame(self.root, style='Modern.TFrame', height=30)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)
        
        self.status_var = tk.StringVar(value="Bot initialisiert - Bereit")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, style='Modern.TLabel')
        status_label.pack(side=tk.LEFT, padx=10)
        
        self.api_stats_var = tk.StringVar(value="API Requests: 0")
        ttk.Label(status_frame, textvariable=self.api_stats_var, style='Modern.TLabel').pack(side=tk.RIGHT, padx=10)
        
    # === GUI METHODEN ===
    
    def initial_updates(self):
        """Führt initiale Updates durch"""
        self.update_balance_display()
        self.load_available_pairs()
        self.load_current_pairs()
        self.update_status("Bot bereit - Verbindung zu API herstellen")
        
    def update_status(self, message):
        """Aktualisiert Status-Anzeige"""
        self.status_var.set(message)
        print(f"Status: {message}")
    
    def update_bot_activity(self, message):
        """Aktualisiert Bot-Aktivitätslog"""
        # Für FullHD GUI können wir das in die Status-Bar oder einen separaten Log schreiben
        if "✅" in message or "❌" in message:
            self.update_status(message)
    
    def toggle_connection(self):
        """Schaltet die Verbindung zur KuCoin API um"""
        try:
            if hasattr(self.bot.api, 'api_key') and self.bot.api.api_key:
                # Teste Verbindung
                if self.bot.api.test_connection():
                    self.status_label.configure(text="🟢 Verbunden")
                    self.connect_button.configure(text="Trennen")
                    self.update_status("✅ Verbindung erfolgreich")
                else:
                    self.status_label.configure(text="🔴 Verbindungsfehler")
                    self.update_status("❌ Verbindung fehlgeschlagen")
            else:
                self.status_label.configure(text="🔴 Nicht verbunden")
                self.connect_button.configure(text="Verbinden")
                
        except Exception as e:
            self.status_label.configure(text="🔴 Fehler")
            self.update_status(f"❌ Verbindungsfehler: {e}")
    
    def show_api_login(self):
        """Zeigt API Login Dialog"""
        login_window = tk.Toplevel(self.root)
        login_window.title("API Login")
        login_window.geometry("400x300")
        login_window.configure(bg='#2d2d2d')
        
        ttk.Label(login_window, text="API Key:", style='Modern.TLabel').pack(pady=5)
        api_key_entry = ttk.Entry(login_window, width=40, show="*", style='Modern.TEntry')
        api_key_entry.pack(pady=5)
        
        ttk.Label(login_window, text="API Secret:", style='Modern.TLabel').pack(pady=5)
        api_secret_entry = ttk.Entry(login_window, width=40, show="*", style='Modern.TEntry')
        api_secret_entry.pack(pady=5)
        
        ttk.Label(login_window, text="API Passphrase:", style='Modern.TLabel').pack(pady=5)
        api_passphrase_entry = ttk.Entry(login_window, width=40, show="*", style='Modern.TEntry')
        api_passphrase_entry.pack(pady=5)
        
        sandbox_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(login_window, text="Sandbox Mode", variable=sandbox_var, style='Modern.TLabel').pack(pady=10)
        
        def connect():
            api_key = api_key_entry.get()
            api_secret = api_secret_entry.get()
            api_passphrase = api_passphrase_entry.get()
            sandbox = sandbox_var.get()
            
            if api_key and api_secret and api_passphrase:
                self.bot.api = KuCoinAPI(api_key, api_secret, api_passphrase, sandbox)
                if self.bot.api.test_connection():
                    self.status_label.configure(text="🟢 Verbunden")
                    self.connect_button.configure(text="Trennen")
                    self.update_status("✅ Verbindung erfolgreich")
                    login_window.destroy()
                    
                    # Aktualisiere Daten
                    self.update_balance_display()
                    self.load_available_pairs()
                else:
                    messagebox.showerror("Fehler", "Verbindung fehlgeschlagen! API Keys prüfen.")
            else:
                messagebox.showerror("Fehler", "Bitte alle Felder ausfüllen!")
        
        ttk.Button(login_window, text="Verbinden", command=connect, style='Success.TButton').pack(pady=10)
    
    def load_available_pairs(self):
        """Lädt verfügbare Trading-Pairs"""
        def load_pairs():
            self.update_status("Lade verfügbare Trading-Pairs...")
            available_pairs = self.bot.get_available_pairs()
            self.root.after(0, self._update_available_pairs, available_pairs)
        
        threading.Thread(target=load_pairs, daemon=True).start()
    
    def _update_available_pairs(self, pairs):
        """Aktualisiert verfügbare Pairs in der GUI"""
        if hasattr(self, 'available_listbox'):
            self.available_listbox.delete(0, tk.END)
            self.available_pairs_list = pairs
            
            for pair in pairs[:100]:  # Begrenze auf 100 für Performance
                self.available_listbox.insert(tk.END, pair)
            
            self.update_status(f"{len(pairs)} verfügbare Pairs geladen")
            self.load_current_pairs()
    
    def load_current_pairs(self):
        """Lädt aktuell ausgewählte Pairs"""
        if hasattr(self, 'selected_listbox'):
            self.selected_listbox.delete(0, tk.END)
            for pair in self.bot.trading_pairs:
                self.selected_listbox.insert(tk.END, pair)
    
    def filter_available_pairs(self, event=None):
        """Filtert verfügbare Pairs basierend auf Suche"""
        if not hasattr(self, 'available_pairs_list') or not hasattr(self, 'available_listbox'):
            return
            
        search_term = self.search_var.get().upper()
        self.available_listbox.delete(0, tk.END)
        
        for pair in self.available_pairs_list:
            if search_term in pair:
                self.available_listbox.insert(tk.END, pair)
    
    def add_selected_pairs(self):
        """Fügt ausgewählte Pairs hinzu"""
        if hasattr(self, 'available_listbox') and hasattr(self, 'selected_listbox'):
            selected_indices = self.available_listbox.curselection()
            current_pairs = list(self.selected_listbox.get(0, tk.END))
            
            for index in selected_indices:
                pair = self.available_listbox.get(index)
                if pair not in current_pairs:
                    self.selected_listbox.insert(tk.END, pair)
    
    def remove_selected_pairs(self):
        """Entfernt ausgewählte Pairs"""
        if hasattr(self, 'selected_listbox'):
            selected_indices = self.selected_listbox.curselection()
            for index in reversed(selected_indices):
                self.selected_listbox.delete(index)
    
    def save_trading_pairs(self):
        """Speichert ausgewählte Trading-Pairs"""
        selected_pairs = list(self.selected_listbox.get(0, tk.END)) if hasattr(self, 'selected_listbox') else []
        
        if not selected_pairs:
            messagebox.showwarning("Warnung", "Bitte wählen Sie mindestens ein Trading-Pair aus!")
            return
        
        self.bot.set_trading_pairs(selected_pairs)
        messagebox.showinfo("Erfolg", f"{len(selected_pairs)} Trading-Pairs gespeichert!")
        self.update_status(f"Trading-Pairs aktualisiert: {len(selected_pairs)} Pairs")
    
    def toggle_auto_trading(self):
        """Schaltet Auto-Trading um"""
        new_state = self.auto_trading_var.get()
        
        if new_state:
            result = messagebox.askyesno(
                "Auto-Trading aktivieren", 
                "WARNUNG: Auto-Trading wird echte Trades ausführen!\n\nMöchten Sie wirklich fortfahren?"
            )
            if not result:
                self.auto_trading_var.set(False)
                return
        
        self.bot.auto_trading = self.auto_trading_var.get()
        status = "AKTIV" if self.bot.auto_trading else "INAKTIV"
        self.update_status(f"Auto-Trading: {status}")
    
    def toggle_auto_trade(self):
        """Schaltet automatische Trade-Ausführung um"""
        if not hasattr(self, 'auto_trade_running') or not self.auto_trade_running:
            self.auto_trade_running = True
            self.update_status("🤖 Auto-Trade gestartet")
            threading.Thread(target=self.auto_trading_loop, daemon=True).start()
        else:
            self.auto_trade_running = False
            self.update_status("🤖 Auto-Trade gestoppt")
    
    def auto_trading_loop(self):
        """Hintergrund-Loop für automatisches Trading"""
        while hasattr(self, 'auto_trade_running') and self.auto_trade_running:
            try:
                # Führe Analyse durch
                results = self.bot.analyze_all_cryptos_parallel()
                
                # Führe Trades basierend auf Signalen aus
                for symbol, analysis in results.items():
                    if analysis['current_signal'] in ['STRONG_BUY', 'BUY'] and analysis['confidence'] >= 70:
                        self.bot.execute_trade(symbol, analysis['current_signal'])
                
                # Warte 1 Minute bis zur nächsten Runde
                time.sleep(60)
                
            except Exception as e:
                self.update_status(f"❌ Auto-Trade Fehler: {e}")
                time.sleep(30)
    
    def save_settings(self):
        """Speichert Trading-Einstellungen"""
        try:
            stop_loss = float(self.stop_loss_var.get())
            trade_size = float(self.trade_size_var.get())
            rsi_oversold = float(self.rsi_oversold_var.get())
            rsi_overbought = float(self.rsi_overbought_var.get())
            take_profit = float(self.take_profit_var.get())
            trailing_stop = float(self.trailing_stop_var.get())
            
            self.bot.set_trading_settings(
                stop_loss=stop_loss,
                trade_size=trade_size,
                rsi_oversold=rsi_oversold,
                rsi_overbought=rsi_overbought,
                take_profit=take_profit,
                trailing_stop=trailing_stop
            )
            
            if hasattr(self, 'interval_var'):
                self.bot.set_interval(self.interval_var.get())
            
            self.update_status("Einstellungen gespeichert")
            messagebox.showinfo("Erfolg", "Einstellungen wurden gespeichert!")
            
        except ValueError:
            messagebox.showerror("Fehler", "Bitte gültige Zahlen eingeben!")
    
    def quick_signal_check(self):
        """Startet schnelle Signalprüfung"""
        def run_quick_check():
            self.bot.quick_signal_check()
            self.root.after(0, self.update_recommendations)
        
        threading.Thread(target=run_quick_check, daemon=True).start()
        self.update_status("⚡ Schnelle Signalprüfung gestartet...")
    
    def start_analysis(self):
        """Startet detaillierte Analyse"""
        threading.Thread(target=self.run_analysis, daemon=True).start()
    
    def run_analysis(self):
        """Führt Analyse im Hintergrund durch"""
        self.update_status("🔄 Starte detaillierte Analyse...")
        results = self.bot.analyze_all_cryptos_parallel()
        self.root.after(0, self.update_analysis_results, results)
    
    def update_analysis_results(self, results):
        """Aktualisiert Analyse-Ergebnisse mit verbesserten Farben"""
        if not hasattr(self, 'analysis_tree'):
            return
            
        # Lösche alte Einträge
        for item in self.analysis_tree.get_children():
            self.analysis_tree.delete(item)
            
        for item in self.dashboard_rec_tree.get_children():
            self.dashboard_rec_tree.delete(item)
            
        if not results:
            self.analysis_tree.insert('', tk.END, values=("Keine", "Daten", "verfügbar", "", "", "", "", "", ""))
            return
            
        for crypto, data in results.items():
            try:
                signal = data.get('current_signal', 'HOLD')
                confidence = data.get('confidence', 0)
                price = data.get('current_price', 0)
                rsi = data.get('rsi', 0)
                ma_signal = "Bullish" if data.get('ma_short', 0) > data.get('ma_long', 0) else "Bearish"
                performance = f"{data.get('total_return', 0):+.1f}%"
                
                # Action Text
                action_text = ""
                if "BUY" in signal and confidence >= 70:
                    action_text = "🟢 HANDELN"
                elif "SELL" in signal:
                    action_text = "🔴 VERKAUFEN"
                else:
                    action_text = "🟡 WARTEN"
                
                # Tags für Farbgebung mit besseren Kontrasten
                tags = ()
                if "BUY" in signal:
                    tags = ('buy',)
                elif "SELL" in signal:
                    tags = ('sell',)
                else:
                    tags = ('hold',)
                
                # Füge zur Analysis Treeview hinzu
                self.analysis_tree.insert('', tk.END, values=(
                    crypto, 
                    f"${price:.6f}", 
                    signal, 
                    f"{confidence:.0f}%", 
                    f"{rsi:.1f}",
                    ma_signal,
                    f"{data.get('macd', {}).get('macd', 0):.4f}",
                    performance,
                    action_text
                ), tags=tags)
                
                # Füge nur starke Signale zur Dashboard Treeview hinzu
                if "BUY" in signal and confidence >= 60:
                    self.dashboard_rec_tree.insert('', tk.END, values=(
                        crypto, 
                        f"${price:.4f}", 
                        signal, 
                        f"{confidence:.0f}%", 
                        action_text
                    ), tags=tags)
                    
            except Exception as e:
                continue
                
        # Konfiguriere Tags mit besseren Kontrasten
        if hasattr(self, 'analysis_tree'):
            self.analysis_tree.tag_configure('buy', background='#d4edda', foreground='#000000')
            self.analysis_tree.tag_configure('sell', background='#f8d7da', foreground='#000000')
            self.analysis_tree.tag_configure('hold', background='#fff3cd', foreground='#000000')
            
        if hasattr(self, 'dashboard_rec_tree'):
            self.dashboard_rec_tree.tag_configure('buy', background='#d4edda', foreground='#000000')
            self.dashboard_rec_tree.tag_configure('sell', background='#f8d7da', foreground='#000000')
            self.dashboard_rec_tree.tag_configure('hold', background='#fff3cd', foreground='#000000')
        
        self.update_status(f"✅ Analyse abgeschlossen: {len(results)} Kryptos")
    
    def update_recommendations(self):
        """Aktualisiert Empfehlungen (Alias für Kompatibilität)"""
        self.update_analysis_results(self.bot.current_recommendations)
    
    def start_backtest(self):
        """Startet Backtest"""
        threading.Thread(target=self.run_backtest, daemon=True).start()
    
    def run_backtest(self):
        """Führt Backtest im Hintergrund durch"""
        self.update_status("📊 Starte Backtest...")
        results = self.bot.run_complete_backtest()
        
        if results:
            self.root.after(0, lambda: self.update_status(f"✅ Backtest abgeschlossen: {len(results)} Kryptos"))
            self.root.after(0, self.update_analysis_results, results)
        else:
            self.root.after(0, lambda: self.update_status("❌ Backtest fehlgeschlagen"))
    
    def update_balance_display(self):
        """Aktualisiert Balance-Anzeige"""
        def update():
            try:
                balance_summary = self.bot.get_balance_summary()
                
                if balance_summary:
                    total_value = balance_summary['total_portfolio_value']
                    last_updated = balance_summary['last_updated'].strftime('%H:%M:%S')
                    
                    # Update Balance Info
                    if hasattr(self, 'balance_info_var'):
                        self.balance_info_var.set(f"Gesamtportfolio: ${total_value:,.2f}")
                    
                    # Update Portfolio Summary
                    if hasattr(self, 'portfolio_summary_var'):
                        self.portfolio_summary_var.set(f"Portfolio Wert: ${total_value:,.2f} (Stand: {last_updated})")
                    
                    # Update Portfolio Tree
                    if hasattr(self, 'portfolio_tree'):
                        for item in self.portfolio_tree.get_children():
                            self.portfolio_tree.delete(item)
                        
                        for asset in balance_summary['assets']:
                            self.portfolio_tree.insert('', tk.END, values=(
                                asset['currency'],
                                f"{asset['balance']:.6f}",
                                f"{asset['available']:.6f}",
                                f"${asset['price']:.6f}" if asset['currency'] != 'USDT' else "$1.000000",
                                f"${asset['value_usd']:,.2f}",
                                f"{asset['percentage']:.1f}%"
                            ))
                    
                    self.update_status("✅ Kontostand aktualisiert")
                else:
                    self.update_status("❌ Keine Kontostandsdaten verfügbar")
                    
            except Exception as e:
                self.update_status(f"❌ Fehler bei Balance-Update: {e}")
        
        threading.Thread(target=update, daemon=True).start()
    
    def update_portfolio_display(self):
        """Aktualisiert Portfolio-Anzeige (Alias)"""
        self.update_balance_display()
    
    def update_active_trades(self):
        """Aktualisiert aktive Trades mit aktuellem Wert - ERWEITERTE VERSION"""
        if not hasattr(self, 'trades_tree'):
            return
            
        for item in self.trades_tree.get_children():
            self.trades_tree.delete(item)
            
        # Verwende die neue Methode, die aktuelle Werte berechnet
        active_trades_with_value = self.bot.get_active_trades_with_current_value()
        
        for symbol, trade in active_trades_with_value.items():
            current_price = trade.get('current_price', 0)
            current_value = trade.get('current_value', 0)
            profit_loss = trade.get('profit_loss', 0)
            profit_loss_percent = trade.get('profit_loss_percent', 0)
            
            time_since = datetime.now() - trade['timestamp']
            hours = int(time_since.total_seconds() / 3600)
            minutes = int((time_since.total_seconds() % 3600) / 60)
            highest_price = trade.get('highest_price', trade['buy_price'])
            
            tags = ('profit',) if profit_loss >= 0 else ('loss',)
            
            self.trades_tree.insert('', tk.END, values=(
                symbol,
                f"${trade['buy_price']:.6f}",
                f"${current_price:.6f}",
                f"{trade['amount']:.6f}",
                f"${current_value:.2f}",  # NEU: Aktueller Wert in €
                f"{profit_loss_percent:+.2f}%",
                f"${profit_loss:+.2f}",
                f"{hours:02d}:{minutes:02d}",
                f"${highest_price:.6f}"
            ), tags=tags)
            
        self.trades_tree.tag_configure('profit', background='#d4edda', foreground='#000000')
        self.trades_tree.tag_configure('loss', background='#f8d7da', foreground='#000000')
    
    def close_all_trades(self):
        """Schließt alle aktiven Trades"""
        if not self.bot.active_trades:
            messagebox.showinfo("Info", "Keine aktiven Trades zum Schließen")
            return
            
        result = messagebox.askyesno(
            "Alle Trades schließen",
            f"Möchten Sie wirklich alle {len(self.bot.active_trades)} aktiven Trades schließen?\n\n"
            f"Symbole: {', '.join(self.bot.active_trades.keys())}"
        )
        
        if result:
            def close_trades():
                success = self.bot.close_all_trades()
                self.root.after(0, self.update_active_trades)
                self.root.after(0, self.update_balance_display)
                if success:
                    self.root.after(0, lambda: messagebox.showinfo("Erfolg", "Alle Trades wurden geschlossen!"))
                else:
                    self.root.after(0, lambda: messagebox.showerror("Fehler", "Fehler beim Schließen der Trades!"))
            
            threading.Thread(target=close_trades, daemon=True).start()
    
    def close_selected_trade(self):
        """Schließt den ausgewählten aktiven Trade"""
        selection = self.trades_tree.selection()
        if not selection:
            messagebox.showwarning("Warnung", "Bitte wählen Sie einen Trade aus!")
            return
            
        item = self.trades_tree.item(selection[0])
        values = item['values']
        symbol = values[0]
        
        result = messagebox.askyesno(
            "Trade schließen",
            f"Möchten Sie den Trade für {symbol} wirklich schließen?\n\n"
            f"Kaufpreis: {values[1]}\n"
            f"Aktueller Preis: {values[2]}\n"
            f"Gewinn/Verlust: {values[5]}"
        )
        
        if result:
            def close_trade():
                success = self.bot.close_trade(symbol, "Manuell geschlossen")
                self.root.after(0, self.update_active_trades)
                self.root.after(0, self.update_balance_display)
                if success:
                    self.root.after(0, lambda: messagebox.showinfo("Erfolg", f"Trade für {symbol} wurde geschlossen!"))
                else:
                    self.root.after(0, lambda: messagebox.showerror("Fehler", f"Fehler beim Schließen des Trades für {symbol}!"))
            
            threading.Thread(target=close_trade, daemon=True).start()
    
    def execute_manual_trade(self, side):
        """Führt manuellen Trade aus"""
        symbol = self.trade_symbol_entry.get().strip().upper()
        amount_str = self.trade_amount_entry.get().strip()
        
        if not symbol or not amount_str:
            messagebox.showerror("Fehler", "Bitte Symbol und Menge eingeben!")
            return
        
        try:
            amount = float(amount_str)
        except ValueError:
            messagebox.showerror("Fehler", "Ungültige Menge!")
            return
        
        # Temporär Auto-Trading aktivieren
        was_auto_trading = self.bot.auto_trading
        self.bot.auto_trading = True
        
        def execute_trade():
            success = False
            if side == "buy":
                success = self.bot.execute_trade(symbol, "MANUAL_BUY")
            else:
                # Für Verkauf müssen wir prüfen ob wir die Krypto haben
                success = self.bot.close_trade(symbol, "Manueller Verkauf")
            
            # Auto-Trading Status zurücksetzen
            self.bot.auto_trading = was_auto_trading
            
            if success:
                self.root.after(0, lambda: messagebox.showinfo("Erfolg", f"Trade für {symbol} erfolgreich!"))
                self.root.after(0, self.update_active_trades)
                self.root.after(0, self.update_balance_display)
            else:
                self.root.after(0, lambda: messagebox.showerror("Fehler", f"Trade für {symbol} fehlgeschlagen!"))
        
        # Bestätigungsdialog
        current_price = self.bot.get_current_price(symbol)
        if current_price:
            confirmation = messagebox.askyesno(
                "Trade bestätigen",
                f"Möchten Sie diesen Trade ausführen?\n\n"
                f"Symbol: {symbol}\n"
                f"Side: {side.upper()}\n"
                f"Menge: {amount}\n"
                f"Aktueller Preis: ${current_price:.6f}\n"
                f"Gesamtwert: ${amount * current_price:.2f}"
            )
            
            if confirmation:
                threading.Thread(target=execute_trade, daemon=True).start()
    
    def execute_selected_trade(self):
        """Führt Trade für ausgewählte Empfehlung aus"""
        selection = self.analysis_tree.selection()
        if not selection:
            messagebox.showwarning("Warnung", "Bitte wählen Sie eine Empfehlung aus!")
            return
            
        item = self.analysis_tree.item(selection[0])
        values = item['values']
        symbol = values[0]
        signal = values[2]
        
        if "BUY" not in signal:
            messagebox.showwarning("Warnung", "Nur KAUF-Signale können ausgeführt werden!")
            return
            
        self.execute_manual_trade_from_signal(symbol, signal)
    
    def execute_manual_trade_from_signal(self, symbol, signal):
        """Führt manuellen Trade basierend auf Signal aus"""
        # Temporär Auto-Trading aktivieren
        was_auto_trading = self.bot.auto_trading
        self.bot.auto_trading = True
        
        def execute_trade():
            success = self.bot.execute_trade(symbol, signal)
            
            # Auto-Trading Status zurücksetzen
            self.bot.auto_trading = was_auto_trading
            
            if success:
                self.root.after(0, lambda: messagebox.showinfo("Erfolg", f"Trade für {symbol} erfolgreich!"))
                self.root.after(0, self.update_active_trades)
                self.root.after(0, self.update_balance_display)
            else:
                self.root.after(0, lambda: messagebox.showerror("Fehler", f"Trade für {symbol} fehlgeschlagen!"))
        
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
                threading.Thread(target=execute_trade, daemon=True).start()
    
    def execute_all_buy_signals(self):
        """Führt alle KAUF-Signale aus"""
        buy_signals = []
        for item in self.analysis_tree.get_children():
            values = self.analysis_tree.item(item)['values']
            if "BUY" in values[2] and float(values[3].replace('%', '')) >= 70:
                buy_signals.append(values[0])
        
        if not buy_signals:
            messagebox.showinfo("Info", "Keine KAUF-Signale mit Confidence >= 70% gefunden")
            return
            
        result = messagebox.askyesno(
            "Bestätigung", 
            f"Möchten Sie {len(buy_signals)} KAUF-Signale ausführen?\n\nSymbole: {', '.join(buy_signals)}"
        )
        
        if result:
            for symbol in buy_signals:
                self.execute_manual_trade_from_signal(symbol, "STRONG_BUY")
    
    def on_analysis_recommendation_click(self, event):
        """Handle Double-Click auf Analysis-Empfehlung"""
        self.execute_selected_trade()
    
    def on_dashboard_recommendation_click(self, event):
        """Handle Double-Click auf Dashboard-Empfehlung"""
        selection = self.dashboard_rec_tree.selection()
        if selection:
            item = self.dashboard_rec_tree.item(selection[0])
            values = item['values']
            symbol = values[0]
            self.execute_manual_trade_from_signal(symbol, "STRONG_BUY")
    
    def force_cache_update(self):
        """Erzwingt Cache-Aktualisierung"""
        def update():
            self.bot.update_caches()
            self.root.after(0, self.update_balance_display)
            self.root.after(0, self.update_recommendations)
            self.root.after(0, self.update_active_trades)
            
        threading.Thread(target=update, daemon=True).start()
        self.update_status("🔄 Cache wird aktualisiert...")
    
    def update_trade_history(self):
        """Aktualisiert Trade-History"""
        if not hasattr(self, 'history_tree'):
            return
            
        try:
            for item in self.history_tree.get_children():
                self.history_tree.delete(item)
            
            trade_history = self.bot.get_trade_history_for_gui(100)
            
            if not trade_history:
                self.history_tree.insert('', tk.END, values=("Keine", "Trades", "verfügbar", "-", "-", "-", "-", "-"))
                return
                
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
                    
                    self.history_tree.insert('', tk.END, values=(
                        str(timestamp)[:16],
                        str(symbol),
                        str(side),
                        f"${float(price):.6f}",
                        f"{float(amount):.6f}",
                        f"{float(profit_loss_percent):+.2f}%",
                        f"${float(profit_loss):+.2f}",
                        str(reason)[:30]
                    ), tags=tags)
                    
                except Exception as e:
                    continue
                    
            self.history_tree.tag_configure('buy', background='#d4edda', foreground='#000000')
            self.history_tree.tag_configure('profit', background='#e8f5e8', foreground='#000000')
            self.history_tree.tag_configure('loss', background='#f8d7da', foreground='#000000')
            self.history_tree.tag_configure('neutral', background='#fff3cd', foreground='#000000')
            
        except Exception as e:
            print(f"❌ Fehler beim Aktualisieren der Trade-History: {e}")
    
    def update_tax_log(self):
        """Aktualisiert Finanzamt-Log mit korrekter Gewinnberechnung"""
        if not hasattr(self, 'tax_tree'):
            return
            
        try:
            for item in self.tax_tree.get_children():
                self.tax_tree.delete(item)
            
            recent_trades = self.bot.tax_logger.get_recent_trades(100)
            
            if not recent_trades:
                self.tax_tree.insert('', tk.END, values=("Keine", "Trades", "verfügbar", "-", "-", "-", "-", "-", "-"))
                return
                
            total_profit = 0
            total_loss = 0
            net_profit = 0
            
            for trade in reversed(recent_trades):
                try:
                    timestamp = trade.get('timestamp', 'Unbekannt')
                    side = trade.get('side', 'UNKNOWN')
                    symbol = trade.get('symbol', 'Unknown')
                    amount = trade.get('amount', 0)
                    price = trade.get('price', 0)
                    total_value = trade.get('total_value', amount * price if amount and price else 0)
                    fees = trade.get('fees', total_value * 0.001)
                    profit_loss = trade.get('profit_loss', 0)
                    reason = trade.get('reason', '')
                    
                    # Korrekte Gewinn/Verlust-Berechnung
                    if profit_loss > 0:
                        total_profit += profit_loss
                    elif profit_loss < 0:
                        total_loss += abs(profit_loss)
                    
                    net_profit += profit_loss
                    
                    tags = ()
                    if profit_loss > 0:
                        tags = ('profit',)
                    elif profit_loss < 0:
                        tags = ('loss',)
                    elif side == 'BUY':
                        tags = ('buy',)
                    else:
                        tags = ('neutral',)
                    
                    self.tax_tree.insert('', tk.END, values=(
                        str(timestamp)[:16],
                        str(side),
                        str(symbol),
                        f"{float(amount):.6f}",
                        f"${float(price):.6f}",
                        f"${float(total_value):.2f}",
                        f"${float(fees):.2f}",
                        f"${float(profit_loss):+.2f}",
                        str(reason)[:30]
                    ), tags=tags)
                    
                except Exception as e:
                    print(f"⚠️ Fehler beim Verarbeiten des Trades: {e}")
                    continue
                    
            self.tax_tree.tag_configure('buy', background='#e8f4fd', foreground='#000000')
            self.tax_tree.tag_configure('profit', background='#d4edda', foreground='#000000')
            self.tax_tree.tag_configure('loss', background='#f8d7da', foreground='#000000')
            self.tax_tree.tag_configure('neutral', background='#fff3cd', foreground='#000000')
            
            # Update Portfolio Info mit korrekten Berechnungen
            portfolio_value = self.bot.calculate_portfolio_value()
            
            if hasattr(self, 'portfolio_var'):
                self.portfolio_var.set(f"Portfolio Wert: ${portfolio_value:,.2f}")
            if hasattr(self, 'total_profit_var'):
                self.total_profit_var.set(f"Netto Gewinn/Verlust: ${net_profit:+.2f}")
            
            print(f"📊 Finanzamt-Update: {len(recent_trades)} Trades, Netto: ${net_profit:.2f}")
                
        except Exception as e:
            print(f"❌ Fehler beim Aktualisieren des Finanzamt-Logs: {e}")
    
    def force_history_update(self):
        """Erzwingt History-Update"""
        print("🔄 Erzwinge History Update...")
        
        def update():
            self.bot.load_trade_history()
            self.root.after(0, self.update_trade_history)
            self.root.after(0, self.update_tax_log)
            self.root.after(0, lambda: self.update_status("History aktualisiert"))
        
        threading.Thread(target=update, daemon=True).start()
    
    def force_tax_update(self):
        """Erzwingt Tax-Update"""
        self.update_tax_log()
        self.update_status("Finanzamt-Daten aktualisiert")
    
    def on_tab_changed(self, event):
        """Wird aufgerufen wenn Tab gewechselt wird"""
        try:
            current_tab = self.notebook.select()
            tab_text = self.notebook.tab(current_tab, "text")
            
            if tab_text == "📋 History":
                self.update_trade_history()
            elif tab_text == "🏛️ Finanzamt":
                self.update_tax_log()
            elif tab_text == "📊 Dashboard":
                self.update_active_trades()
                self.update_recommendations()
                
        except Exception as e:
            print(f"Tab change error: {e}")
    
    def generate_tax_report(self):
        """Generiert korrekten Steuerreport"""
        try:
            recent_trades = self.bot.tax_logger.get_recent_trades(1000)
            
            if not recent_trades:
                messagebox.showinfo("Info", "Keine Trades für Report verfügbar")
                return
                
            total_trades = len(recent_trades)
            buy_trades = len([t for t in recent_trades if t['side'] == 'BUY'])
            sell_trades = len([t for t in recent_trades if t['side'] == 'SELL'])
            total_volume = sum(t['total_value'] for t in recent_trades)
            
            # Korrekte Gewinn/Verlust-Berechnung
            total_profit = sum(t['profit_loss'] for t in recent_trades if t['profit_loss'] > 0)
            total_loss = abs(sum(t['profit_loss'] for t in recent_trades if t['profit_loss'] < 0))
            net_profit = sum(t['profit_loss'] for t in recent_trades)  # Direkte Summe aller profit_loss Werte
            
            # Berechne Trading-Gebühren
            total_fees = sum(t.get('fees', t['total_value'] * 0.001) for t in recent_trades)
            
            # Finde beste und schlechteste Trades
            profitable_trades = [t for t in recent_trades if t['profit_loss'] > 0]
            losing_trades = [t for t in recent_trades if t['profit_loss'] < 0]
            
            best_trade = max(recent_trades, key=lambda x: x.get('profit_loss', 0)) if recent_trades else None
            worst_trade = min(recent_trades, key=lambda x: x.get('profit_loss', 0)) if recent_trades else None
            
            report_text = f"""📊 Detaillierter Steuerreport (Letzte {total_trades} Trades)

    📈 Handelsaktivität:
    • Gesamte Trades: {total_trades}
    • Kauf-Trades: {buy_trades}
    • Verkauf-Trades: {sell_trades}
    • Erfolgsquote: {(len(profitable_trades)/total_trades*100):.1f}% ({len(profitable_trades)} profitable Trades)

    💰 Finanzielle Übersicht:
    • Handelsvolumen: ${total_volume:,.2f}
    • Gesamtgewinn: ${total_profit:,.2f}
    • Gesamtverlust: ${total_loss:,.2f}
    • Netto Gewinn/Verlust: ${net_profit:+,.2f}
    • Gezahlte Gebühren: ${total_fees:,.2f}

    🏆 Beste/Schlechteste Trades:"""
            
            if best_trade:
                report_text += f"\n• Bester Trade: {best_trade.get('symbol', 'Unknown')} (+${best_trade.get('profit_loss', 0):.2f})"
            if worst_trade:
                report_text += f"\n• Schlechtester Trade: {worst_trade.get('symbol', 'Unknown')} (${worst_trade.get('profit_loss', 0):.2f})"
            
            # Monatliche Aufschlüsselung
            from collections import defaultdict
            monthly_data = defaultdict(lambda: {'trades': 0, 'profit': 0, 'volume': 0})
            
            for trade in recent_trades:
                try:
                    # Extrahiere Jahr-Monat aus dem Timestamp
                    timestamp_str = trade.get('timestamp_iso', '') or trade.get('timestamp', '')
                    if timestamp_str:
                        month_key = timestamp_str[:7]  # YYYY-MM
                        monthly_data[month_key]['trades'] += 1
                        monthly_data[month_key]['profit'] += trade.get('profit_loss', 0)
                        monthly_data[month_key]['volume'] += trade.get('total_value', 0)
                except:
                    continue
            
            if monthly_data:
                report_text += "\n\n📅 Monatliche Performance:"
                for month in sorted(monthly_data.keys(), reverse=True)[:6]:  # Letzte 6 Monate
                    data = monthly_data[month]
                    report_text += f"\n• {month}: {data['trades']} Trades, ${data['profit']:+,.2f} Gewinn"
            
            messagebox.showinfo("Detaillierter Steuerreport", report_text)
                
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Generieren: {e}")
    
    def export_logs(self):
        """Exportiert Logs"""
        try:
            import shutil
            from datetime import datetime
            
            export_dir = f"finanzamt_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(export_dir)
            
            # Kopiere Log-Dateien
            if os.path.exists("trade_logs/trades_finanzamt.csv"):
                shutil.copy2("trade_logs/trades_finanzamt.csv", export_dir)
            if os.path.exists("trade_logs/trading_history.json"):
                shutil.copy2("trade_logs/trading_history.json", export_dir)
            
            messagebox.showinfo("Erfolg", f"Logs exportiert nach: {export_dir}")
        except Exception as e:
            messagebox.showerror("Fehler", f"Export fehlgeschlagen: {e}")
    
    def start_auto_updates(self):
        """Startet automatische Updates"""
        def auto_update_loop():
            update_count = 0
            while True:
                try:
                    # Aktualisiere aktive Trades alle 30 Sekunden
                    self.root.after(0, self.update_active_trades)
                    self.root.after(0, self.update_recommendations)
                    
                    # Kontostand nur alle 2 Minuten aktualisieren (statt jede 30 Sekunden)
                    if update_count % 4 == 0:  # 4 * 30 Sekunden = 2 Minuten
                        self.root.after(0, self.update_balance_display)
                    
                    # API Stats aktualisieren
                    api_stats = self.bot.api.get_api_stats()
                    if api_stats and hasattr(self, 'api_stats_var'):
                        self.root.after(0, lambda: self.api_stats_var.set(f"API Requests: {api_stats['request_count']}"))
                    
                    update_count += 1
                    
                except Exception as e:
                    print(f"Auto-update error: {e}")
                time.sleep(30)
                
        threading.Thread(target=auto_update_loop, daemon=True).start()
        print("✅ Auto-Updates gestartet (Kontostand nur alle 2 Minuten)")
    
    def run(self):
        """Startet die GUI"""
        self.root.mainloop()

# =============================================================================
# HAUPTPROGRAMM
# =============================================================================

def load_env_file():
    """Lädt Umgebungsvariablen aus .env Datei"""
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
    print("🚀 Starte KuCoin Trading Bot - FullHD Optimiert")
    print(f"📊 Technische Analyse: {'TA-Lib + NumPy' if TA_LIB_AVAILABLE else 'NumPy'} mit Caching")
    print("🎨 Modernes FullHD GUI geladen")
    
    # Lade API-Daten
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
        
        # Starte trotzdem mit Sandbox
        API_KEY = 'sandbox'
        API_SECRET = 'sandbox' 
        API_PASSPHRASE = 'sandbox'
        SANDBOX = True
        print("🔄 Starte im Sandbox-Modus...")
    
    try:
        # Bot initialisieren
        bot = KuCoinTradingBot(
            api_key=API_KEY,
            api_secret=API_SECRET,
            api_passphrase=API_PASSPHRASE,
            sandbox=SANDBOX
        )

        print(f"📊 Bot initialisiert mit {len(bot.trade_history)} Trades in History")
        print(f"📈 {len(bot.active_trades)} aktive Trades geladen")
        
        # GUI starten
        print("🎨 Starte FullHD GUI...")
        gui = ModernTradingGUI(bot)
        print("✅ GUI erfolgreich gestartet")
        gui.run()
        
    except Exception as e:
        print(f"❌ Fehler beim Starten: {e}")
        print("💡 Tipps zur Problembehebung:")
        print("   1. Prüfen Sie die API Keys in der .env Datei")
        print("   2. Stellen Sie sicher, dass Tkinter installiert ist")
        print("   3. Starten Sie das System neu bei Display-Problemen")

if __name__ == "__main__":
    main()