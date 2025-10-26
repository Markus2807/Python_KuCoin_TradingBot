
import hmac
import hashlib
import base64
import json
import requests
from urllib.parse import urlencode
import time
from datetime import datetime

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
                # Sichere Konvertierung mit Fehlerbehandlung
                try:
                    base_increment = float(symbol['baseIncrement']) if symbol['baseIncrement'] else 0.0
                    quote_increment = float(symbol['quoteIncrement']) if symbol['quoteIncrement'] else 0.0
                    base_min_size = float(symbol['baseMinSize']) if symbol['baseMinSize'] else 0.0
                    base_max_size = float(symbol['baseMaxSize']) if symbol['baseMaxSize'] else 0.0
                    price_increment = float(symbol['priceIncrement']) if symbol['priceIncrement'] else 0.0
                    
                    # minFunds kann None sein, daher spezielle Behandlung
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
                    print(f"⚠️  Fehler beim Verarbeiten von Symbol {symbol['symbol']}: {e}")
                    continue
                    
            print(f"✅ Symbol-Informationen für {len(self.symbols_info)} Paare geladen")
            return True
        else:
            print("❌ Fehler beim Laden der Symbol-Informationen")
            return False
    
    def validate_order_size(self, symbol, size, price=None):
        """Validiert die Order-Größe gemäß KuCoin's Anforderungen"""
        if symbol not in self.symbols_info:
            # Versuche Symbol-Informationen zu laden falls nicht vorhanden
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
            # Berechne wie oft das Inkrement in die Größe passt
            steps = size / base_increment
            if not steps.is_integer():
                # Runde auf das nächste gültige Inkrement
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
            return desired_size  # Fallback
        
        base_increment = symbol_info['baseIncrement']
        base_min_size = symbol_info['baseMinSize']
        
        # Stelle sicher, dass die Größe mindestens dem Minimum entspricht
        if desired_size < base_min_size:
            desired_size = base_min_size
        
        # Runde auf das nächste gültige Inkrement
        if base_increment > 0:
            steps = desired_size / base_increment
            valid_steps = round(steps)
            valid_size = round(valid_steps * base_increment, 8)
            
            # Stelle sicher, dass die Größe nicht unter Minimum fällt
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
            print(f"❌ Signatur Fehler: {e}")
            return None
    
    def _get_headers(self, method, endpoint, body=''):
        try:
            timestamp = str(int(time.time() * 1000))
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
            print(f"❌ Header Fehler: {e}")
            return None
    
    def _make_request(self, method, endpoint, body='', params=None):
        """Macht API-Request mit Rate-Limiting und Logging"""
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
            return data
            
        except Exception as e:
            print(f"❌ API Request Fehler: {e}")
            return None
    
    def get_klines(self, symbol, interval, start_time=None, end_time=None):
        """Holt historische Kursdaten ohne pandas"""
        print(f"📊 Hole Candlestick-Daten für {symbol} - Intervall: {interval}")
        
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
            klines.reverse()  # Älteste zuerst
            
            # Konvertiere Daten in ein einfaches Format ohne pandas
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
                except (ValueError, IndexError) as e:
                    print(f"⚠️  Fehler beim Verarbeiten der Kline-Daten: {e}")
                    continue
            
            print(f"✅ {len(processed_data)} Candles für {symbol} erhalten")
            return processed_data
        else:
            error_msg = data.get('msg', 'Unbekannter Fehler') if data else 'Keine Verbindung'
            print(f"❌ API Fehler für {symbol}: {error_msg}")
            return None
    
    def get_account_balance(self):
        """Holt Kontostand"""
        data = self._make_request('GET', '/api/v1/accounts')
        
        if data and data['code'] == '200000':
            return data['data']
        else:
            error_msg = data.get('msg', 'Unbekannter Fehler') if data else 'Keine Verbindung'
            print(f"❌ Balance API Fehler: {error_msg}")
            return None

    def get_account_balances_detailed(self):
        """Holt detaillierte Kontostände aller Assets"""
        data = self._make_request('GET', '/api/v1/accounts')
        
        if data and data['code'] == '200000':
            # Filtere nur Trade-Accounts und Assets mit Balance > 0
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
            error_msg = data.get('msg', 'Unbekannter Fehler') if data else 'Keine Verbindung'
            print(f"❌ Detaillierte Balance API Fehler: {error_msg}")
            return None
    
    def place_order(self, symbol, side, order_type, size, price=None):
        """Platziert eine Order mit automatischer Validierung"""
        # Validiere Order-Größe
        is_valid, validation_msg = self.validate_order_size(symbol, size, price)
        if not is_valid:
            print(f"❌ Order-Validierung fehlgeschlagen: {validation_msg}")
            
            # Versuche automatische Korrektur
            corrected_size = self.calculate_valid_size(symbol, size)
            if corrected_size != size:
                print(f"🔄 Korrigiere Größe von {size} auf {corrected_size}")
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
        print(f"📨 Platziere Order: {side} {size} {symbol}")
        
        data = self._make_request('POST', '/api/v1/orders', body_str)
        
        if data and data['code'] == '200000':
            print("✅ Order erfolgreich platziert")
            return data['data']
        else:
            error_msg = data.get('msg', 'Unbekannter Fehler') if data else 'Keine Verbindung'
            print(f"❌ Order API Fehler: {error_msg}")
            return None

    def get_ticker(self, symbol):
        """Holt aktuellen Preis"""
        params = {'symbol': symbol}
        data = self._make_request('GET', '/api/v1/market/orderbook/level1', params=params)
        
        if data and data['code'] == '200000':
            return float(data['data']['price'])
        else:
            error_msg = data.get('msg', 'Unbekannter Fehler') if data else 'Keine Verbindung'
            print(f"❌ Ticker API Fehler für {symbol}: {error_msg}")
            return None

    def test_connection(self):
        """Testet die API-Verbindung"""
        print("🔍 Teste API-Verbindung...")
        # Lade zuerst Symbol-Informationen
        self.get_symbols_info()
        
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
            'last_request_time': self.last_request_time
        }