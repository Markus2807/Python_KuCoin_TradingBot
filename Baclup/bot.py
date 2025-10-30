import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import hmac
import hashlib
import base64
import json
import requests
from urllib.parse import urlencode
import time
import schedule
import os
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
import csv
import json
from decimal import Decimal, ROUND_DOWN
matplotlib.use('TkAgg')

warnings.filterwarnings('ignore')

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

# Lade Umgebungsvariablen
env_vars = load_env_file()

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
                
        self.json_log_path = os.path.join(self.log_directory, "trading_history.json")
        if not os.path.exists(self.json_log_path):
            with open(self.json_log_path, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=2, ensure_ascii=False)
    
    def log_trade(self, trade_data):
        """Protokolliert einen Trade für das Finanzamt"""
        timestamp = datetime.now().strftime('%d.%m.%Y %H:%M:%S')
        
        self._log_to_csv(timestamp, trade_data)
        self._log_to_json(timestamp, trade_data)
        self._update_daily_summary(trade_data)
        
    def _log_to_csv(self, timestamp, trade_data):
        """Loggt Trade in CSV Format"""
        csv_path = os.path.join(self.log_directory, "trades_finanzamt.csv")
        
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')
            
            total_amount = trade_data['amount'] * trade_data['price']
            fees = total_amount * 0.001
            net_amount = total_amount - fees if trade_data['side'] == 'BUY' else total_amount + fees
            profit_loss = trade_data.get('profit_loss', 0)
            profit_loss_percent = trade_data.get('profit_loss_percent', 0)
            
            writer.writerow([
                timestamp,
                trade_data['side'],
                trade_data['symbol'],
                f"{trade_data['amount']:.8f}",
                f"{trade_data['price']:.8f}",
                f"{total_amount:.2f}",
                f"{fees:.2f}",
                f"{net_amount:.2f}",
                f"{profit_loss:.2f}",
                f"{profit_loss_percent:.2f}",
                trade_data.get('reason', ''),
                trade_data.get('order_id', ''),
                trade_data.get('portfolio_value', 0)
            ])
    
    def _log_to_json(self, timestamp, trade_data):
        """Loggt Trade in JSON Format für detaillierte Historie"""
        try:
            with open(self.json_log_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except:
            history = []
            
        trade_record = {
            'timestamp': timestamp,
            'timestamp_iso': datetime.now().isoformat(),
            'trade_id': f"TRADE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{trade_data['symbol']}",
            'side': trade_data['side'],
            'symbol': trade_data['symbol'],
            'amount': float(trade_data['amount']),
            'price': float(trade_data['price']),
            'total_value': float(trade_data['amount'] * trade_data['price']),
            'fees': float(trade_data['amount'] * trade_data['price'] * 0.001),
            'reason': trade_data.get('reason', ''),
            'profit_loss': trade_data.get('profit_loss', 0),
            'profit_loss_percent': trade_data.get('profit_loss_percent', 0),
            'order_id': trade_data.get('order_id', ''),
            'portfolio_value': trade_data.get('portfolio_value', 0)
        }
        
        history.append(trade_record)
        
        with open(self.json_log_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    
    def _update_daily_summary(self, trade_data):
        """Aktualisiert tägliche Zusammenfassung"""
        today = datetime.now().strftime('%Y-%m-%d')
        summary_path = os.path.join(self.log_directory, f"daily_summary_{today}.json")
        
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
        except:
            summary = {
                'date': today,
                'total_trades': 0,
                'buy_trades': 0,
                'sell_trades': 0,
                'total_volume': 0,
                'total_fees': 0,
                'total_profit_loss': 0,
                'trades': []
            }
        
        summary['total_trades'] += 1
        if trade_data['side'] == 'BUY':
            summary['buy_trades'] += 1
        else:
            summary['sell_trades'] += 1
            
        trade_value = trade_data['amount'] * trade_data['price']
        summary['total_volume'] += trade_value
        summary['total_fees'] += trade_value * 0.001
        summary['total_profit_loss'] += trade_data.get('profit_loss', 0)
        
        summary['trades'].append({
            'symbol': trade_data['symbol'],
            'side': trade_data['side'],
            'amount': float(trade_data['amount']),
            'price': float(trade_data['price']),
            'profit_loss': trade_data.get('profit_loss', 0)
        })
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def generate_tax_report(self, start_date, end_date):
        """Generiert Steuerreport für einen Zeitraum"""
        report = {
            'period': f"{start_date} bis {end_date}",
            'generated_at': datetime.now().isoformat(),
            'total_trades': 0,
            'total_volume': 0,
            'total_fees': 0,
            'total_profit': 0,
            'total_loss': 0,
            'net_profit': 0,
            'trades_by_symbol': {}
        }
        
        try:
            with open(self.json_log_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
                
            for trade in history:
                trade_date = datetime.fromisoformat(trade['timestamp_iso']).date()
                start = datetime.strptime(start_date, '%Y-%m-%d').date()
                end = datetime.strptime(end_date, '%Y-%m-%d').date()
                
                if start <= trade_date <= end:
                    report['total_trades'] += 1
                    report['total_volume'] += trade['total_value']
                    report['total_fees'] += trade['fees']
                    
                    if trade['profit_loss'] > 0:
                        report['total_profit'] += trade['profit_loss']
                    else:
                        report['total_loss'] += abs(trade['profit_loss'])
                    
                    symbol = trade['symbol']
                    if symbol not in report['trades_by_symbol']:
                        report['trades_by_symbol'][symbol] = {
                            'trades': 0,
                            'volume': 0,
                            'profit_loss': 0
                        }
                    
                    report['trades_by_symbol'][symbol]['trades'] += 1
                    report['trades_by_symbol'][symbol]['volume'] += trade['total_value']
                    report['trades_by_symbol'][symbol]['profit_loss'] += trade['profit_loss']
            
            report['net_profit'] = report['total_profit'] - report['total_loss']
            
            return report
            
        except Exception as e:
            print(f"Fehler beim Generieren des Steuerreports: {e}")
            return None
    
    def get_recent_trades(self, limit=50):
        """Holt die neuesten Trades"""
        try:
            with open(self.json_log_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
            return history[-limit:]
        except:
            return []

class KuCoinAPI:
    def __init__(self, api_key='', api_secret='', api_passphrase='', sandbox=False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase
        self.base_url = 'https://api-sandbox.kucoin.com' if sandbox else 'https://api.kucoin.com'
        print(f"✅ KuCoin API initialisiert - Modus: {'SANDBOX' if sandbox else 'LIVE'}")
        
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
    
    def get_klines(self, symbol, interval, start_time=None, end_time=None):
        """Holt historische Kursdaten"""
        endpoint = f'/api/v1/market/candles'
        
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
            
        query_string = urlencode(params)
        full_endpoint = f"{endpoint}?{query_string}"
        headers = self._get_headers('GET', full_endpoint)
        
        if not headers:
            return None
            
        url = f"{self.base_url}{full_endpoint}"
        
        try:
            print(f"📊 Hole Daten für {symbol} - Intervall: {interval}")
            response = requests.get(url, headers=headers, timeout=30)
            data = response.json()
            
            if data['code'] == '200000' and data['data']:
                klines = data['data']
                klines.reverse()
                
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'
                ])
                
                for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                    df[col] = pd.to_numeric(df[col])
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
                print(f"✅ Daten erhalten: {len(df)} Candles für {symbol}")
                return df
            else:
                print(f"❌ API Fehler für {symbol}: {data.get('msg', 'Unbekannter Fehler')}")
                return None
                
        except Exception as e:
            print(f"❌ Fehler beim Abrufen der Daten für {symbol}: {e}")
            return None
    
    def get_account_balance(self):
        """Holt Kontostand"""
        endpoint = '/api/v1/accounts'
        headers = self._get_headers('GET', endpoint)
        
        if not headers:
            return None
            
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            data = response.json()
            
            if data['code'] == '200000':
                return data['data']
            else:
                print(f"❌ Balance API Fehler: {data.get('msg', 'Unbekannter Fehler')}")
                return None
        except Exception as e:
            print(f"❌ Fehler beim Abrufen des Kontostands: {e}")
            return None

    def get_account_balances_detailed(self):
        """Holt detaillierte Kontostände aller Assets"""
        endpoint = '/api/v1/accounts'
        headers = self._get_headers('GET', endpoint)
        
        if not headers:
            return None
            
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            data = response.json()
            
            if data['code'] == '200000':
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
                print(f"❌ Detaillierte Balance API Fehler: {data.get('msg', 'Unbekannter Fehler')}")
                return None
        except Exception as e:
            print(f"❌ Fehler beim Abrufen der detaillierten Kontostände: {e}")
            return None
    
    def place_order(self, symbol, side, order_type, size, price=None):
        """Platziert eine Order"""
        endpoint = '/api/v1/orders'
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
        headers = self._get_headers('POST', endpoint, body_str)
        
        if not headers:
            return None
            
        url = f"{self.base_url}{endpoint}"
        
        try:
            print(f"📨 Platziere Order: {side} {size} {symbol}")
            response = requests.post(url, headers=headers, json=body, timeout=30)
            data = response.json()
            
            if data['code'] == '200000':
                print("✅ Order erfolgreich platziert")
                return data['data']
            else:
                print(f"❌ Order API Fehler: {data.get('msg', 'Unbekannter Fehler')}")
                return None
        except Exception as e:
            print(f"❌ Fehler beim Platzieren der Order: {e}")
            return None

    def get_ticker(self, symbol):
        """Holt aktuellen Preis"""
        endpoint = f'/api/v1/market/orderbook/level1'
        params = {'symbol': symbol}
        query_string = urlencode(params)
        full_endpoint = f"{endpoint}?{query_string}"
        
        headers = self._get_headers('GET', full_endpoint)
        if not headers:
            return None
            
        url = f"{self.base_url}{full_endpoint}"
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            data = response.json()
            
            if data['code'] == '200000':
                return float(data['data']['price'])
            else:
                print(f"❌ Ticker API Fehler für {symbol}: {data.get('msg', 'Unbekannter Fehler')}")
                return None
        except Exception as e:
            print(f"❌ Fehler beim Abrufen des Tickers für {symbol}: {e}")
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

class KuCoinTradingBot:
    def __init__(self, api_key='', api_secret='', api_passphrase='', initial_balance=1000, sandbox=False):
        self.api = KuCoinAPI(api_key, api_secret, api_passphrase, sandbox)
        self.initial_balance = initial_balance
        self.transaction_fee = 0.001
        
        # Cache für reduzierte API-Aufrufe
        self.price_cache = {}
        self.balance_cache = None
        self.last_balance_update = None
        self.last_price_update = {}
        self.cache_duration_prices = 10  # 10 Sekunden für Preise
        self.cache_duration_balance = 30  # 30 Sekunden für Kontostand
        
        # Tax Logger für Finanzamt
        self.tax_logger = TaxLogger()
        
        self.crypto_pairs = {
            'BTC': 'BTC-USDT',
            'ETH': 'ETH-USDT',
            'XRP': 'XRP-USDT', 
            'DOGE': 'DOGE-USDT',
            'SOL': 'SOL-USDT'
        }
        
        self.auto_trading = False
        self.stop_loss_percent = 5.0
        self.trade_size_percent = 50.0
        self.max_open_trades = 3
        
        self.backtest_interval = '1hour'
        self.backtest_days = 30
        self.last_update = None
        self.current_recommendations = {}
        self.active_trades = {}
        self.trade_history = []
        self.portfolio_value = initial_balance
        
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        self.gui = None

        # Teste API-Verbindung beim Start
        self.api.test_connection()

    def get_cached_price(self, symbol):
        """Holt Preis mit Cache-Mechanismus"""
        now = time.time()
        kucoin_symbol = self.crypto_pairs[symbol]
        
        # Cache ist gültig?
        if (symbol in self.price_cache and 
            now - self.last_price_update.get(symbol, 0) < self.cache_duration_prices):
            return self.price_cache[symbol]
        
        # Neuen Preis von API holen
        current_price = self.api.get_ticker(kucoin_symbol)
        if current_price:
            self.price_cache[symbol] = current_price
            self.last_price_update[symbol] = now
        
        return current_price

    def get_cached_balance(self):
        """Holt Kontostand mit Cache-Mechanismus"""
        now = time.time()
        
        # Cache ist gültig?
        if (self.balance_cache and 
            now - self.last_balance_update < self.cache_duration_balance):
            return self.balance_cache
        
        # Neuen Kontostand von API holen
        balance = self.api.get_account_balance()
        if balance:
            self.balance_cache = balance
            self.last_balance_update = now
        
        return balance

    def get_detailed_balance(self):
        """Holt detaillierte Kontoinformationen mit aktuellen Preisen"""
        balances = self.api.get_account_balances_detailed()
        if not balances:
            return None
        
        total_portfolio_value = 0
        detailed_balances = []
        
        for balance in balances:
            currency = balance['currency']
            balance_amount = balance['balance']
            
            if currency == 'USDT':
                # Für USDT direkt den Wert berechnen
                usd_value = balance_amount
                detailed_balances.append({
                    'currency': currency,
                    'balance': balance_amount,
                    'available': balance['available'],
                    'holds': balance['holds'],
                    'current_price': 1.0,
                    'usd_value': usd_value,
                    'percentage': 0  # Wird später berechnet
                })
                total_portfolio_value += usd_value
            else:
                # Für Kryptowährungen Preis von API holen
                symbol_pair = f"{currency}-USDT"
                current_price = self.api.get_ticker(symbol_pair)
                
                if current_price:
                    usd_value = balance_amount * current_price
                    detailed_balances.append({
                        'currency': currency,
                        'balance': balance_amount,
                        'available': balance['available'],
                        'holds': balance['holds'],
                        'current_price': current_price,
                        'usd_value': usd_value,
                        'percentage': 0  # Wird später berechnet
                    })
                    total_portfolio_value += usd_value
                else:
                    # Falls Preis nicht verfügbar, trotzdem anzeigen
                    detailed_balances.append({
                        'currency': currency,
                        'balance': balance_amount,
                        'available': balance['available'],
                        'holds': balance['holds'],
                        'current_price': 0,
                        'usd_value': 0,
                        'percentage': 0
                    })
        
        # Berechne Prozentsätze
        for balance in detailed_balances:
            if total_portfolio_value > 0:
                balance['percentage'] = (balance['usd_value'] / total_portfolio_value) * 100
        
        return {
            'balances': detailed_balances,
            'total_value': total_portfolio_value,
            'last_updated': datetime.now()
        }

    def get_balance_summary(self):
        """Gibt eine Zusammenfassung des Kontostands zurück"""
        detailed_balance = self.get_detailed_balance()
        if not detailed_balance:
            return None
        
        summary = {
            'total_portfolio_value': detailed_balance['total_value'],
            'last_updated': detailed_balance['last_updated'],
            'assets': []
        }
        
        # Sortiere nach Wert (absteigend)
        sorted_balances = sorted(detailed_balance['balances'], key=lambda x: x['usd_value'], reverse=True)
        
        for balance in sorted_balances:
            if balance['usd_value'] > 0.01:  # Nur Assets mit mehr als 1 Cent anzeigen
                summary['assets'].append({
                    'currency': balance['currency'],
                    'balance': balance['balance'],
                    'available': balance['available'],
                    'price': balance['current_price'],
                    'value_usd': balance['usd_value'],
                    'percentage': balance['percentage']
                })
        
        return summary

    def set_gui_reference(self, gui):
        self.gui = gui

    def set_trading_settings(self, auto_trading=None, stop_loss=None, trade_size=None):
        if auto_trading is not None:
            self.auto_trading = auto_trading
        if stop_loss is not None:
            self.stop_loss_percent = stop_loss
        if trade_size is not None:
            self.trade_size_percent = trade_size

    def calculate_portfolio_value(self):
        """Berechnet den aktuellen Portfolio-Wert mit Cache"""
        total_value = self.initial_balance
        
        # USDT Balance von API (mit Cache)
        balance = self.get_cached_balance()
        if balance:
            usdt_balance = next((acc for acc in balance if acc['currency'] == 'USDT' and acc['type'] == 'trade'), None)
            if usdt_balance:
                total_value = float(usdt_balance['available'])
        
        # Wert der Krypto-Holdings (mit Cache)
        for symbol, trade in self.active_trades.items():
            current_price = self.get_cached_price(symbol)
            if current_price:
                total_value += trade['amount'] * current_price
        
        self.portfolio_value = total_value
        return total_value

    def check_stop_loss(self):
        """Überprüft Stop-Loss für aktive Trades mit Cache"""
        if not self.active_trades:
            return
            
        trades_to_close = []
        for symbol, trade in self.active_trades.items():
            current_price = self.get_cached_price(symbol)
            
            if current_price:
                stop_loss_price = trade['buy_price'] * (1 - self.stop_loss_percent / 100)
                if current_price <= stop_loss_price:
                    trades_to_close.append(symbol)
        
        for symbol in trades_to_close:
            self.close_trade(symbol, "STOP-LOSS")

    def close_trade(self, symbol, reason="MANUELL"):
        if symbol not in self.active_trades:
            return False
            
        trade = self.active_trades[symbol]
        current_price = self.get_cached_price(symbol)
        if not current_price:
            return False
        
        # Gewinn/Verlust berechnen
        pl_amount = (current_price - trade['buy_price']) * trade['amount']
        pl_percent = ((current_price - trade['buy_price']) / trade['buy_price']) * 100
        
        # Trade-Daten für Logging
        trade_data = {
            'symbol': symbol,
            'side': 'SELL',
            'price': current_price,
            'amount': trade['amount'],
            'profit_loss': pl_amount,
            'profit_loss_percent': pl_percent,
            'reason': reason,
            'order_id': f"MANUAL_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'portfolio_value': self.calculate_portfolio_value()
        }
        
        # Für Finanzamt loggen
        self.tax_logger.log_trade(trade_data)
        
        if self.auto_trading:
            kucoin_symbol = self.crypto_pairs[symbol]
            order = self.api.place_order(
                symbol=kucoin_symbol,
                side='sell',
                order_type='market',
                size=trade['amount']
            )
            if order:
                trade_data['order_id'] = order.get('orderId', 'UNKNOWN')
        
        # Zur History hinzufügen
        self.trade_history.append({
            **trade_data,
            'timestamp': datetime.now()
        })
        
        del self.active_trades[symbol]
        
        # Portfolio Wert aktualisieren
        self.calculate_portfolio_value()
        
        if self.gui:
            self.gui.update_active_trades()
            self.gui.update_trade_history()
            self.gui.update_tax_log()
        
        return True

    def set_interval(self, interval):
        """Setzt das Trading-Intervall"""
        valid_intervals = ['1min', '5min', '15min', '1hour', '4hour', '1day', '1week']
        if interval in valid_intervals:
            self.backtest_interval = interval
            print(f"✅ Intervall geändert auf: {interval}")
        else:
            print(f"❌ Ungültiges Intervall. Verwende: {valid_intervals}")

    def fetch_historical_data(self, symbol, days=30, interval=None):
        if interval is None:
            interval = self.backtest_interval
            
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        df = self.api.get_klines(symbol, interval, start_time, end_time)
        return df
    
    def calculate_indicators(self, df):
        if df is None or len(df) < 20:
            return df
            
        try:
            df['SMA_5'] = ta.trend.sma_indicator(df['close'], window=5)
            df['SMA_10'] = ta.trend.sma_indicator(df['close'], window=10)
            df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['EMA_12'] = ta.trend.ema_indicator(df['close'], window=12)
            df['EMA_26'] = ta.trend.ema_indicator(df['close'], window=26)
            df['RSI'] = ta.momentum.rsi(df['close'], window=14)
            
            macd = ta.trend.MACD(df['close'])
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            df['MACD_histogram'] = macd.macd_diff()
            
            bollinger = ta.volatility.BollingerBands(df['close'])
            df['BB_upper'] = bollinger.bollinger_hband()
            df['BB_lower'] = bollinger.bollinger_lband()
            df['BB_middle'] = bollinger.bollinger_mavg()
            df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
            
            df['volatility'] = df['close'].pct_change().rolling(window=10).std()
            df['momentum'] = ta.momentum.roc(df['close'], window=5)
            
        except Exception as e:
            print(f"Fehler bei Indikatorberechnung: {e}")
            
        return df
    
    def generate_trading_signal(self, df):
        if df is None or len(df) < 20:
            return 'HOLD', 0, []
            
        try:
            current = df.iloc[-1]
            signals = []
            score = 0
            
            # Trend-Analyse
            trend_score = 0
            if not pd.isna(current['SMA_5']) and not pd.isna(current['SMA_20']):
                if current['SMA_5'] > current['SMA_20']:
                    trend_score += 2
                    signals.append("SMA5 > SMA20")
                if current['SMA_10'] > current['SMA_20']:
                    trend_score += 1
                    signals.append("SMA10 > SMA20")
            
            if not pd.isna(current['EMA_12']) and not pd.isna(current['EMA_26']):
                if current['EMA_12'] > current['EMA_26']:
                    trend_score += 2
                    signals.append("EMA12 > EMA26")
            
            score += trend_score * 0.4
            
            # Momentum-Analyse
            momentum_score = 0
            if not pd.isna(current['RSI']):
                if current['RSI'] < 30:
                    momentum_score += 3
                    signals.append("RSI überverkauft")
                elif current['RSI'] > 70:
                    momentum_score -= 3
                    signals.append("RSI überkauft")
                elif current['RSI'] > 50:
                    momentum_score += 1
                    signals.append("RSI bullish")
            
            if not pd.isna(current['MACD']) and not pd.isna(current['MACD_signal']):
                if current['MACD'] > current['MACD_signal']:
                    momentum_score += 2
                    signals.append("MACD bullish")
                else:
                    momentum_score -= 1
                    signals.append("MACD bearish")
            
            score += momentum_score * 0.3
            
            # Volatilitäts-Analyse
            vol_score = 0
            if not pd.isna(current['BB_lower']) and not pd.isna(current['BB_upper']):
                if current['close'] < current['BB_lower']:
                    vol_score += 2
                    signals.append("Unter Bollinger Band")
                elif current['close'] > current['BB_upper']:
                    vol_score -= 2
                    signals.append("Über Bollinger Band")
            
            score += vol_score * 0.2
            
            # Zusätzliche Signale
            extra_score = 0
            if not pd.isna(current['momentum']):
                if current['momentum'] > 0:
                    extra_score += 1
                    signals.append("Positives Momentum")
            
            score += extra_score * 0.1
            
            confidence = min(100, max(0, (abs(score) / 5) * 100))
            
            if score >= 2.5:
                signal = 'STRONG_BUY'
            elif score >= 1:
                signal = 'BUY'
            elif score <= -2.5:
                signal = 'STRONG_SELL'
            elif score <= -1:
                signal = 'SELL'
            else:
                signal = 'HOLD'
                confidence = 50
                
        except Exception as e:
            signal = 'HOLD'
            confidence = 0
            signals = ["Analyse Fehler"]
            
        return signal, confidence, signals

    def run_backtest_for_crypto(self, crypto_symbol):
        """Führt Backtesting für eine Kryptowährung durch"""
        symbol = self.crypto_pairs.get(crypto_symbol)
        if not symbol:
            return None
            
        df = self.fetch_historical_data(symbol, self.backtest_days, self.backtest_interval)
        if df is None or len(df) == 0:
            print(f"❌ Keine Daten für {crypto_symbol} erhalten")
            return None
            
        df = self.calculate_indicators(df)
        
        if len(df) < 20:
            print(f"❌ Nicht genug Daten für {crypto_symbol}: {len(df)} Zeilen")
            return None
        
        # Backtesting Variablen
        balance = self.initial_balance
        crypto_balance = 0
        trades = []
        in_position = False
        buy_price = 0
        
        for i in range(20, len(df)):
            current_data = df.iloc[:i+1]
            signal, confidence, _ = self.generate_trading_signal(current_data)
            current_price = df.iloc[i]['close']
            current_date = df.index[i]
            
            if signal in ['BUY', 'STRONG_BUY'] and not in_position and balance > 10:
                # Kauf durchführen
                investment = balance * 0.95
                crypto_balance = (investment * 0.99) / current_price
                buy_price = current_price
                balance -= investment
                in_position = True
                
                trades.append({
                    'date': current_date,
                    'action': 'BUY',
                    'price': current_price,
                    'amount': crypto_balance,
                    'investment': investment,
                    'signal_strength': signal
                })
                
            elif signal in ['SELL', 'STRONG_SELL'] and in_position and crypto_balance > 0:
                # Verkauf durchführen
                sale_value = (crypto_balance * current_price) * 0.99
                profit_loss = ((current_price - buy_price) / buy_price) * 100
                balance += sale_value
                in_position = False
                
                trades.append({
                    'date': current_date,
                    'action': 'SELL',
                    'price': current_price,
                    'amount': crypto_balance,
                    'sale_value': sale_value,
                    'profit_loss': profit_loss,
                    'signal_strength': signal
                })
        
        # Finalen Wert berechnen
        if in_position:
            final_value = crypto_balance * df.iloc[-1]['close'] + balance
        else:
            final_value = balance
            
        total_return = ((final_value - self.initial_balance) / self.initial_balance) * 100
        
        # Aktuelle Empfehlung
        current_signal, confidence, signals = self.generate_trading_signal(df)
        current_price = df.iloc[-1]['close']
        
        # Aktuellen Preis von API holen (mit Cache)
        current_price_api = self.get_cached_price(crypto_symbol)
        if current_price_api:
            current_price = current_price_api
        
        return {
            'symbol': crypto_symbol,
            'kucoin_symbol': symbol,
            'current_price': current_price,
            'current_signal': current_signal,
            'confidence': confidence,
            'signals': signals,
            'total_return': total_return,
            'final_value': final_value,
            'total_trades': len([t for t in trades if t['action'] in ['BUY', 'SELL']]),
            'winning_trades': len([t for t in trades if 'profit_loss' in t and t['profit_loss'] > 0]),
            'losing_trades': len([t for t in trades if 'profit_loss' in t and t['profit_loss'] <= 0]),
            'data': df,
            'trades': trades,
            'in_position': in_position
        }

    def execute_trade(self, crypto_symbol):
        """Führt einen Trade aus - NICHT blockierend"""
        if crypto_symbol not in self.current_recommendations:
            return False
            
        data = self.current_recommendations[crypto_symbol]
        signal = data['current_signal']
        symbol = data['kucoin_symbol']
        current_price = data['current_price']
        
        if self.auto_trading and signal in ['STRONG_BUY', 'BUY']:
            if len(self.active_trades) >= self.max_open_trades:
                return False
            if crypto_symbol in self.active_trades:
                return False
            
            balance = self.get_cached_balance()
            if balance:
                usdt_balance = next((acc for acc in balance if acc['currency'] == 'USDT' and acc['type'] == 'trade'), None)
                if usdt_balance and float(usdt_balance['available']) > 10:
                    investment = float(usdt_balance['available']) * (self.trade_size_percent / 100)
                    size = investment / current_price
                    
                    # Trade-Daten für Logging
                    trade_data = {
                        'symbol': crypto_symbol,
                        'side': 'BUY',
                        'price': current_price,
                        'amount': size,
                        'profit_loss': 0,
                        'profit_loss_percent': 0,
                        'reason': f'Auto Trade - {signal}',
                        'order_id': f"AUTO_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        'portfolio_value': self.calculate_portfolio_value()
                    }
                    
                    # In separatem Thread ausführen um GUI nicht zu blockieren
                    def execute_order():
                        order = None
                        if self.auto_trading:
                            try:
                                order = self.api.place_order(
                                    symbol=symbol,
                                    side='buy',
                                    order_type='market',
                                    size=round(size, 6)
                                )
                                if order:
                                    trade_data['order_id'] = order.get('orderId', 'UNKNOWN')
                                    print(f"✅ Auto-Trade ausgeführt: {crypto_symbol} - {size:.6f} zu ${current_price:.6f}")
                                else:
                                    print(f"❌ Auto-Trade fehlgeschlagen für {crypto_symbol}")
                            except Exception as e:
                                print(f"❌ Fehler beim Auto-Trade: {e}")
                                return
                        
                        # Für Finanzamt loggen
                        self.tax_logger.log_trade(trade_data)
                        
                        # Aktiven Trade hinzufügen
                        self.active_trades[crypto_symbol] = {
                            'buy_price': current_price,
                            'amount': size,
                            'timestamp': datetime.now(),
                            'signal': signal,
                            'order_id': trade_data['order_id']
                        }
                        
                        # Zur History hinzufügen
                        self.trade_history.append({
                            **trade_data,
                            'timestamp': datetime.now()
                        })
                        
                        # Portfolio Wert aktualisieren
                        self.calculate_portfolio_value()
                        
                        # GUI update im Hauptthread
                        if self.gui:
                            self.gui.root.after(0, self.gui.update_active_trades)
                            self.gui.root.after(0, self.gui.update_trade_history)
                            self.gui.root.after(0, self.gui.update_tax_log)
                            self.gui.root.after(0, lambda: self.gui.update_status(f"Auto-Trade: {crypto_symbol} gekauft"))
                    
                    # Trade in separatem Thread ausführen
                    threading.Thread(target=execute_order, daemon=True).start()
                    return True
        
        return False

    def run_complete_backtest(self):
        """Führt Backtesting für alle Kryptowährungen durch"""
        print("Starte kompletten Backtest...")
        self.check_stop_loss()
        
        results = {}
        successful = 0
        
        for crypto in self.crypto_pairs.keys():
            try:
                print(f"Analysiere {crypto}...")
                result = self.run_backtest_for_crypto(crypto)
                if result:
                    results[crypto] = result
                    successful += 1
                    print(f"✅ {crypto} analysiert - Signal: {result['current_signal']}")
                    
                    # Auto-Trading in separatem Thread
                    if self.auto_trading and result['current_signal'] in ['STRONG_BUY', 'BUY']:
                        def auto_trade():
                            try:
                                time.sleep(1)  # Kurze Verzögerung
                                self.execute_trade(crypto)
                            except Exception as e:
                                print(f"❌ Auto-Trading Fehler bei {crypto}: {e}")
                        
                        threading.Thread(target=auto_trade, daemon=True).start()
                    
                    time.sleep(1)  # Kurze Pause zwischen API Calls
                    
            except Exception as e:
                print(f"❌ Fehler bei {crypto}: {str(e)}")
    
        print(f"Backtest abgeschlossen: {successful}/{len(self.crypto_pairs)} erfolgreich")
        
        self.current_recommendations = results
        self.last_update = datetime.now()
        
        # GUI update
        if self.gui:
            print("Aktualisiere GUI...")
            self.gui.update_recommendations()
            self.gui.update_status(f"Letztes Update: {self.last_update.strftime('%d.%m.%Y %H:%M:%S')}")
        
        return results

class TradingBotGUI:
    def __init__(self, bot):
        self.bot = bot
        self.bot.set_gui_reference(self)
        
        self.root = tk.Tk()
        self.root.title("KuCoin Trading Bot - Optimiert für Raspberry Pi")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2c3e50')
        
        self.setup_gui()
        self.start_auto_updates()
        
    def setup_gui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        trading_tab = ttk.Frame(notebook)
        notebook.add(trading_tab, text="📊 Trading")
        
        tax_tab = ttk.Frame(notebook)
        notebook.add(tax_tab, text="💰 Finanzamt")
        
        self.setup_trading_tab(trading_tab)
        self.setup_tax_tab(tax_tab)
        
        self.status_var = tk.StringVar(value="Bot initialisiert - Bereit")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def setup_trading_tab(self, parent):
        left_frame = ttk.Frame(parent)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        right_frame = ttk.Frame(parent)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.setup_balance_panel(left_frame)  # NEU: Balance Panel
        self.setup_control_panel(left_frame)
        self.setup_recommendations_panel(left_frame)
        self.setup_active_trades_panel(right_frame)
        self.setup_trade_history_panel(right_frame)
        
    def setup_balance_panel(self, parent):
        """Erstellt das Panel für Kontostand-Informationen"""
        balance_frame = ttk.LabelFrame(parent, text="💰 Kontostand & Bestände", padding=10)
        balance_frame.pack(fill=tk.X, pady=5)
        
        # Refresh Button
        refresh_frame = ttk.Frame(balance_frame)
        refresh_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(refresh_frame, text="🔄 Aktualisieren", 
                  command=self.update_balance_display).pack(side=tk.LEFT)
        
        # Balance Informationen
        self.balance_info_var = tk.StringVar(value="Lade Kontostand...")
        balance_label = ttk.Label(balance_frame, textvariable=self.balance_info_var, 
                                 font=('Arial', 10, 'bold'))
        balance_label.pack(anchor=tk.W)
        
        # Detaillierte Bestände
        columns = ('Asset', 'Bestand', 'Verfügbar', 'Preis', 'Wert (USD)', 'Anteil')
        self.balance_tree = ttk.Treeview(balance_frame, columns=columns, show='headings', height=6)
        
        for col in columns:
            self.balance_tree.heading(col, text=col)
            self.balance_tree.column(col, width=80)
        
        self.balance_tree.column('Asset', width=80)
        self.balance_tree.column('Bestand', width=100)
        self.balance_tree.column('Verfügbar', width=100)
        self.balance_tree.column('Preis', width=100)
        self.balance_tree.column('Wert (USD)', width=100)
        self.balance_tree.column('Anteil', width=80)
        
        self.balance_tree.pack(fill=tk.BOTH, expand=True)
        
        # Initiale Aktualisierung
        self.update_balance_display()
        
    def setup_tax_tab(self, parent):
        tax_frame = ttk.LabelFrame(parent, text="📋 Steuerliche Handelsaufzeichnungen", padding=10)
        tax_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        report_frame = ttk.Frame(tax_frame)
        report_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(report_frame, text="Steuerreport Generieren", 
                  command=self.generate_tax_report).pack(side=tk.LEFT, padx=2)
        ttk.Button(report_frame, text="Logs Exportieren", 
                  command=self.export_logs).pack(side=tk.LEFT, padx=2)
        ttk.Button(report_frame, text="Logs Anzeigen", 
                  command=self.show_tax_logs).pack(side=tk.LEFT, padx=2)
        ttk.Button(report_frame, text="Debug Info", 
                  command=self.debug_status).pack(side=tk.LEFT, padx=2)
        
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
        
    def setup_control_panel(self, parent):
        control_frame = ttk.LabelFrame(parent, text="🤖 Bot Steuerung", padding=10)
        control_frame.pack(fill=tk.X, pady=5)
        
        self.auto_trading_var = tk.BooleanVar(value=self.bot.auto_trading)
        auto_switch = ttk.Checkbutton(control_frame, text="Auto-Trading", 
                                    variable=self.auto_trading_var,
                                    command=self.toggle_auto_trading)
        auto_switch.pack(anchor=tk.W)
        
        settings_frame = ttk.Frame(control_frame)
        settings_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(settings_frame, text="Stop-Loss %:").grid(row=0, column=0, sticky=tk.W)
        self.stop_loss_var = tk.StringVar(value=str(self.bot.stop_loss_percent))
        ttk.Entry(settings_frame, textvariable=self.stop_loss_var, width=8).grid(row=0, column=1, padx=5)
        
        ttk.Label(settings_frame, text="Trade Größe %:").grid(row=0, column=2, sticky=tk.W, padx=(20,0))
        self.trade_size_var = tk.StringVar(value=str(self.bot.trade_size_percent))
        ttk.Entry(settings_frame, textvariable=self.trade_size_var, width=8).grid(row=0, column=3, padx=5)
        
        ttk.Label(settings_frame, text="Intervall:").grid(row=1, column=0, sticky=tk.W, pady=(10,0))
        self.interval_var = tk.StringVar(value=self.bot.backtest_interval)
        interval_combo = ttk.Combobox(settings_frame, textvariable=self.interval_var, 
                                    values=['1min', '5min', '15min', '1hour', '4hour', '1day'], width=8)
        interval_combo.grid(row=1, column=1, padx=5, pady=(10,0))
        
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, text="Einstellungen Speichern", 
                  command=self.save_settings).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Backtest Starten", 
                  command=self.start_backtest).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Alle Trades Schließen", 
                  command=self.close_all_trades).pack(side=tk.LEFT, padx=2)
        
    def setup_recommendations_panel(self, parent):
        rec_frame = ttk.LabelFrame(parent, text="📊 Trading Empfehlungen", padding=10)
        rec_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        columns = ('Symbol', 'Preis', 'Signal', 'Confidence', 'Performance', 'Signale')
        self.rec_tree = ttk.Treeview(rec_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.rec_tree.heading(col, text=col)
            self.rec_tree.column(col, width=90)
        
        self.rec_tree.column('Signale', width=150)
        self.rec_tree.pack(fill=tk.BOTH, expand=True)
        
    def setup_active_trades_panel(self, parent):
        trades_frame = ttk.LabelFrame(parent, text="💼 Aktive Trades", padding=10)
        trades_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        columns = ('Symbol', 'Kaufpreis', 'Aktuell', 'Menge', 'P/L %', 'P/L €', 'Seit')
        self.trades_tree = ttk.Treeview(trades_frame, columns=columns, show='headings', height=6)
        
        for col in columns:
            self.trades_tree.heading(col, text=col)
            self.trades_tree.column(col, width=80)
        
        self.trades_tree.pack(fill=tk.BOTH, expand=True)
        
        action_frame = ttk.Frame(trades_frame)
        action_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(action_frame, text="Trade Schließen", 
                  command=self.close_selected_trade).pack(side=tk.LEFT, padx=2)
        
    def setup_trade_history_panel(self, parent):
        history_frame = ttk.LabelFrame(parent, text="📈 Trade History", padding=10)
        history_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        columns = ('Datum', 'Symbol', 'Side', 'Preis', 'Menge', 'P/L %', 'P/L €', 'Grund')
        self.history_tree = ttk.Treeview(history_frame, columns=columns, show='headings', height=6)
        
        for col in columns:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=80)
        
        self.history_tree.column('Datum', width=120)
        self.history_tree.column('Grund', width=120)
        self.history_tree.pack(fill=tk.BOTH, expand=True)
        
    def update_balance_display(self):
        """Aktualisiert die Kontostand-Anzeige"""
        def update():
            try:
                balance_summary = self.bot.get_balance_summary()
                
                if balance_summary:
                    # Gesamtportfolio Wert
                    total_value = balance_summary['total_portfolio_value']
                    last_updated = balance_summary['last_updated'].strftime('%H:%M:%S')
                    
                    self.balance_info_var.set(
                        f"Gesamtportfolio: ${total_value:,.2f} (Stand: {last_updated})"
                    )
                    
                    # Detaillierte Bestände
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
                    self.balance_info_var.set("Fehler beim Laden der Kontostände")
                    
            except Exception as e:
                self.balance_info_var.set(f"Fehler: {str(e)}")
                print(f"Balance update error: {e}")
        
        # Führe im Hintergrund aus um GUI nicht zu blockieren
        threading.Thread(target=update, daemon=True).start()

    def update_recommendations(self):
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
                signals_text = ", ".join(data.get('signals', [])[:2])
                
                tags = ()
                if 'BUY' in signal:
                    tags = ('buy',)
                elif 'SELL' in signal:
                    tags = ('sell',)
                else:
                    tags = ('hold',)
                    
                self.rec_tree.insert('', tk.END, values=(
                    crypto, 
                    f"${price:.6f}", 
                    signal, 
                    f"{confidence:.0f}%", 
                    performance, 
                    signals_text
                ), tags=tags)
                
            except Exception as e:
                print(f"Fehler bei {crypto}: {e}")
                continue
                
        self.rec_tree.tag_configure('buy', background='#d4edda')
        self.rec_tree.tag_configure('sell', background='#f8d7da')
        self.rec_tree.tag_configure('hold', background='#fff3cd')

    def update_tax_log(self):
        for item in self.tax_tree.get_children():
            self.tax_tree.delete(item)
            
        recent_trades = self.bot.tax_logger.get_recent_trades(100)
        
        for trade in recent_trades:
            total_value = trade['amount'] * trade['price']
            fees = total_value * 0.001
            
            tags = ()
            if trade['profit_loss'] > 0:
                tags = ('profit',)
            elif trade['profit_loss'] < 0:
                tags = ('loss',)
            else:
                tags = ('neutral',)
                
            self.tax_tree.insert('', tk.END, values=(
                trade['timestamp'],
                trade['side'],
                trade['symbol'],
                f"{trade['amount']:.6f}",
                f"${trade['price']:.6f}",
                f"${total_value:.2f}",
                f"${fees:.2f}",
                f"${trade['profit_loss']:.2f}",
                trade['reason']
            ), tags=tags)
            
        self.tax_tree.tag_configure('profit', background='#d4edda')
        self.tax_tree.tag_configure('loss', background='#f8d7da')
        self.tax_tree.tag_configure('neutral', background='#fff3cd')
        
        portfolio_value = self.bot.calculate_portfolio_value()
        self.portfolio_var.set(f"Portfolio Wert: ${portfolio_value:.2f}")
        
        total_profit = sum(trade['profit_loss'] for trade in recent_trades if trade['side'] == 'SELL')
        self.total_profit_var.set(f"Gesamtgewinn: ${total_profit:.2f}")
    
    def generate_tax_report(self):
        report_window = tk.Toplevel(self.root)
        report_window.title("Steuerreport Generieren")
        report_window.geometry("300x150")
        
        ttk.Label(report_window, text="Startdatum (YYYY-MM-DD):").pack(pady=5)
        start_entry = ttk.Entry(report_window)
        start_entry.pack(pady=5)
        start_entry.insert(0, datetime.now().strftime('%Y-%m-01'))
        
        ttk.Label(report_window, text="Enddatum (YYYY-MM-DD):").pack(pady=5)
        end_entry = ttk.Entry(report_window)
        end_entry.pack(pady=5)
        end_entry.insert(0, datetime.now().strftime('%Y-%m-%d'))
        
        def generate():
            start_date = start_entry.get()
            end_date = end_entry.get()
            
            report = self.bot.tax_logger.generate_tax_report(start_date, end_date)
            if report:
                report_text = f"""Steuerreport für {report['period']}
                
Gesamte Trades: {report['total_trades']}
Handelsvolumen: ${report['total_volume']:.2f}
Gesamtgebühren: ${report['total_fees']:.2f}
Gesamtgewinn: ${report['total_profit']:.2f}
Gesamtverlust: ${report['total_loss']:.2f}
Netto Gewinn: ${report['net_profit']:.2f}
                """
                messagebox.showinfo("Steuerreport", report_text)
            else:
                messagebox.showerror("Fehler", "Konnte Report nicht generieren")
                
            report_window.destroy()
        
        ttk.Button(report_window, text="Report Generieren", command=generate).pack(pady=10)
    
    def export_logs(self):
        try:
            import shutil
            export_dir = f"finanzamt_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(export_dir)
            
            shutil.copy2("trade_logs/trades_finanzamt.csv", export_dir)
            shutil.copy2("trade_logs/trading_history.json", export_dir)
            
            messagebox.showinfo("Erfolg", f"Logs exportiert nach: {export_dir}")
        except Exception as e:
            messagebox.showerror("Fehler", f"Export fehlgeschlagen: {e}")
    
    def show_tax_logs(self):
        self.update_tax_log()
    
    def debug_status(self):
        print(f"Auto-Trading: {self.bot.auto_trading}")
        print(f"Aktive Trades: {len(self.bot.active_trades)}")
        print(f"Empfehlungen: {len(self.bot.current_recommendations)}")
        print(f"Letztes Update: {self.bot.last_update}")
        
        if self.bot.current_recommendations:
            for crypto, data in self.bot.current_recommendations.items():
                print(f"  {crypto}: {data['current_signal']} ({data['confidence']}%)")
    
    def toggle_auto_trading(self):
        """Schaltet Auto-Trading um - mit Bestätigung"""
        new_state = self.auto_trading_var.get()
        
        if new_state:
            # Warnung bei Aktivierung
            result = messagebox.askyesno(
                "Auto-Trading aktivieren", 
                "⚠️  WARNUNG: Auto-Trading wird echte Trades ausführen!\n\n"
                "Möchten Sie wirklich fortfahren?"
            )
            if not result:
                self.auto_trading_var.set(False)
                return
        
        self.bot.auto_trading = self.auto_trading_var.get()
        status = "AKTIV" if self.bot.auto_trading else "INAKTIV"
        self.update_status(f"Auto-Trading: {status}")
        
        if self.bot.auto_trading:
            messagebox.showinfo("Auto-Trading", 
                              "✅ Auto-Trading ist jetzt AKTIV\n\n"
                              "Der Bot wird automatisch Trades basierend auf den Signalen ausführen.")
        else:
            messagebox.showinfo("Auto-Trading", 
                              "❌ Auto-Trading ist jetzt INAKTIV\n\n"
                              "Es werden keine automatischen Trades mehr ausgeführt.")
        
    def save_settings(self):
        try:
            stop_loss = float(self.stop_loss_var.get())
            trade_size = float(self.trade_size_var.get())
            interval = self.interval_var.get()
            
            self.bot.stop_loss_percent = stop_loss
            self.bot.trade_size_percent = trade_size
            self.bot.set_interval(interval)
            
            self.update_status("Einstellungen gespeichert")
            messagebox.showinfo("Erfolg", "Einstellungen wurden gespeichert!")
            
        except ValueError:
            messagebox.showerror("Fehler", "Bitte gültige Zahlen eingeben!")
            
    def start_backtest(self):
        def run_backtest():
            try:
                self.update_status("Starte Backtest...")
                print("Backtest wird gestartet...")
                
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
                self.root.after(0, self._update_after_backtest, results, was_auto_trading)
                
            except Exception as e:
                error_msg = f"Backtest Fehler: {str(e)}"
                print(error_msg)
                self.root.after(0, lambda: self.update_status(error_msg))
        
        # Backtest in separatem Thread starten
        threading.Thread(target=run_backtest, daemon=True).start()

    def _update_after_backtest(self, results, was_auto_trading):
        """Aktualisiert die GUI nach Backtest-Abschluss"""
        try:
            if results:
                self.update_recommendations()
                
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
        if not self.bot.active_trades:
            messagebox.showinfo("Info", "Keine aktiven Trades")
            return
            
        for symbol in list(self.bot.active_trades.keys()):
            self.bot.close_trade(symbol, "MANUELL GESCHLOSSEN")
            
        self.update_active_trades()
        self.update_trade_history()
        self.update_tax_log()
        messagebox.showinfo("Erfolg", "Alle Trades geschlossen")
        
    def close_selected_trade(self):
        selection = self.trades_tree.selection()
        if not selection:
            messagebox.showwarning("Warnung", "Bitte wählen Sie einen Trade aus")
            return
            
        item = self.trades_tree.item(selection[0])
        symbol = item['values'][0]
        
        self.bot.close_trade(symbol, "MANUELL GESCHLOSSEN")
        self.update_active_trades()
        self.update_trade_history()
        self.update_tax_log()

    def update_active_trades(self):
        for item in self.trades_tree.get_children():
            self.trades_tree.delete(item)
            
        for symbol, trade in self.bot.active_trades.items():
            current_price = self.bot.get_cached_price(symbol)
            if current_price:
                pl_percent = ((current_price - trade['buy_price']) / trade['buy_price']) * 100
                pl_amount = (current_price - trade['buy_price']) * trade['amount']
                time_since = datetime.now() - trade['timestamp']
                hours = int(time_since.total_seconds() / 3600)
                minutes = int((time_since.total_seconds() % 3600) / 60)
                
                tags = ('profit',) if pl_percent >= 0 else ('loss',)
                
                self.trades_tree.insert('', tk.END, values=(
                    symbol,
                    f"${trade['buy_price']:.6f}",
                    f"${current_price:.6f}",
                    f"{trade['amount']:.4f}",
                    f"{pl_percent:+.2f}%",
                    f"${pl_amount:+.2f}",
                    f"{hours:02d}:{minutes:02d}"
                ), tags=tags)
                
        self.trades_tree.tag_configure('profit', background='#d4edda')
        self.trades_tree.tag_configure('loss', background='#f8d7da')
        
    def update_trade_history(self):
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
            
        for trade in reversed(self.bot.trade_history[-20:]):
            tags = ('buy',) if trade['side'] == 'BUY' else ('sell',)
            
            self.history_tree.insert('', tk.END, values=(
                trade['timestamp'].strftime('%d.%m.%Y %H:%M'),
                trade['symbol'],
                trade['side'],
                f"${trade['price']:.6f}",
                f"{trade['amount']:.4f}",
                f"{trade['profit_loss_percent']:+.2f}%",
                f"${trade['profit_loss']:+.2f}",
                trade['reason']
            ), tags=tags)
            
        self.history_tree.tag_configure('buy', background='#d4edda')
        self.history_tree.tag_configure('sell', background='#f8d7da')

    def update_status(self, message):
        self.status_var.set(message)

    def start_auto_updates(self):
        def auto_update():
            while True:
                try:
                    self.bot.check_stop_loss()
                    self.update_active_trades()
                    self.update_tax_log()
                    self.update_balance_display()  # NEU: Balance automatisch aktualisieren
                    time.sleep(30)  # Nur alle 30 Sekunden updaten
                except:
                    time.sleep(30)
                    
        threading.Thread(target=auto_update, daemon=True).start()

    def run(self):
        self.root.mainloop()

def main():
    print("🚀 Starte KuCoin Trading Bot - Optimiert für Raspberry Pi...")
    
    # Lade API-Daten aus .env Datei
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
    
    bot = KuCoinTradingBot(
        api_key=API_KEY,
        api_secret=API_SECRET,
        api_passphrase=API_PASSPHRASE,
        initial_balance=1000,
        sandbox=SANDBOX
    )
    
    gui = TradingBotGUI(bot)
    gui.run()

if __name__ == "__main__":
    main()