
import time
import json
import csv
import requests
from datetime import datetime, timedelta
import threading
from collections import defaultdict, deque
import math
import os
import random
from api_client import KuCoinAPI
from tax_logger import TaxLogger

class KuCoinTradingBot:
    def __init__(self, api_key, api_secret, api_passphrase, sandbox=False):
        # Verwende den echten API Client aus api_client.py
        self.api = KuCoinAPI(api_key, api_secret, api_passphrase, sandbox)
        self.tax_logger = TaxLogger()
        self.active_trades = {}
        self.trade_history = []
        self.current_recommendations = {}
        
        # Trading Einstellungen
        self.auto_trading = False  # Standardmäßig inaktiv für Sicherheit
        self.stop_loss_percent = 2.0
        self.trade_size_percent = 10.0
        self.max_open_trades = 5
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.backtest_interval = '15min'
        
        # Konfigurierbare Kryptowährungen
        self.trading_pairs = ['BTC-USDT', 'ETH-USDT', 'ADA-USDT', 'DOT-USDT', 'LINK-USDT']
        
        # Cache
        self.price_cache = {}
        self.balance_cache = None
        self.last_update = None
        self.next_scheduled_update = None
        self.last_trade_time = None
        
        # Performance Tracking
        self.use_quick_signals = True
        self.bot_activity_log = []
        self.gui_reference = None
        self.headless_reference = None
        
        # Teste API-Verbindung
        self.test_api_connection()
        
        print(f"✅ KuCoin Trading Bot initialisiert - Sandbox: {sandbox}")
        
    def test_api_connection(self):
        """Testet die API-Verbindung"""
        self.update_bot_activity("🔍 Teste API-Verbindung...")
        if self.api.test_connection():
            self.update_bot_activity("✅ API-Verbindung erfolgreich!")
            return True
        else:
            self.update_bot_activity("❌ API-Verbindung fehlgeschlagen!")
            return False
        
    def set_gui_reference(self, gui):
        self.gui_reference = gui
        
    def set_headless_reference(self, headless):
        self.headless_reference = headless
        
    def update_bot_activity(self, message):
        """Aktualisiert Bot-Aktivität (für GUI und Headless)"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        self.bot_activity_log.append(log_entry)
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
            symbols_info = self.api.symbols_info
            usdt_pairs = [symbol for symbol in symbols_info.keys() if symbol.endswith('-USDT')]
            return sorted(usdt_pairs)
        except Exception as e:
            self.update_bot_activity(f"❌ Fehler beim Laden der verfügbaren Pairs: {e}")
            return ['BTC-USDT', 'ETH-USDT', 'ADA-USDT', 'DOT-USDT', 'LINK-USDT']
    
    def calculate_rsi(self, prices, period=14):
        """Berechnet RSI ohne pandas"""
        if len(prices) < period + 1:
            return 50  # Neutraler Wert bei unzureichenden Daten
            
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
            
        # Verwende die letzten 'period' Werte
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
            self.update_bot_activity(f"📊 Hole historische Daten für {symbol}...")
            
            # Verwende den echten API Client
            klines_data = self.api.get_klines(symbol, interval)
            
            if klines_data:
                # Extrahiere nur die Schlusskurse
                prices = [kline['close'] for kline in klines_data]
                self.update_bot_activity(f"✅ {len(prices)} historische Preise für {symbol} erhalten")
                return prices[-limit:]  # Rückgabe der letzten 'limit' Preise
            else:
                self.update_bot_activity(f"❌ Keine historischen Daten für {symbol} verfügbar")
                return None
                
        except Exception as e:
            self.update_bot_activity(f"❌ Fehler bei historischen Daten für {symbol}: {e}")
            return None
    
    def analyze_crypto(self, symbol):
        """Analysiert Kryptowährung mit technischen Indikatoren"""
        try:
            current_price = self.get_current_price(symbol)
            if not current_price:
                return None
                
            # Hole historische Daten
            historical_data = self.get_historical_data(symbol, self.backtest_interval, 50)
            if not historical_data:
                return None
                
            # Berechne technische Indikatoren
            rsi = self.calculate_rsi(historical_data)
            ma_short = self.calculate_moving_average(historical_data, 10)
            ma_long = self.calculate_moving_average(historical_data, 20)
            
            # Generiere Signale basierend auf Indikatoren
            signals = []
            confidence = 50  # Basis Confidence
            
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
            
        except Exception as e:
            self.update_bot_activity(f"❌ Analyse Fehler für {symbol}: {e}")
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
                    signal_emoji = "🟢" if "BUY" in analysis['current_signal'] else "🔴" if "SELL" in analysis['current_signal'] else "🟡"
                    self.update_bot_activity(f"{signal_emoji} {crypto}: {analysis['current_signal']} ({analysis['confidence']}%)")
                    
            self.current_recommendations = results
            self.update_bot_activity("✅ Schnelle Signalprüfung abgeschlossen")
            return results
            
        except Exception as e:
            self.update_bot_activity(f"❌ Fehler bei schneller Signalprüfung: {e}")
            return {}
    
    def run_complete_backtest(self, pairs=None):
        """Führt vollständigen Backtest durch"""
        try:
            self.update_bot_activity("📊 Starte Backtest...")
            
            # Verwende übergebene Pairs oder Standard-Pairs
            cryptos = pairs if pairs else self.trading_pairs
            
            results = {}
            for crypto in cryptos:
                analysis = self.analyze_crypto(crypto)
                if analysis:
                    results[crypto] = analysis
                    self.update_bot_activity(f"📈 {crypto} analysiert: {analysis['current_signal']}")
                    
            self.current_recommendations = results
            self.last_update = datetime.now()
            self.next_scheduled_update = self.last_update + timedelta(minutes=15)
            
            self.update_bot_activity(f"✅ Backtest abgeschlossen - {len(results)} Kryptos analysiert")
            return results
            
        except Exception as e:
            self.update_bot_activity(f"❌ Backtest Fehler: {e}")
            return {}
    
    def get_balance_summary(self):
        """Gibt echte Kontostand-Übersicht zurück"""
        try:
            # Verwende echte API für Kontostände
            balances = self.api.get_account_balances_detailed()
            
            if not balances:
                self.update_bot_activity("❌ Keine Kontostände verfügbar")
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
                
                if value_usd > 0.01:  # Nur Assets mit signifikantem Wert anzeigen
                    assets.append({
                        'currency': currency,
                        'balance': balance['balance'],
                        'available': balance['available'],
                        'price': price,
                        'value_usd': value_usd,
                        'percentage': 0  # Wird später berechnet
                    })
                    total_value += value_usd
            
            # Berechne Prozentsätze
            for asset in assets:
                asset['percentage'] = (asset['value_usd'] / total_value) * 100 if total_value > 0 else 0
            
            return {
                'total_portfolio_value': total_value,
                'assets': sorted(assets, key=lambda x: x['value_usd'], reverse=True),
                'last_updated': datetime.now()
            }
            
        except Exception as e:
            self.update_bot_activity(f"❌ Balance Fehler: {e}")
            return None
    
    def get_current_price(self, symbol):
        """Holt aktuellen Preis von der echten API"""
        try:
            price = self.api.get_ticker(symbol)
            if price:
                return price
            else:
                return None
        except Exception as e:
            self.update_bot_activity(f"❌ Preis Fehler für {symbol}: {e}")
            return None
    
    def update_caches(self):
        """Aktualisiert alle Caches mit echten Daten"""
        try:
            self.update_bot_activity("🔄 Aktualisiere alle Caches...")
            
            # Aktualisiere Preise für alle Trading-Pairs
            for symbol in self.trading_pairs:
                self.get_current_price(symbol)
            
            # Aktualisiere Kontostände
            self.balance_cache = self.get_balance_summary()
            
            self.update_bot_activity("✅ Alle Caches aktualisiert")
            
        except Exception as e:
            self.update_bot_activity(f"❌ Cache Update Fehler: {e}")
    
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
            
        self.update_bot_activity("⚙️ Trading-Einstellungen aktualisiert")
    
    def set_interval(self, interval):
        """Setzt Analyse-Interval"""
        self.backtest_interval = interval
        self.update_bot_activity(f"🕐 Analyse-Interval geändert: {interval}")
    
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
                self.update_bot_activity(f"❌ Kann Preis für {symbol} nicht abrufen")
                return
                
            profit_loss = (current_price - trade['buy_price']) * trade['amount']
            
            # Echten Trade über API ausführen
            if self.auto_trading:
                order_result = self.api.place_order(
                    symbol=symbol,
                    side='sell',
                    order_type='market',
                    size=trade['amount']
                )
                
                if order_result:
                    order_id = order_result.get('orderId', 'unknown')
                    self.update_bot_activity(f"✅ Verkauf order platziert für {symbol}")
                else:
                    order_id = 'failed'
                    self.update_bot_activity(f"❌ Verkauf order fehlgeschlagen für {symbol}")
            else:
                order_id = 'simulated'
            
            # Logge den Trade
            closed_trade = self.tax_logger.log_trade(
                symbol=symbol,
                side='SELL', 
                amount=trade['amount'],
                price=current_price,
                reason=reason
            )
            closed_trade['profit_loss'] = profit_loss
            closed_trade['profit_loss_percent'] = (profit_loss / (trade['buy_price'] * trade['amount'])) * 100
            closed_trade['order_id'] = order_id
            
            self.update_bot_activity(f"🔒 Trade geschlossen: {symbol} - {reason} - P/L: ${profit_loss:.2f}")
    
    def execute_trade(self, symbol, signal):
        """Führt einen Trade mit echter API aus"""
        if not self.auto_trading:
            return False
            
        if symbol in self.active_trades:
            self.update_bot_activity(f"⚠️ Trade bereits aktiv für {symbol}")
            return False
            
        if len(self.active_trades) >= self.max_open_trades:
            self.update_bot_activity("⚠️ Maximale Anzahl offener Trades erreicht")
            return False
            
        current_price = self.get_current_price(symbol)
        if not current_price:
            self.update_bot_activity(f"❌ Kann Preis für {symbol} nicht abrufen")
            return False
        
        # Berechne Trade-Größe basierend auf Portfolio
        portfolio_value = self.calculate_portfolio_value()
        if portfolio_value <= 0:
            self.update_bot_activity("❌ Kein Portfolio-Wert verfügbar")
            return False
            
        trade_value = portfolio_value * (self.trade_size_percent / 100)
        trade_amount = trade_value / current_price
        
        # Validiere und korrigiere Trade-Größe
        valid_amount = self.api.calculate_valid_size(symbol, trade_amount)
        
        if "BUY" in signal:
            # Echten Trade über API ausführen
            order_result = self.api.place_order(
                symbol=symbol,
                side='buy',
                order_type='market',
                size=valid_amount
            )
            
            if order_result:
                self.active_trades[symbol] = {
                    'buy_price': current_price,
                    'amount': valid_amount,
                    'timestamp': datetime.now(),
                    'order_id': order_result.get('orderId', 'unknown')
                }
                
                self.tax_logger.log_trade(
                    symbol=symbol,
                    side='BUY',
                    amount=valid_amount,
                    price=current_price,
                    reason=f"Auto-Trade: {signal}"
                )
                self.last_trade_time = datetime.now()
                self.update_bot_activity(f"🟢 Trade eröffnet: {symbol} - {valid_amount:.4f} @ ${current_price:.2f}")
                return True
            else:
                self.update_bot_activity(f"❌ Trade fehlgeschlagen für {symbol}")
                return False
        return False

# Hauptprogramm
if __name__ == "__main__":
    # Demo-Konfiguration - ersetzen Sie diese mit Ihren echten API-Daten
    API_KEY = "demo_key"
    API_SECRET = "demo_secret" 
    API_PASSPHRASE = "demo_passphrase"
    
    bot = KuCoinTradingBot(API_KEY, API_SECRET, API_PASSPHRASE)
    
    print("🤖 KuCoin Trading Bot gestartet")
    print("📊 Führe ersten Backtest durch...")
    
    # Initialisiere Caches
    bot.update_caches()
    
    # Führe Backtest durch
    results = bot.run_complete_backtest()
    
    print(f"\n📈 Analyseergebnisse ({len(results)} Kryptos):")
    for crypto, data in results.items():
        signal_emoji = "🟢" if "BUY" in data['current_signal'] else "🔴" if "SELL" in data['current_signal'] else "🟡"
        print(f"  {signal_emoji} {crypto}: {data['current_signal']} ({data['confidence']}%)")
    
    print(f"\n💼 Portfolio Wert: ${bot.calculate_portfolio_value():.2f}")
    print("✅ Bot ist bereit für den Betrieb!")