import time
import json
from datetime import datetime, timedelta
import threading
from api_client import KuCoinAPI
from tax_logger import TaxLogger

class KuCoinTradingBot:
    def __init__(self, api_key, api_secret, api_passphrase, sandbox=False):
        self.api = KuCoinAPI(api_key, api_secret, api_passphrase, sandbox)
        self.tax_logger = TaxLogger()
        self.active_trades = {}
        self.trade_history = []
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
        
        # Lade Trade-History beim Start
        self.load_trade_history()
        
        print(f"✅ KuCoin Trading Bot initialisiert - Sandbox: {sandbox}")
        
    def load_trade_history(self):
        """Lädt Trade-History aus dem Tax-Logger"""
        try:
            recent_trades = self.tax_logger.get_recent_trades(1000)  # Letzte 1000 Trades
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
        """Führt einen Trade mit echter API aus"""
        if not self.auto_trading:
            return False
                
        if symbol in self.active_trades:
            return False
                
        if len(self.active_trades) >= self.max_open_trades:
            return False
                
        current_price = self.get_current_price(symbol)
        if not current_price:
            return False
            
        portfolio_value = self.calculate_portfolio_value()
        if portfolio_value <= 0:
            return False
                
        trade_value = portfolio_value * (self.trade_size_percent / 100)
        trade_amount = trade_value / current_price
            
        valid_amount = self.api.calculate_valid_size(symbol, trade_amount)
            
        if "BUY" in signal:
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
                    
                trade_data = {
                    'symbol': symbol,
                    'side': 'BUY',
                    'amount': valid_amount,
                    'price': current_price,
                    'profit_loss': 0,
                    'profit_loss_percent': 0,
                    'reason': f"Auto-Trade: {signal}",
                    'order_id': order_result.get('orderId', 'unknown'),
                    'portfolio_value': self.calculate_portfolio_value()
                }
                self.tax_logger.log_trade(trade_data)
                
                # Aktualisiere Trade-History
                self.load_trade_history()
                
                self.last_trade_time = datetime.now()
                self.update_bot_activity(f"🟢 Trade eröffnet: {symbol} - {valid_amount:.4f} @ ${current_price:.2f}")
                return True
            else:
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
                    
                except Exception as e:
                    print(f"⚠️  Fehler beim Konvertieren des Trades: {e}")
                    continue
            
            print(f"✅ {len(gui_trades)} Trades für GUI vorbereitet")
            return gui_trades
            
        except Exception as e:
            print(f"❌ Fehler in get_trade_history_for_gui: {e}")
            return []