import csv
import json
import os
from datetime import datetime

class TaxLogger:
    """Klasse für Finanzamt-konforme Protokollierung aller Trades"""
    
    def __init__(self, log_directory="trade_logs"):
        self.log_directory = log_directory
        self.json_log_path = os.path.join(self.log_directory, "trading_history.json")
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
                
        # Stelle sicher, dass JSON-Datei existiert
        if not os.path.exists(self.json_log_path):
            with open(self.json_log_path, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=2, ensure_ascii=False)
            print("✅ Neue Trading-History JSON Datei erstellt")
    
    def log_trade(self, trade_data):
        """Protokolliert einen Trade für das Finanzamt - KORRIGIERT"""
        timestamp = datetime.now().strftime('%d.%m.%Y %H:%M:%S')
        
        # Logge in alle Formate
        self._log_to_csv(timestamp, trade_data)
        trade_record = self._log_to_json(timestamp, trade_data)
        self._update_daily_summary(trade_data)
        
        return trade_record
    
    def _log_to_json(self, timestamp, trade_data):
        """Loggt Trade in JSON Format - KORRIGIERT"""
        try:
            # Lade bestehende History
            if os.path.exists(self.json_log_path):
                with open(self.json_log_path, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            else:
                history = []
                
            # Erstelle Trade-Record
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
        """Holt die neuesten Trades aus der JSON-Historie - KORRIGIERT"""
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
    
    def get_trade_statistics(self):
        """Berechnet Handelsstatistiken"""
        try:
            with open(self.json_log_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            if not history:
                return {}
            
            total_trades = len(history)
            buy_trades = len([t for t in history if t['side'] == 'BUY'])
            sell_trades = len([t for t in history if t['side'] == 'SELL'])
            
            profitable_trades = len([t for t in history if t.get('profit_loss', 0) > 0])
            losing_trades = len([t for t in history if t.get('profit_loss', 0) < 0])
            
            total_profit_loss = sum(t.get('profit_loss', 0) for t in history)
            total_fees = sum(t.get('fees', 0) for t in history)
            total_volume = sum(t.get('total_value', 0) for t in history)
            
            win_rate = (profitable_trades / sell_trades * 100) if sell_trades > 0 else 0
            
            return {
                'total_trades': total_trades,
                'buy_trades': buy_trades,
                'sell_trades': sell_trades,
                'profitable_trades': profitable_trades,
                'losing_trades': losing_trades,
                'win_rate': round(win_rate, 2),
                'total_profit_loss': round(total_profit_loss, 2),
                'total_fees': round(total_fees, 2),
                'total_volume': round(total_volume, 2),
                'average_trade_value': round(total_volume / total_trades, 2) if total_trades > 0 else 0
            }
            
        except Exception as e:
            return {}
        
    def get_trade_statistics(self):
        """Berechnet detaillierte Handelsstatistiken - KORRIGIERT"""
        try:
            recent_trades = self.get_recent_trades(10000)  # Alle Trades
            
            if not recent_trades:
                return {
                    'total_trades': 0,
                    'buy_trades': 0,
                    'sell_trades': 0,
                    'total_volume': 0,
                    'total_fees': 0,
                    'total_profit': 0,
                    'total_loss': 0,
                    'net_profit': 0,
                    'win_rate': 0
                }
            
            total_trades = len(recent_trades)
            buy_trades = len([t for t in recent_trades if t.get('side') == 'BUY'])
            sell_trades = len([t for t in recent_trades if t.get('side') == 'SELL'])
            
            profitable_trades = len([t for t in recent_trades if t.get('side') == 'SELL' and t.get('profit_loss', 0) > 0])
            losing_trades = len([t for t in recent_trades if t.get('side') == 'SELL' and t.get('profit_loss', 0) < 0])
            
            total_volume = sum(t.get('total_value', 0) for t in recent_trades)
            total_fees = sum(t.get('fees', 0) for t in recent_trades)
            
            # Nur Verkaufs-Trades für Gewinn/Verlust-Berechnung
            sell_trades_list = [t for t in recent_trades if t.get('side') == 'SELL']
            total_profit = sum(t.get('profit_loss', 0) for t in sell_trades_list if t.get('profit_loss', 0) > 0)
            total_loss = abs(sum(t.get('profit_loss', 0) for t in sell_trades_list if t.get('profit_loss', 0) < 0))
            net_profit = total_profit - total_loss
            
            win_rate = (profitable_trades / sell_trades * 100) if sell_trades > 0 else 0
            
            return {
                'total_trades': total_trades,
                'buy_trades': buy_trades,
                'sell_trades': sell_trades,
                'profitable_trades': profitable_trades,
                'losing_trades': losing_trades,
                'win_rate': round(win_rate, 2),
                'total_volume': round(total_volume, 2),
                'total_fees': round(total_fees, 2),
                'total_profit': round(total_profit, 2),
                'total_loss': round(total_loss, 2),
                'net_profit': round(net_profit, 2)
            }
            
        except Exception as e:
            print(f"❌ Fehler bei Trade-Statistiken: {e}")
            return {}
        
    def generate_tax_report(self, start_date, end_date):
        """Generiert Steuerreport für einen Zeitraum - KORRIGIERT"""
        try:
            report = {
                'period': f"{start_date} bis {end_date}",
                'generated_at': datetime.now().isoformat(),
                'total_trades': 0,
                'buy_trades': 0,
                'sell_trades': 0,
                'total_volume': 0,
                'total_fees': 0,
                'total_profit': 0,
                'total_loss': 0,
                'net_profit': 0,
                'trades_by_symbol': {}
            }
            
            recent_trades = self.get_recent_trades(10000)  # Alle Trades
            
            for trade in recent_trades:
                try:
                    trade_date = datetime.fromisoformat(trade['timestamp_iso']).date()
                    start = datetime.strptime(start_date, '%Y-%m-%d').date()
                    end = datetime.strptime(end_date, '%Y-%m-%d').date()
                    
                    if start <= trade_date <= end:
                        report['total_trades'] += 1
                        
                        if trade['side'] == 'BUY':
                            report['buy_trades'] += 1
                        else:
                            report['sell_trades'] += 1
                        
                        report['total_volume'] += trade['total_value']
                        report['total_fees'] += trade['fees']
                        
                        if trade['side'] == 'SELL':
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
                        report['trades_by_symbol'][symbol]['profit_loss'] += trade.get('profit_loss', 0)
                        
                except Exception as e:
                    print(f"⚠️  Fehler beim Verarbeiten des Trades für Report: {e}")
                    continue
            
            report['net_profit'] = report['total_profit'] - report['total_loss']
            
            # Berechne zusätzliche Metriken
            if report['sell_trades'] > 0:
                report['win_rate'] = (len([t for t in recent_trades if t.get('side') == 'SELL' and t.get('profit_loss', 0) > 0]) / report['sell_trades']) * 100
            
            if report['total_trades'] > 0:
                report['average_trade_value'] = report['total_volume'] / report['total_trades']
            
            print(f"✅ Steuerreport generiert: {report['total_trades']} Trades im Zeitraum")
            return report
            
        except Exception as e:
            print(f"❌ Fehler beim Generieren des Steuerreports: {e}")
            return None