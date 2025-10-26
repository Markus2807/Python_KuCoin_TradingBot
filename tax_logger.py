import csv
import json
import os
from datetime import datetime

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