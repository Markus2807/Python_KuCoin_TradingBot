import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
from datetime import datetime
import os
import platform

class TradingBotGUI:
    def __init__(self, bot):
        self.bot = bot
        self.bot.set_gui_reference(self)
        
        self.root = tk.Tk()
        self.root.title("KuCoin Trading Bot - Raspberry Pi")
        self.root.geometry("780x460")  # Perfekt für 800×480
        self.root.configure(bg='#2c3e50')
        
        # Raspberry Pi optimierte Schriftgrößen
        self.small_font = ('Arial', 8)
        self.medium_font = ('Arial', 9)
        self.large_font = ('Arial', 10, 'bold')
        
        # Aktivitätslog
        self.bot_activity_log = []
        self.activity_log = None
        
        # Status Variable
        self.status_var = tk.StringVar(value="Bot initialisiert - Bereit")
        
        self.setup_optimized_gui()
        self.start_auto_updates()
        
    def setup_optimized_gui(self):
        """Optimierte GUI für Raspberry Pi 800×480"""
        # Haupt-Frame mit Scrollbar für kleine Displays
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Notebook für Tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Weniger Tabs für bessere Übersicht
        trading_tab = ttk.Frame(notebook)
        status_tab = ttk.Frame(notebook)
        config_tab = ttk.Frame(notebook)
        
        notebook.add(trading_tab, text="Trading")
        notebook.add(status_tab, text="Status")
        notebook.add(config_tab, text="Einstellungen")
        
        self.setup_trading_tab_optimized(trading_tab)
        self.setup_status_tab_optimized(status_tab)
        self.setup_config_tab_optimized(config_tab)
        
        # Status Bar unten
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, font=self.small_font)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def setup_trading_tab_optimized(self, parent):
        """Optimierte Trading-Tab für kleines Display"""
        # Obere Reihe: Balance und Kontrolle
        top_frame = ttk.Frame(parent)
        top_frame.pack(fill=tk.X, pady=2)
        
        # Balance Frame
        balance_frame = ttk.LabelFrame(top_frame, text="Kontostand", padding=3)
        balance_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        self.balance_info_var = tk.StringVar(value="Lade Kontostand...")
        ttk.Label(balance_frame, textvariable=self.balance_info_var, font=self.small_font).pack()
        
        ttk.Button(balance_frame, text="Aktualisieren", 
                  command=self.update_balance_display, width=12).pack(pady=2)
        
        # Control Frame
        control_frame = ttk.LabelFrame(top_frame, text="Steuerung", padding=3)
        control_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)
        
        self.auto_trading_var = tk.BooleanVar(value=self.bot.auto_trading)
        ttk.Checkbutton(control_frame, text="Auto-Trading", 
                       variable=self.auto_trading_var,
                       command=self.toggle_auto_trading,
                       font=self.small_font).pack()
        
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(pady=2)
        
        ttk.Button(btn_frame, text="Schnell-Check", 
                  command=self.quick_signal_check, width=10).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_frame, text="Backtest", 
                  command=self.start_backtest, width=8).pack(side=tk.LEFT, padx=1)
        
        # Mittlere Reihe: Empfehlungen
        rec_frame = ttk.LabelFrame(parent, text="Trading Empfehlungen", padding=3)
        rec_frame.pack(fill=tk.BOTH, expand=True, pady=2)
        
        # Baum für Empfehlungen mit Scrollbar
        tree_frame = ttk.Frame(rec_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ('Symbol', 'Signal', 'Confidence', 'Preis')
        self.rec_tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=6)
        
        # Schmalere Spalten
        self.rec_tree.column('Symbol', width=80, minwidth=80)
        self.rec_tree.column('Signal', width=80, minwidth=80)
        self.rec_tree.column('Confidence', width=70, minwidth=70)
        self.rec_tree.column('Preis', width=90, minwidth=90)
        
        for col in columns:
            self.rec_tree.heading(col, text=col)
        
        # Scrollbar für Treeview
        tree_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.rec_tree.yview)
        self.rec_tree.configure(yscrollcommand=tree_scroll.set)
        
        self.rec_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Untere Reihe: Aktive Trades
        trades_frame = ttk.LabelFrame(parent, text="Aktive Trades", padding=3)
        trades_frame.pack(fill=tk.X, pady=2)
        
        trades_columns = ('Symbol', 'Kaufpreis', 'Aktuell', 'P/L %')
        self.trades_tree = ttk.Treeview(trades_frame, columns=trades_columns, show='headings', height=3)
        
        # Noch schmalere Spalten
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
        
        # Initiale Aktualisierung
        self.update_balance_display()
        
    def setup_status_tab_optimized(self, parent):
        """Optimierte Status-Tab"""
        # Linke Seite: Bot Status
        left_frame = ttk.LabelFrame(parent, text="Bot Status", padding=5)
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
            
            ttk.Label(row_frame, text=f"{label}:", width=15, anchor=tk.W, font=self.small_font).pack(side=tk.LEFT)
            self.bot_status_vars[key] = tk.StringVar(value="-")
            ttk.Label(row_frame, textvariable=self.bot_status_vars[key], font=self.small_font).pack(side=tk.LEFT)
        
        # Aktions-Buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Cache Aktualisieren", 
                  command=self.force_cache_update).pack(pady=2)
        ttk.Button(button_frame, text="Log Leeren", 
                  command=self.clear_activity_log).pack(pady=2)
        
        # Rechte Seite: Aktivitätslog
        right_frame = ttk.LabelFrame(parent, text="Aktivitätslog", padding=5)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        self.activity_log = scrolledtext.ScrolledText(
            right_frame, 
            wrap=tk.WORD, 
            width=40, 
            height=15,
            font=('Arial', 7)  # Kleinere Schrift für Log
        )
        self.activity_log.pack(fill=tk.BOTH, expand=True)
        self.activity_log.config(state=tk.DISABLED)
        
        # Initiale Aktualisierung
        self.update_bot_status()
        
    def setup_config_tab_optimized(self, parent):
        """Optimierte Konfigurations-Tab"""
        # Trading-Pairs Auswahl
        pairs_frame = ttk.LabelFrame(parent, text="Trading-Pairs", padding=5)
        pairs_frame.pack(fill=tk.BOTH, expand=True, pady=2)
        
        # Verfügbare Pairs
        available_frame = ttk.Frame(pairs_frame)
        available_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(available_frame, text="Verfügbare Pairs:", font=self.small_font).pack(side=tk.LEFT)
        ttk.Button(available_frame, text="Laden", 
                  command=self.load_available_pairs, width=8).pack(side=tk.RIGHT, padx=2)
        
        self.available_listbox = tk.Listbox(pairs_frame, height=4, font=self.small_font)
        self.available_listbox.pack(fill=tk.X, pady=2)
        
        # Ausgewählte Pairs
        selected_frame = ttk.Frame(pairs_frame)
        selected_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(selected_frame, text="Ausgewählte Pairs:", font=self.small_font).pack(side=tk.LEFT)
        
        btn_frame = ttk.Frame(selected_frame)
        btn_frame.pack(side=tk.RIGHT)
        
        ttk.Button(btn_frame, text="Hinzufügen", 
                  command=self.add_selected_pairs, width=8).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_frame, text="Entfernen", 
                  command=self.remove_selected_pairs, width=8).pack(side=tk.LEFT, padx=1)
        
        self.selected_listbox = tk.Listbox(pairs_frame, height=4, font=self.small_font)
        self.selected_listbox.pack(fill=tk.X, pady=2)
        
        ttk.Button(pairs_frame, text="Pairs Speichern", 
                  command=self.save_trading_pairs).pack(pady=2)
        
        # Schnellauswahl für beliebte Pairs
        quick_frame = ttk.LabelFrame(parent, text="Schnellauswahl", padding=5)
        quick_frame.pack(fill=tk.X, pady=2)
        
        popular_pairs = ['BTC-USDT', 'ETH-USDT', 'ADA-USDT', 'DOT-USDT', 'LINK-USDT']
        
        for i in range(0, len(popular_pairs), 3):
            row_frame = ttk.Frame(quick_frame)
            row_frame.pack(fill=tk.X, pady=1)
            for pair in popular_pairs[i:i+3]:
                ttk.Button(row_frame, text=pair, 
                          command=lambda p=pair: self.add_single_pair(p),
                          width=10).pack(side=tk.LEFT, padx=1)
        
        # Trading Einstellungen
        settings_frame = ttk.LabelFrame(parent, text="Trading Einstellungen", padding=5)
        settings_frame.pack(fill=tk.X, pady=2)
        
        # Erste Zeile
        row1 = ttk.Frame(settings_frame)
        row1.pack(fill=tk.X, pady=1)
        
        ttk.Label(row1, text="Stop-Loss %:", width=12, font=self.small_font).pack(side=tk.LEFT)
        self.stop_loss_var = tk.StringVar(value=str(self.bot.stop_loss_percent))
        ttk.Entry(row1, textvariable=self.stop_loss_var, width=8, font=self.small_font).pack(side=tk.LEFT)
        
        ttk.Label(row1, text="Trade Größe %:", width=12, font=self.small_font).pack(side=tk.LEFT, padx=(10,0))
        self.trade_size_var = tk.StringVar(value=str(self.bot.trade_size_percent))
        ttk.Entry(row1, textvariable=self.trade_size_var, width=8, font=self.small_font).pack(side=tk.LEFT)
        
        # Zweite Zeile
        row2 = ttk.Frame(settings_frame)
        row2.pack(fill=tk.X, pady=1)
        
        ttk.Label(row2, text="RSI Oversold:", width=12, font=self.small_font).pack(side=tk.LEFT)
        self.rsi_oversold_var = tk.StringVar(value=str(self.bot.rsi_oversold))
        ttk.Entry(row2, textvariable=self.rsi_oversold_var, width=8, font=self.small_font).pack(side=tk.LEFT)
        
        ttk.Label(row2, text="RSI Overbought:", width=12, font=self.small_font).pack(side=tk.LEFT, padx=(10,0))
        self.rsi_overbought_var = tk.StringVar(value=str(self.bot.rsi_overbought))
        ttk.Entry(row2, textvariable=self.rsi_overbought_var, width=8, font=self.small_font).pack(side=tk.LEFT)
        
        ttk.Button(settings_frame, text="Einstellungen Speichern", 
                  command=self.save_settings).pack(pady=5)

    def update_bot_status(self):
        """Aktualisiert den Bot-Status für optimierte GUI"""
        try:
            # Basis Status
            self.bot_status_vars['auto_trading'].set(
                "AKTIV" if self.bot.auto_trading else "INAKTIV"
            )
            self.bot_status_vars['active_trades'].set(
                f"{len(self.bot.active_trades)}"
            )
            self.bot_status_vars['trading_pairs'].set(
                f"{len(self.bot.trading_pairs)}"
            )
            
            # Zeit Informationen
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
            
            # Trade Informationen
            if self.bot.last_trade_time:
                self.bot_status_vars['last_trade'].set(
                    self.bot.last_trade_time.strftime('%H:%M')
                )
            else:
                self.bot_status_vars['last_trade'].set("--:--")
            
            # API Statistiken
            api_stats = self.bot.api.get_api_stats()
            if api_stats:
                self.bot_status_vars['api_requests'].set(str(api_stats['request_count']))
                    
        except Exception as e:
            print(f"Status update error: {e}")
            
        # Nächste Aktualisierung in 5 Sekunden
        self.root.after(5000, self.update_bot_status)

    def update_recommendations(self):
        """Aktualisiert die Trading-Empfehlungen für optimierte GUI"""
        if not hasattr(self, 'rec_tree'):
            return
            
        for item in self.rec_tree.get_children():
            self.rec_tree.delete(item)
            
        if not self.bot.current_recommendations:
            self.rec_tree.insert('', tk.END, values=(
                "Keine", "Daten", "verfügbar", "-"
            ))
            return
            
        for crypto, data in self.bot.current_recommendations.items():
            try:
                signal = data.get('current_signal', 'HOLD')
                confidence = data.get('confidence', 0)
                price = data.get('current_price', 0)
                
                # Kürze Signal für bessere Darstellung
                short_signal = signal.replace('STRONG_', 'S_')
                
                tags = ()
                if 'BUY' in signal:
                    tags = ('buy',)
                elif 'SELL' in signal:
                    tags = ('sell',)
                else:
                    tags = ('hold',)
                    
                self.rec_tree.insert('', tk.END, values=(
                    crypto, 
                    short_signal, 
                    f"{confidence:.0f}%", 
                    f"${price:.4f}"
                ), tags=tags)
                
            except Exception as e:
                print(f"Fehler bei {crypto}: {e}")
                continue
                
        if hasattr(self, 'rec_tree'):
            self.rec_tree.tag_configure('buy', background='#d4edda')
            self.rec_tree.tag_configure('sell', background='#f8d7da')
            self.rec_tree.tag_configure('hold', background='#fff3cd')

    def update_active_trades(self):
        """Aktualisiert aktive Trades für optimierte GUI"""
        if not hasattr(self, 'trades_tree'):
            return
            
        for item in self.trades_tree.get_children():
            self.trades_tree.delete(item)
            
        for symbol, trade in self.bot.active_trades.items():
            current_price = self.bot.get_current_price(symbol)
            if current_price:
                pl_percent = ((current_price - trade['buy_price']) / trade['buy_price']) * 100
                
                tags = ('profit',) if pl_percent >= 0 else ('loss',)
                
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
        self.available_listbox.delete(0, tk.END)
        self.available_pairs_list = pairs
        
        for pair in pairs[:50]:  # Begrenze auf erste 50 Pairs für Performance
            self.available_listbox.insert(tk.END, pair)
        
        self.update_status(f"{len(pairs)} verfügbare Pairs geladen")
        self.load_current_pairs()
    
    def load_current_pairs(self):
        """Lädt aktuell ausgewählte Trading-Pairs"""
        self.selected_listbox.delete(0, tk.END)
        for pair in self.bot.trading_pairs:
            self.selected_listbox.insert(tk.END, pair)
    
    def add_selected_pairs(self):
        """Fügt ausgewählte Pairs zur Auswahlliste hinzu"""
        selected_indices = self.available_listbox.curselection()
        current_pairs = list(self.selected_listbox.get(0, tk.END))
        
        for index in selected_indices:
            pair = self.available_listbox.get(index)
            if pair not in current_pairs:
                self.selected_listbox.insert(tk.END, pair)
    
    def add_single_pair(self, pair):
        """Fügt ein einzelnes Pair zur Auswahlliste hinzu"""
        current_pairs = list(self.selected_listbox.get(0, tk.END))
        if pair not in current_pairs:
            self.selected_listbox.insert(tk.END, pair)
    
    def remove_selected_pairs(self):
        """Entfernt ausgewählte Pairs aus der Auswahlliste"""
        selected_indices = self.selected_listbox.curselection()
        for index in reversed(selected_indices):
            self.selected_listbox.delete(index)
    
    def get_selected_pairs(self):
        """Gibt alle ausgewählten Pairs zurück"""
        return list(self.selected_listbox.get(0, tk.END))
    
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
                self.update_recommendations()
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
            
            self.update_status("Einstellungen gespeichert")
            messagebox.showinfo("Erfolg", "Einstellungen wurden gespeichert!")
            
        except ValueError:
            messagebox.showerror("Fehler", "Bitte gültige Zahlen eingeben!")

    def quick_signal_check(self):
        """Startet einen schnellen Signal-Check"""
        def run_quick_check():
            self.bot.quick_signal_check()
            self.root.after(0, self.update_recommendations)
        
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
            self.update_recommendations()
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
        self.status_var.set(message)
        print(f"Status: {message}")

    def start_auto_updates(self):
        """Startet automatische Updates"""
        def update_loop():
            while True:
                try:
                    self.root.after(0, self.update_balance_display)
                    self.root.after(0, self.update_recommendations)
                    self.root.after(0, self.update_active_trades)
                    self.root.after(0, self.update_bot_status)
                except Exception as e:
                    print(f"Auto-update error: {e}")
                    
                time.sleep(30)
                
        threading.Thread(target=update_loop, daemon=True).start()
        
    def run(self):
        """Startet die GUI-Hauptschleife"""
        # Initiale Aktualisierungen
        self.update_balance_display()
        self.load_current_pairs()
        self.load_available_pairs()
        
        self.root.mainloop()