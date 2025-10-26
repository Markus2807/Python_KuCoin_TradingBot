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
        
    def setup_large_gui(self):
        """Große GUI für High-Res Displays (ab 1280x720)"""
        self.root.geometry("1400x900")
        self.root.configure(bg='#2c3e50')
        
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
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
        self.setup_recommendations_panel_large(left_frame)
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
        """Control Panel für große Displays"""
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
        ttk.Button(button_frame, text="Alle Trades Schliessen", 
                  command=self.close_all_trades).pack(side=tk.LEFT, padx=2)
        
    def setup_recommendations_panel_large(self, parent):
        """Recommendations Panel für große Displays"""
        rec_frame = ttk.LabelFrame(parent, text="Trading Empfehlungen", padding=10)
        rec_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        columns = ('Symbol', 'Preis', 'Signal', 'Confidence', 'Performance', 'Signale')
        self.rec_tree = ttk.Treeview(rec_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.rec_tree.heading(col, text=col)
            self.rec_tree.column(col, width=90)
        
        self.rec_tree.column('Signale', width=150)
        self.rec_tree.pack(fill=tk.BOTH, expand=True)
        
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
        """Tax Tab für große Displays"""
        tax_frame = ttk.LabelFrame(parent, text="Steuerliche Handelsaufzeichnungen", padding=10)
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
        ttk.Button(button_frame, text="Cache Aktualisieren", 
                  command=self.force_cache_update).pack(side=tk.LEFT, padx=2)
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
                if api_stats and 'api_requests' in self.bot_status_vars:
                    self.bot_status_vars['api_requests'].set(str(api_stats['request_count']))
                        
        except Exception as e:
            print(f"Status update error: {e}")
            
        self.root.after(5000, self.update_bot_status)

    def update_recommendations(self):
        """Aktualisiert die Trading-Empfehlungen"""
        if not hasattr(self, 'rec_tree'):
            return
            
        for item in self.rec_tree.get_children():
            self.rec_tree.delete(item)
            
        if not self.bot.current_recommendations:
            if hasattr(self, 'screen_width') and self.screen_width >= 1280:
                self.rec_tree.insert('', tk.END, values=(
                    "Keine", "Daten", "verfügbar", "", "", ""
                ))
            else:
                self.rec_tree.insert('', tk.END, values=(
                    "Keine", "Daten", "verfügbar", "-"
                ))
            return
            
        for crypto, data in self.bot.current_recommendations.items():
            try:
                signal = data.get('current_signal', 'HOLD')
                confidence = data.get('confidence', 0)
                price = data.get('current_price', 0)
                
                if hasattr(self, 'screen_width') and self.screen_width >= 1280:
                    # Große GUI
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
                else:
                    # Kleine GUI
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
        if hasattr(self, 'status_var'):
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
                    if hasattr(self, 'bot_status_vars'):
                        self.root.after(0, self.update_bot_status)
                except Exception as e:
                    print(f"Auto-update error: {e}")
                    
                time.sleep(30)
                
        threading.Thread(target=update_loop, daemon=True).start()

    # Tax und Debug Methoden
    def generate_tax_report(self):
        """Generiert einen Steuerreport"""
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
        """Exportiert die Logs"""
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
        """Zeigt die Steuerlogs an"""
        self.update_tax_log()

    def update_tax_log(self):
        """Aktualisiert die Steuerlog-Anzeige"""
        if not hasattr(self, 'tax_tree'):
            return
            
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
            
        if hasattr(self, 'tax_tree'):
            self.tax_tree.tag_configure('profit', background='#d4edda')
            self.tax_tree.tag_configure('loss', background='#f8d7da')
            self.tax_tree.tag_configure('neutral', background='#fff3cd')
        
        portfolio_value = self.bot.calculate_portfolio_value()
        if hasattr(self, 'portfolio_var'):
            self.portfolio_var.set(f"Portfolio Wert: ${portfolio_value:.2f}")
        
        total_profit = sum(trade['profit_loss'] for trade in recent_trades if trade['side'] == 'SELL')
        if hasattr(self, 'total_profit_var'):
            self.total_profit_var.set(f"Gesamtgewinn: ${total_profit:.2f}")

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