
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
from datetime import datetime
import os
import shutil
import platform

class TradingBotGUI:
    def __init__(self, bot):
        self.bot = bot
        self.bot.set_gui_reference(self)
        
        self.root = tk.Tk()
        self.root.title("KuCoin Trading Bot - Raspberry Pi")
        
        # Raspberry Pi optimierte Einstellungen
        if platform.system() == "Linux" and 'raspberrypi' in platform.uname().release.lower():
            self.root.geometry("1200x700")  # Kleinere Auflösung für Pi
            print("Raspberry Pi optimierte GUI-Einstellungen aktiviert")
        else:
            self.root.geometry("1400x900")  # Normale Größe für PC
            
        self.root.configure(bg='#2c3e50')
        
        # Aktivitätslog initialisieren
        self.bot_activity_log = []
        self.activity_log = None
        
        # Status Variable frühzeitig initialisieren
        self.status_var = tk.StringVar(value="Bot initialisiert - Bereit")
        
        self.setup_gui()
        self.start_auto_updates()
        
    def setup_gui(self):
        # Vereinfachtes Layout für Raspberry Pi
        if platform.system() == "Linux" and 'raspberrypi' in platform.uname().release.lower():
            self.setup_raspberry_gui()
        else:
            self.setup_normal_gui()
        
    def setup_raspberry_gui(self):
        """Optimiertes GUI für Raspberry Pi"""
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Weniger Tabs für bessere Performance
        trading_tab = ttk.Frame(notebook)
        notebook.add(trading_tab, text="Trading")
        
        monitoring_tab = ttk.Frame(notebook)
        notebook.add(monitoring_tab, text="Monitoring")
        
        self.setup_trading_tab_simple(trading_tab)
        self.setup_monitoring_tab_simple(monitoring_tab)
        
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def setup_normal_gui(self):
        """Normales GUI für PC"""
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        trading_tab = ttk.Frame(notebook)
        notebook.add(trading_tab, text="Trading")
        
        config_tab = ttk.Frame(notebook)
        notebook.add(config_tab, text="Konfiguration")
        
        tax_tab = ttk.Frame(notebook)
        notebook.add(tax_tab, text="Finanzamt")
        
        monitoring_tab = ttk.Frame(notebook)
        notebook.add(monitoring_tab, text="Bot Monitoring")
        
        self.setup_trading_tab(trading_tab)
        self.setup_config_tab(config_tab)
        self.setup_tax_tab(tax_tab)
        self.setup_monitoring_tab(monitoring_tab)
        
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def setup_trading_tab_simple(self, parent):
        """Vereinfachte Trading-Tab für Raspberry Pi"""
        # Balance Panel
        balance_frame = ttk.LabelFrame(parent, text="Kontostand", padding=5)
        balance_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(balance_frame, text="Aktualisieren", 
                  command=self.update_balance_display).pack(side=tk.LEFT)
        
        self.balance_info_var = tk.StringVar(value="Lade Kontostand...")
        ttk.Label(balance_frame, textvariable=self.balance_info_var, 
                 font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=10)
        
        # Control Panel
        control_frame = ttk.LabelFrame(parent, text="Steuerung", padding=5)
        control_frame.pack(fill=tk.X, pady=2)
        
        self.auto_trading_var = tk.BooleanVar(value=self.bot.auto_trading)
        ttk.Checkbutton(control_frame, text="Auto-Trading", 
                       variable=self.auto_trading_var,
                       command=self.toggle_auto_trading).pack(side=tk.LEFT)
        
        ttk.Button(control_frame, text="Schnell-Check", 
                  command=self.quick_signal_check).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Backtest", 
                  command=self.start_backtest).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Alle Schliessen", 
                  command=self.close_all_trades).pack(side=tk.LEFT, padx=5)
        
        # Empfehlungen
        rec_frame = ttk.LabelFrame(parent, text="Empfehlungen", padding=5)
        rec_frame.pack(fill=tk.BOTH, expand=True, pady=2)
        
        columns = ('Symbol', 'Preis', 'Signal', 'Confidence')
        self.rec_tree = ttk.Treeview(rec_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.rec_tree.heading(col, text=col)
            self.rec_tree.column(col, width=100)
        
        self.rec_tree.pack(fill=tk.BOTH, expand=True)
        
        # Initiale Aktualisierung
        self.update_balance_display()
        
    def setup_monitoring_tab_simple(self, parent):
        """Vereinfachtes Monitoring für Raspberry Pi"""
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Status Informationen
        status_frame = ttk.LabelFrame(main_frame, text="Bot Status", padding=10)
        status_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.bot_status_vars = {}
        
        status_fields = [
            ("Auto-Trading", "auto_trading"),
            ("Aktive Trades", "active_trades"),
            ("Trading-Pairs", "trading_pairs"),
            ("Letzte Analyse", "last_analysis"),
            ("Nächste Analyse", "next_analysis")
        ]
        
        for i, (label, key) in enumerate(status_fields):
            ttk.Label(status_frame, text=f"{label}:").grid(row=i, column=0, sticky=tk.W, pady=2)
            self.bot_status_vars[key] = tk.StringVar(value="-")
            ttk.Label(status_frame, textvariable=self.bot_status_vars[key]).grid(row=i, column=1, sticky=tk.W, pady=2, padx=(10, 0))
        
        # Aktivitätslog
        log_frame = ttk.LabelFrame(main_frame, text="Aktivitätslog", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.activity_log = scrolledtext.ScrolledText(
            log_frame, 
            wrap=tk.WORD, 
            width=60, 
            height=15,
            font=('Arial', 8)
        )
        self.activity_log.pack(fill=tk.BOTH, expand=True)
        self.activity_log.config(state=tk.DISABLED)
        
        # Aktions-Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, text="Sofort Analyse", 
                  command=self.quick_signal_check).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Cache Aktualisieren", 
                  command=self.force_cache_update).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Log Leeren", 
                  command=self.clear_activity_log).pack(side=tk.LEFT, padx=2)
        
        # Initiale Aktualisierung
        self.update_bot_status()

    def setup_trading_tab(self, parent):
        left_frame = ttk.Frame(parent)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        right_frame = ttk.Frame(parent)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.setup_balance_panel(left_frame)
        self.setup_control_panel(left_frame)
        self.setup_recommendations_panel(left_frame)
        self.setup_active_trades_panel(right_frame)
        self.setup_trade_history_panel(right_frame)
        
    def setup_balance_panel(self, parent):
        """Erstellt das Panel für Kontostand-Informationen"""
        balance_frame = ttk.LabelFrame(parent, text="Kontostand & Bestände", padding=10)
        balance_frame.pack(fill=tk.X, pady=5)
        
        # Refresh Button
        refresh_frame = ttk.Frame(balance_frame)
        refresh_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(refresh_frame, text="Aktualisieren", 
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
        
    def setup_control_panel(self, parent):
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
        
    def setup_recommendations_panel(self, parent):
        rec_frame = ttk.LabelFrame(parent, text="Trading Empfehlungen", padding=10)
        rec_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        columns = ('Symbol', 'Preis', 'Signal', 'Confidence', 'Performance', 'Signale')
        self.rec_tree = ttk.Treeview(rec_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.rec_tree.heading(col, text=col)
            self.rec_tree.column(col, width=90)
        
        self.rec_tree.column('Signale', width=150)
        self.rec_tree.pack(fill=tk.BOTH, expand=True)
        
    def setup_active_trades_panel(self, parent):
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
        
    def setup_trade_history_panel(self, parent):
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
        
    def setup_config_tab(self, parent):
        """Erstellt das Konfigurations-Tab für Trading-Pairs"""
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Linke Seite - Verfügbare Pairs
        left_frame = ttk.LabelFrame(main_frame, text="Verfügbare Trading-Pairs", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Rechte Seite - Ausgewählte Pairs
        right_frame = ttk.LabelFrame(main_frame, text="Ausgewählte Trading-Pairs", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Verfügbare Pairs
        available_frame = ttk.Frame(left_frame)
        available_frame.pack(fill=tk.BOTH, expand=True)
        
        # Suchleiste für verfügbare Pairs
        search_frame = ttk.Frame(available_frame)
        search_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(search_frame, text="Suchen:").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=20)
        search_entry.pack(side=tk.LEFT, padx=5)
        search_entry.bind('<KeyRelease>', self.filter_available_pairs)
        
        ttk.Button(search_frame, text="Alle laden", 
                  command=self.load_available_pairs).pack(side=tk.RIGHT, padx=5)
        
        # Liste verfügbarer Pairs
        self.available_listbox = tk.Listbox(available_frame, selectmode=tk.MULTIPLE, font=('Arial', 10))
        self.available_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Scrollbar für verfügbare Pairs
        available_scrollbar = ttk.Scrollbar(self.available_listbox)
        available_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.available_listbox.config(yscrollcommand=available_scrollbar.set)
        available_scrollbar.config(command=self.available_listbox.yview)
        
        # Buttons für verfügbare Pairs
        available_buttons = ttk.Frame(available_frame)
        available_buttons.pack(fill=tk.X, pady=5)
        
        ttk.Button(available_buttons, text="Auswählen", 
                  command=self.add_selected_pairs).pack(side=tk.LEFT, padx=2)
        ttk.Button(available_buttons, text="Alle auswählen", 
                  command=self.add_all_pairs).pack(side=tk.LEFT, padx=2)
        
        # Ausgewählte Pairs
        selected_frame = ttk.Frame(right_frame)
        selected_frame.pack(fill=tk.BOTH, expand=True)
        
        # Liste ausgewählter Pairs
        self.selected_listbox = tk.Listbox(selected_frame, selectmode=tk.MULTIPLE, font=('Arial', 10))
        self.selected_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Scrollbar für ausgewählte Pairs
        selected_scrollbar = ttk.Scrollbar(self.selected_listbox)
        selected_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.selected_listbox.config(yscrollcommand=selected_scrollbar.set)
        selected_scrollbar.config(command=self.selected_listbox.yview)
        
        # Buttons für ausgewählte Pairs
        selected_buttons = ttk.Frame(selected_frame)
        selected_buttons.pack(fill=tk.X, pady=5)
        
        ttk.Button(selected_buttons, text="Entfernen", 
                  command=self.remove_selected_pairs).pack(side=tk.LEFT, padx=2)
        ttk.Button(selected_buttons, text="Alle entfernen", 
                  command=self.remove_all_pairs).pack(side=tk.LEFT, padx=2)
        ttk.Button(selected_buttons, text="Speichern", 
                  command=self.save_trading_pairs).pack(side=tk.RIGHT, padx=2)
        
        # Standard-Pairs vorschlagen
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
        
        # Initial verfügbare Pairs laden
        self.load_available_pairs()
        self.load_current_pairs()
        
    def setup_tax_tab(self, parent):
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
        
    def setup_monitoring_tab(self, parent):
        """Erstellt das Monitoring-Tab für Bot-Aktivitäten"""
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Linke Seite - Bot Status
        left_frame = ttk.LabelFrame(main_frame, text="Bot Status", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Rechte Seite - Aktivitätslog
        right_frame = ttk.LabelFrame(main_frame, text="Aktivitätslog", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Bot Status Informationen
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
            ttk.Label(left_frame, textvariable=self.bot_status_vars[key], 
                     font=('Arial', 9, 'bold')).grid(row=i, column=1, sticky=tk.W, pady=2, padx=(10, 0))
        
        # API Statistiken
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).grid(row=len(status_fields), column=0, 
                                                           columnspan=2, sticky=tk.EW, pady=10)
        
        ttk.Label(left_frame, text="API Statistiken", font=('Arial', 10, 'bold')).grid(
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
        
        # Aktions-Buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.grid(row=len(status_fields)+4, column=0, columnspan=2, sticky=tk.EW, pady=20)
        
        ttk.Button(button_frame, text="Sofort Analyse", 
                  command=self.quick_signal_check).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Cache Aktualisieren", 
                  command=self.force_cache_update).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Log Leeren", 
                  command=self.clear_activity_log).pack(side=tk.LEFT, padx=2)
        
        # Aktivitätslog
        self.activity_log = scrolledtext.ScrolledText(
            right_frame, 
            wrap=tk.WORD, 
            width=60, 
            height=20,
            font=('Consolas', 9)
        )
        self.activity_log.pack(fill=tk.BOTH, expand=True)
        self.activity_log.config(state=tk.DISABLED)
        
        # Initiale Aktualisierung
        self.update_bot_status()

    # Konfigurations-Methoden
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
        
        for pair in pairs:
            self.available_listbox.insert(tk.END, pair)
        
        self.update_status(f"{len(pairs)} verfügbare Trading-Pairs geladen")
    
    def load_current_pairs(self):
        """Lädt aktuell ausgewählte Trading-Pairs"""
        self.selected_listbox.delete(0, tk.END)
        for pair in self.bot.trading_pairs:
            self.selected_listbox.insert(tk.END, pair)
    
    def filter_available_pairs(self, event=None):
        """Filtert die verfügbaren Pairs basierend auf der Suche"""
        if not hasattr(self, 'available_pairs_list'):
            return
            
        search_term = self.search_var.get().upper()
        self.available_listbox.delete(0, tk.END)
        
        for pair in self.available_pairs_list:
            if search_term in pair:
                self.available_listbox.insert(tk.END, pair)
    
    def add_selected_pairs(self):
        """Fügt ausgewählte Pairs zur Auswahlliste hinzu"""
        selected_indices = self.available_listbox.curselection()
        current_pairs = self.get_selected_pairs()
        
        for index in selected_indices:
            pair = self.available_listbox.get(index)
            if pair not in current_pairs:
                self.selected_listbox.insert(tk.END, pair)
    
    def add_all_pairs(self):
        """Fügt alle verfügbaren Pairs zur Auswahlliste hinzu"""
        if not hasattr(self, 'available_pairs_list'):
            return
            
        self.selected_listbox.delete(0, tk.END)
        for pair in self.available_pairs_list:
            self.selected_listbox.insert(tk.END, pair)
    
    def add_single_pair(self, pair):
        """Fügt ein einzelnes Pair zur Auswahlliste hinzu"""
        current_pairs = self.get_selected_pairs()
        if pair not in current_pairs:
            self.selected_listbox.insert(tk.END, pair)
    
    def remove_selected_pairs(self):
        """Entfernt ausgewählte Pairs aus der Auswahlliste"""
        selected_indices = self.selected_listbox.curselection()
        for index in reversed(selected_indices):
            self.selected_listbox.delete(index)
    
    def remove_all_pairs(self):
        """Entfernt alle Pairs aus der Auswahlliste"""
        self.selected_listbox.delete(0, tk.END)
    
    def get_selected_pairs(self):
        """Gibt alle ausgewählten Pairs zurück"""
        return list(self.selected_listbox.get(0, tk.END))
    
    def save_trading_pairs(self):
        """Speichert die ausgewählten Trading-Pairs"""
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
            self.root.after(0, self._update_after_backtest_simple, results)
        
        threading.Thread(target=run_backtest, daemon=True).start()

    def _update_after_backtest_simple(self, results):
        """Aktualisiert die GUI nach Backtest-Abschluss (vereinfachte Version)"""
        try:
            if results:
                self.update_recommendations()
                self.update_status(f"Backtest abgeschlossen - {len(results)} Kryptos analysiert")
            else:
                self.update_status("Backtest fehlgeschlagen - Keine Ergebnisse")
        except Exception as e:
            self.update_status(f"Update Fehler: {str(e)}")

    # Trading-Methoden
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
                
            if hasattr(self, 'update_active_trades'):
                self.update_active_trades()
            if hasattr(self, 'update_trade_history'):
                self.update_trade_history()
            if hasattr(self, 'update_tax_log'):
                self.update_tax_log()
                
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
        
        if hasattr(self, 'update_active_trades'):
            self.update_active_trades()
        if hasattr(self, 'update_trade_history'):
            self.update_trade_history()
        if hasattr(self, 'update_tax_log'):
            self.update_tax_log()

    def update_active_trades(self):
        """Aktualisiert die Anzeige der aktiven Trades"""
        if not hasattr(self, 'trades_tree'):
            return
            
        for item in self.trades_tree.get_children():
            self.trades_tree.delete(item)
            
        for symbol, trade in self.bot.active_trades.items():
            current_price = self.bot.get_current_price(symbol)
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
                
        if hasattr(self, 'trades_tree'):
            self.trades_tree.tag_configure('profit', background='#d4edda')
            self.trades_tree.tag_configure('loss', background='#f8d7da')
        
    def update_trade_history(self):
        """Aktualisiert die Trade-Historie"""
        if not hasattr(self, 'history_tree'):
            return
            
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
            
        recent_trades = self.bot.tax_logger.get_recent_trades(50)
        
        for trade in recent_trades:
            pl_percent = trade.get('profit_loss_percent', 0)
            pl_amount = trade.get('profit_loss', 0)
            
            tags = ()
            if pl_amount > 0:
                tags = ('profit',)
            elif pl_amount < 0:
                tags = ('loss',)
            else:
                tags = ('neutral',)
                
            self.history_tree.insert('', tk.END, values=(
                trade['timestamp'],
                trade['symbol'],
                trade['side'],
                f"${trade['price']:.6f}",
                f"{trade['amount']:.6f}",
                f"{pl_percent:+.2f}%",
                f"${pl_amount:+.2f}",
                trade['reason']
            ), tags=tags)
            
        if hasattr(self, 'history_tree'):
            self.history_tree.tag_configure('profit', background='#d4edda')
            self.history_tree.tag_configure('loss', background='#f8d7da')
            self.history_tree.tag_configure('neutral', background='#fff3cd')

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

    # Weitere Methoden
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
    
    def debug_status(self):
        """Zeigt Debug-Informationen an"""
        print(f"Auto-Trading: {self.bot.auto_trading}")
        print(f"Aktive Trades: {len(self.bot.active_trades)}")
        print(f"Trading-Pairs: {self.bot.trading_pairs}")
        print(f"Empfehlungen: {len(self.bot.current_recommendations)}")
        print(f"Letztes Update: {self.bot.last_update}")
        print(f"Schnelle Signale: {self.bot.use_quick_signals}")
        
        if self.bot.current_recommendations:
            for crypto, data in self.bot.current_recommendations.items():
                print(f"  {crypto}: {data['current_signal']} ({data['confidence']}%)")

    # Status und Aktivitäts-Methoden
    def update_bot_status(self):
        """Aktualisiert den Bot-Status"""
        try:
            # Basis Status
            self.bot_status_vars['auto_trading'].set(
                "AKTIV" if self.bot.auto_trading else "INAKTIV"
            )
            self.bot_status_vars['active_trades'].set(
                f"{len(self.bot.active_trades)} / {self.bot.max_open_trades}"
            )
            self.bot_status_vars['trading_pairs'].set(
                f"{len(self.bot.trading_pairs)} Pairs"
            )
            
            # Signal-Modus
            if hasattr(self.bot, 'use_quick_signals'):
                self.bot_status_vars['signal_mode'].set(
                    "Schnell-Modus" if self.bot.use_quick_signals else "Backtest-Modus"
                )
            
            # Zeit Informationen
            if self.bot.last_update:
                self.bot_status_vars['last_analysis'].set(
                    self.bot.last_update.strftime('%H:%M:%S')
                )
            else:
                self.bot_status_vars['last_analysis'].set("Noch keine")
                
            if self.bot.next_scheduled_update:
                time_diff = self.bot.next_scheduled_update - datetime.now()
                minutes = max(0, int(time_diff.total_seconds() / 60))
                self.bot_status_vars['next_analysis'].set(
                    f"in {minutes} min"
                )
            else:
                self.bot_status_vars['next_analysis'].set("-")
            
            # Trade Informationen
            if self.bot.last_trade_time:
                self.bot_status_vars['last_trade'].set(
                    self.bot.last_trade_time.strftime('%H:%M:%S')
                )
            else:
                self.bot_status_vars['last_trade'].set("Noch keine")
            
            # API Statistiken
            api_stats = self.bot.api.get_api_stats()
            if api_stats:
                self.bot_status_vars['total_requests'].set(str(api_stats['request_count']))
                if api_stats['last_request_time']:
                    self.bot_status_vars['last_request'].set(
                        api_stats['last_request_time'].strftime('%H:%M:%S')
                    )
                else:
                    self.bot_status_vars['last_request'].set("-")
                    
        except Exception as e:
            print(f"Status update error: {e}")
            
        # Nächste Aktualisierung in 5 Sekunden
        self.root.after(5000, self.update_bot_status)
    
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
        print(f"Activity: {message}")
    
    def quick_signal_check(self):
        """Startet einen schnellen Signal-Check"""
        def run_quick_check():
            self.bot.quick_signal_check()
        
        threading.Thread(target=run_quick_check, daemon=True).start()
        self.update_status("Schnelle Signalprüfung gestartet...")
    
    def force_cache_update(self):
        """Erzwingt eine Cache-Aktualisierung"""
        def update():
            self.bot.update_caches()
            self.update_balance_display()
            
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
                        f"Gesamtportfolio: ${total_value:,.2f} (Stand: {last_updated})"
                    )
                    
                    # Detaillierte Bestände
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
                    self.balance_info_var.set("Keine Kontostandsdaten verfügbar")
                    
            except Exception as e:
                self.balance_info_var.set(f"Fehler: {str(e)}")
                print(f"Balance update error: {e}")
        
        threading.Thread(target=update, daemon=True).start()

    def update_recommendations(self):
        """Aktualisiert die Trading-Empfehlungen"""
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
                
        if hasattr(self, 'rec_tree'):
            self.rec_tree.tag_configure('buy', background='#d4edda')
            self.rec_tree.tag_configure('sell', background='#f8d7da')
            self.rec_tree.tag_configure('hold', background='#fff3cd')

    def toggle_auto_trading(self):
        """Schaltet Auto-Trading um - mit Bestätigung"""
        new_state = self.auto_trading_var.get()
        
        if new_state:
            # Warnung bei Aktivierung
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
        
        if self.bot.auto_trading:
            messagebox.showinfo("Auto-Trading", 
                              "Auto-Trading ist jetzt AKTIV\n\n"
                              "Der Bot wird automatisch Trades basierend auf den Signalen ausführen.")
        else:
            messagebox.showinfo("Auto-Trading", 
                              "Auto-Trading ist jetzt INAKTIV\n\n"
                              "Es werden keine automatischen Trades mehr ausgeführt.")
        
    def save_settings(self):
        """Speichert die Einstellungen"""
        try:
            stop_loss = float(self.stop_loss_var.get())
            trade_size = float(self.trade_size_var.get())
            rsi_oversold = float(self.rsi_oversold_var.get())
            rsi_overbought = float(self.rsi_overbought_var.get())
            interval = self.interval_var.get()
            
            self.bot.set_trading_settings(
                stop_loss=stop_loss,
                trade_size=trade_size,
                rsi_oversold=rsi_oversold,
                rsi_overbought=rsi_overbought
            )
            self.bot.set_interval(interval)
            
            self.update_status("Einstellungen gespeichert")
            messagebox.showinfo("Erfolg", "Einstellungen wurden gespeichert!")
            
        except ValueError:
            messagebox.showerror("Fehler", "Bitte gültige Zahlen eingeben!")
            
    def start_backtest(self):
        """Startet einen Backtest"""
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
                self.root.after(0, lambda: self._update_after_backtest_complete(results, was_auto_trading))
                
            except Exception as e:
                error_msg = f"Backtest Fehler: {str(e)}"
                print(error_msg)
                self.root.after(0, lambda: self.update_status(error_msg))
        
        threading.Thread(target=run_backtest, daemon=True).start()

    def _update_after_backtest_complete(self, results, was_auto_trading):
        """Aktualisiert die GUI nach Backtest-Abschluss (komplette Version)"""
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
                    if hasattr(self, 'update_active_trades'):
                        self.root.after(0, self.update_active_trades)
                    if hasattr(self, 'update_trade_history'):
                        self.root.after(0, self.update_trade_history)
                    if hasattr(self, 'update_tax_log'):
                        self.root.after(0, self.update_tax_log)
                except Exception as e:
                    print(f"Auto-update error: {e}")
                    
                time.sleep(30)
                
        threading.Thread(target=update_loop, daemon=True).start()
        
    def run(self):
        """Startet die GUI-Hauptschleife"""
        self.root.mainloop()