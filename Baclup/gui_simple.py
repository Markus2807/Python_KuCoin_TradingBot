# gui_simple.py - Ultra-kompakte Version für Raspberry Pi
import tkinter as tk
from tkinter import ttk, messagebox
import threading
from datetime import datetime

class SimpleTradingBotGUI:
    def __init__(self, bot):
        self.bot = bot
        self.bot.set_gui_reference(self)
        
        self.root = tk.Tk()
        self.root.title("Trading Bot")
        self.root.geometry("780x460")
        
        self.setup_simple_gui()
        
    def setup_simple_gui(self):
        # Haupt-Frame
        main_frame = ttk.Frame(self.root, padding=5)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Status
        self.status_var = tk.StringVar(value="Bereit")
        ttk.Label(main_frame, textvariable=self.status_var, font=('Arial', 10, 'bold')).pack(pady=5)
        
        # Balance
        self.balance_var = tk.StringVar(value="Lade Kontostand...")
        ttk.Label(main_frame, textvariable=self.balance_var).pack(pady=2)
        
        # Auto-Trading
        self.auto_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(main_frame, text="Auto-Trading", variable=self.auto_var, 
                       command=self.toggle_auto).pack(pady=2)
        
        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="Backtest", command=self.start_backtest, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Kontostand", command=self.update_balance, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Signale", command=self.quick_check, width=12).pack(side=tk.LEFT, padx=2)
        
        # Empfehlungen
        ttk.Label(main_frame, text="Empfehlungen:", font=('Arial', 9, 'bold')).pack(pady=(20,5))
        
        tree_frame = ttk.Frame(main_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ('Symbol', 'Signal', 'Confidence')
        self.tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)
            
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # Initial
        self.update_balance()
        
    def toggle_auto(self):
        self.bot.auto_trading = self.auto_var.get()
        self.status_var.set("Auto-Trading: " + ("AKTIV" if self.bot.auto_trading else "INAKTIV"))
        
    def update_balance(self):
        def update():
            balance = self.bot.get_balance_summary()
            if balance:
                self.balance_var.set(f"Portfolio: ${balance['total_portfolio_value']:,.2f}")
        threading.Thread(target=update, daemon=True).start()
        
    def quick_check(self):
        def check():
            results = self.bot.quick_signal_check()
            self.root.after(0, self.update_tree, results)
        threading.Thread(target=check, daemon=True).start()
        
    def start_backtest(self):
        def backtest():
            results = self.bot.run_complete_backtest()
            self.root.after(0, self.update_tree, results)
        threading.Thread(target=backtest, daemon=True).start()
        
    def update_tree(self, results):
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        for crypto, data in results.items():
            self.tree.insert('', tk.END, values=(
                crypto, data['current_signal'], f"{data['confidence']}%"
            ))
            
    def run(self):
        self.root.mainloop()