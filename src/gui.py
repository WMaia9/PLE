# src/gui.py

import tkinter as tk
import os
from typing import List, Dict, Any

class ParameterForm:
    def __init__(self, parent: tk.Tk, file_paths: List[str]):
        self.form_window = tk.Toplevel(parent)
        self.form_window.title("Analysis Parameters")
        
        self.file_paths = file_paths
        self.result = None

        frame = tk.Frame(self.form_window, padx=15, pady=15)
        frame.pack()

        # --- Labels updated to match the image ---
        tk.Label(frame, text="[ Global Parameters ]", font=('Arial', 10, 'bold')).grid(row=0, columnspan=2, pady=5)
        self.background_entry = self._add_labeled_entry(frame, "Background:", 990.0, 1)
        
        # --- Changed from Half-Window to Window Size ---
        self.window_size_entry = self._add_labeled_entry(frame, "Window Size (points):", 15, 2)
        
        tk.Label(frame, text="[ lambda_0 Values Per File ]", font=('Arial', 10, 'bold')).grid(row=3, columnspan=2, pady=10)
        self.lambda_entries = {}
        unique_labels = sorted(list(set([_clean_label(f) for f in self.file_paths])))
        
        for i, label in enumerate(unique_labels):
            default_lambda0 = 583.0
            self.lambda_entries[label] = self._add_labeled_entry(frame, f"{label}:", default_lambda0, 4 + i)

        submit_btn = tk.Button(frame, text="Run Analysis", command=self.submit, bg="#4CAF50", fg="white")
        submit_btn.grid(row=5 + len(unique_labels), columnspan=2, pady=15)

    def _add_labeled_entry(self, parent, label_text: str, default_value, row: int) -> tk.Entry:
        tk.Label(parent, text=label_text).grid(row=row, column=0, sticky="e", padx=5, pady=2)
        entry = tk.Entry(parent)
        entry.insert(0, str(default_value))
        entry.grid(row=row, column=1, padx=5, pady=2)
        return entry

    def submit(self):
        lambda_0_dict = {}
        for f in self.file_paths:
            label = _clean_label(f)
            lambda_0_dict[f] = float(self.lambda_entries[label].get())
            
        self.result = {
            "background": float(self.background_entry.get()),
            # --- Pass the full window size ---
            "window_size": int(self.window_size_entry.get()),
            "lambda_0_dict": lambda_0_dict,
        }
        self.form_window.destroy()

def _clean_label(p: str) -> str:
    return os.path.basename(p).lower().replace("perp", "").replace("par", "").replace("_", "").strip()

def launch_parameter_form(parent: tk.Tk, file_paths: List[str]) -> Dict[str, Any]:
    app = ParameterForm(parent, file_paths)
    parent.wait_window(app.form_window)
    return app.result