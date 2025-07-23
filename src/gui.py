import tkinter as tk
import os
from typing import List, Dict, Any

class ParameterForm:
    def __init__(self, parent: tk.Tk, file_paths: List[str]):
        self.parent = parent
        self.parent.title("Anisotropy Run Parameters")
        self.file_paths = file_paths
        self.result = None

        frame = tk.Frame(parent, padx=15, pady=15)
        frame.pack()

        tk.Label(frame, text="[ Global Parameters ]", font=('Arial', 10, 'bold')).grid(row=0, columnspan=2, pady=5)
        self.background_entry = self._add_labeled_entry(frame, "Background Level:", 1000.0, 1)
        self.half_window_entry = self._add_labeled_entry(frame, "Half-Window (points):", 12, 2)
        
        self.correction_mode = tk.StringVar(value='vector')
        tk.Label(frame, text="G-Factor Correction Mode:").grid(row=3, column=0, sticky="e", padx=5)
        tk.OptionMenu(frame, self.correction_mode, 'scalar', 'vector').grid(row=3, column=1, sticky="w", padx=5)
        
        tk.Label(frame, text="[ Peak Emission λ₀ (nm) ]", font=('Arial', 10, 'bold')).grid(row=4, columnspan=2, pady=10)
        self.lambda_entries = {}
        unique_labels = sorted(list(set([os.path.basename(f).split('_')[0] for f in self.file_paths])))
        
        for i, label in enumerate(unique_labels):
            default_lambda0 = 583.0 if "dye" in label.lower() else 583.0
            self.lambda_entries[label] = self._add_labeled_entry(frame, f"{label}:", default_lambda0, 5 + i)

        submit_btn = tk.Button(frame, text="Run Analysis", command=self.submit, bg="#4CAF50", fg="white")
        submit_btn.grid(row=6 + len(unique_labels), columnspan=2, pady=15)

    def _add_labeled_entry(self, parent, label_text: str, default_value, row: int) -> tk.Entry:
        tk.Label(parent, text=label_text).grid(row=row, column=0, sticky="e", padx=5, pady=2)
        entry = tk.Entry(parent)
        entry.insert(0, str(default_value))
        entry.grid(row=row, column=1, padx=5, pady=2)
        return entry

    def submit(self):
        lambda_0_dict = {}
        for f in self.file_paths:
            label = os.path.basename(f).split('_')[0]
            lambda_0_dict[f] = float(self.lambda_entries[label].get())
            
        self.result = {
            "background": float(self.background_entry.get()),
            "half_window_pts": int(self.half_window_entry.get()),
            "correction_mode": self.correction_mode.get(),
            "lambda_0_dict": lambda_0_dict,
        }
        self.parent.destroy()

def launch_parameter_form(file_paths: List[str]) -> Dict[str, Any]:
    root = tk.Tk()
    app = ParameterForm(root, file_paths)
    root.mainloop()
    return app.result