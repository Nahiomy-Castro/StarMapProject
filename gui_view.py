import tkinter as tk

class MainView(tk.Tk):
    def __init__(self):
        super().__init__()

        self.controller = None   # controller will be attached later

        self.title("StarMapProject")
        self.geometry("900x600")

        tk.Label(self, text="Star Map Project", font=("Arial", 20)).pack(pady=20)

        # Buttons WITHOUT controller yet
        self.train_btn = tk.Button(self, text="Train ML Model", width=20, height=2)
        self.train_btn.pack(pady=10)

        self.graph_btn = tk.Button(self, text="View Graph", width=20, height=2)
        self.graph_btn.pack(pady=10)

        tk.Button(self, text="Exit", command=self.quit, width=20, height=2).pack(pady=10)
