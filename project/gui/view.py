import tkinter as tk
from tkinter import ttk

import PIL
import cv2
from PIL import ImageTk


class AppView(tk.Tk):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.title("EAN-13 Barcode Scanner")
        self.geometry("1920x1080")
        self.configure(bg="#2b2b2b")

        self._setup_styles()
        self._build_ui()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#2b2b2b")
        style.configure("TLabel", background="#2b2b2b", foreground="white")
        style.configure("TButton", background="#4a4a4a", foreground="white", borderwidth=1)
        style.map("TButton", background=[("active", "#666666")])
        style.configure("TNotebook", background="#2b2b2b", borderwidth=0)
        style.configure("TNotebook.Tab", background="#333333", foreground="white", padding=[10, 2])
        style.map("TNotebook.Tab", background=[("selected", "#555555")])

    def _build_ui(self):
        # 1. Sidebar
        self.sidebar = ttk.Frame(self, width=250)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        lbl_title = ttk.Label(self.sidebar, text="Input Settings", font=("Arial", 12, "bold"))
        lbl_title.pack(pady=(0, 10), anchor="w")

        self.btn_load = ttk.Button(self.sidebar, text="Load Image", command=self.controller.handle_load)
        self.btn_load.pack(fill=tk.X, pady=5)

        ttk.Separator(self.sidebar, orient="horizontal").pack(fill=tk.X, pady=15)

        self.var_verbose = tk.BooleanVar(value=True)
        self.chk_verbose = tk.Checkbutton(self.sidebar, text="Verbose Mode", variable=self.var_verbose,
                                          bg="#2b2b2b", fg="white", selectcolor="#4a4a4a", activebackground="#2b2b2b")
        self.chk_verbose.pack(anchor="w")

        self.btn_ocr = ttk.Button(self.sidebar, text="Run Barcode Detector", command=self.controller.handle_ocr)
        self.btn_ocr.pack(fill=tk.X, pady=20)

        self.btn_clear_logs = ttk.Button(self.sidebar, text="Clear Logs", command=self.controller.handle_clear_logs)
        self.btn_clear_logs.pack(fill=tk.X)

        # 2. Main content (tabs and logs)
        self.content = ttk.Frame(self)
        self.content.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tabs for images
        self.notebook = ttk.Notebook(self.content)
        self.notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Create default tab
        self.tabs = {}
        self._create_tab("Original")

        # Log console
        self.log_frame = ttk.Frame(self.content, height=200)
        self.log_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))

        self.log_text = tk.Text(self.log_frame, height=10, bg="#1e1e1e", fg="#00ff00",
                                font=("Consolas", 10), state="disabled")
        self.scrollbar = ttk.Scrollbar(self.log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def clear_tabs(self):
        for tab_id in self.notebook.tabs():
            self.notebook.forget(tab_id)

        self.tabs = {}

        self._create_tab("Original")

    def _create_tab(self, name):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=name)

        # Image canvas
        canvas = tk.Canvas(frame, bg="#1e1e1e", highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)

        self.tabs[name] = {"frame": frame, "canvas": canvas, "image_ref": None}

    def clear_logs(self):
        self.log_text.config(state="normal")
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state="disabled")

    def update_logs(self, message):
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")

    def update_images(self, processed_images):
        for name, img_data in processed_images.items():
            if name not in self.tabs:
                self._create_tab(name)

            # Convert BGR to RGB
            if len(img_data.shape) == 2:
                color_converted = cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB)
            else:
                color_converted = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)

            pil_image = PIL.Image.fromarray(color_converted)

            # Resize for display (simple aspect ratio fit)
            canvas = self.tabs[name]["canvas"]
            c_width = canvas.winfo_width()
            c_height = canvas.winfo_height()

            # Initial size check (if window not fully rendered yet)
            if c_width < 100:
                c_width = 800
            if c_height < 100:
                c_height = 600

            # Compute aspect ratio
            img_w, img_h = pil_image.size
            ratio = min(c_width / img_w, c_height / img_h)
            new_size = (int(img_w * ratio), int(img_h * ratio))

            pil_image = pil_image.resize(new_size, PIL.Image.Resampling.LANCZOS)
            tk_image = ImageTk.PhotoImage(pil_image)

            # Update Canvas
            canvas.delete("all")

            # Center image
            x_pos = c_width // 2
            y_pos = c_height // 2
            canvas.create_image(x_pos, y_pos, anchor=tk.CENTER, image=tk_image)

            # Keep reference to prevent garbage collection
            self.tabs[name]["image_ref"] = tk_image
