from tkinter import filedialog

from gui.model import AppModel
from gui.view import AppView


class AppController:
    def __init__(self):
        self.model = AppModel()
        self.view = AppView(self)

        # Bind model to view updates (Observer pattern)
        self.model.add_log_observer(self.view.update_logs)
        self.model.add_image_observer(self.view.update_images)

    def start(self):
        self.view.mainloop()

    def handle_load(self):
        path = filedialog.askopenfilename(
            title="Select Barcode Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
        )
        if path:
            self.view.clear_tabs()
            self.model.load_image(path)

    def handle_ocr(self):
        self.model.verbose = self.view.var_verbose.get()
        self.model.run_ocr_pipeline()

    def handle_clear_logs(self):
        self.view.clear_logs()
        self.model.clear_logs()
