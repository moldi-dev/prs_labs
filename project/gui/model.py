import datetime
import os
from zoneinfo import ZoneInfo

import numpy as np

from gui.ml_utils import DigitPredictor, segment_digits, validate_ean13
import cv2


class AppModel:
    def __init__(self):
        self.image_path = None
        self.original_image = None
        self.processed_stages = {}
        self.logs = []
        self.verbose = True
        self.predictor = DigitPredictor()

        # Observers for MVC updates
        self._log_observers = []
        self._image_observers = []

    def load_image(self, path):
        self.log(f"[IO] Loading: {path}")
        if not os.path.exists(path):
            self.log("[Error] File not found.")
            return

        self.image_path = path
        self.original_image = cv2.imread(path)

        if self.original_image is None:
            self.log("[Error] Failed to decode image.")
        else:
            h, w = self.original_image.shape[:2]
            self.log(f"[IO] Loaded {w}x{h}")
            self.processed_stages = {"Original": self.original_image}
            self.notify_image_update()

    def run_preprocessing(self):
        if self.original_image is None:
            self.log("[Error] No image loaded.")
            return

        self.log("[Step 1] Starting Preprocessing...")

        # 1. Grayscale
        if len(self.original_image.shape) == 3:
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.original_image.copy()

        # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)

        # 3. Gaussian Blur
        blurred = cv2.GaussianBlur(clahe_img, (5, 5), 0)

        self.processed_stages["Preprocessed"] = blurred

        if self.verbose:
            self.log("[Debug] Preprocessing complete (CLAHE + Blur).")

        return blurred

    def detect_regions(self, preprocessed_img):
        self.log("[Step 2] Detecting Edges...")

        # 1. Sobel gradients
        grad_x = cv2.Sobel(preprocessed_img, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(preprocessed_img, cv2.CV_16S, 0, 1, ksize=3)

        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        # 2. Combine gradients
        gradient = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        # 3. Threshold (Otsu)
        _, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # 4. Morphological closing
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 11))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)

        # 5. Erosion and dilation
        kernel_noise = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        eroded = cv2.erode(closed, kernel_noise, iterations=4)
        dilated = cv2.dilate(eroded, kernel_noise, iterations=4)

        self.processed_stages["Edges"] = dilated
        self.notify_image_update()

        if self.verbose:
            self.log("[Debug] Edge detection complete.")

        return dilated

    def extract_barcode_crop(self, edge_mask):
        contours, _ = cv2.findContours(edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

        best_rect = None
        max_area = 0
        for c in contours:
            rect = cv2.minAreaRect(c)
            (center, (w, h), angle) = rect
            area = w * h
            if area > 500 and area > max_area:
                max_area = area
                best_rect = rect

        if best_rect is None:
            return None, None

        box = cv2.boxPoints(best_rect)
        box = np.int32(box)
        width = int(best_rect[1][0])
        height = int(best_rect[1][1])
        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(self.original_image, M, (width, height))

        if height > width:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

        return warped, box

    def run_ocr_pipeline(self):
        if self.original_image is None:
            self.log("[Error] Please load an image first.")
            return

        self.log("=== Starting CNN Pipeline ===")

        # 1. Processing
        pre = self.run_preprocessing()
        edges = self.detect_regions(pre)

        # 2. Extract Barcode
        self.log("[Step 3] Extracting Barcode Region...")
        crop, box_points = self.extract_barcode_crop(edges)

        if crop is None:
            self.log("[Error] No barcode region found.")
            return

        self.processed_stages["Barcode Crop"] = crop

        # 3. Segment Digits
        self.log("[Step 4] Segmenting Digits...")
        digit_crops, debug_thresh = segment_digits(crop)

        if len(digit_crops) == 0:
            self.log("[Error] No digits found in bottom strip.")
            self.processed_stages["Segmented Digits"] = debug_thresh
            self.notify_image_update()
            return

        self.processed_stages["Segmented Digits"] = debug_thresh

        # 4. Predict
        self.log(f"[Step 5] Predicting on {len(digit_crops)} segments...")
        result_string = ""

        for i, digit_img in enumerate(digit_crops):
            prediction = self.predictor.predict(digit_img)
            result_string += prediction
            self.processed_stages[f"Digit_{i + 1}"] = digit_img

        self.log(f"[Result] Model Output: {result_string}")

        # 5. Validation Logic
        status_text = "INVALID"
        color = (0, 0, 255)  # Red

        if len(result_string) == 13:
            is_valid, correct = validate_ean13(result_string)
            if is_valid:
                self.log(f"[Success] Valid EAN-13: {result_string}")
                status_text = f"EAN-13: {result_string} (OK)"
                color = (0, 255, 0)
            else:
                self.log(f"[Warning] Invalid Checksum. Expected end: {correct}")
                status_text = f"{result_string} (Bad Checksum)"
                color = (0, 165, 255)
        else:
            self.log(f"[Error] Length mismatch ({len(result_string)} digits)")
            status_text = f"Partial: {result_string}"

        # 6. Visualization
        vis_image = self.original_image.copy()
        if 'box_points' in locals() and box_points is not None:
            cv2.drawContours(vis_image, [box_points], 0, color, 3)

        cv2.rectangle(vis_image, (0, 0), (800, 80), (0, 0, 0), -1)
        cv2.putText(vis_image, status_text, (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        self.processed_stages["Final Result"] = vis_image
        self.notify_image_update()

    def clear_logs(self):
        self.logs = []

    def log(self, message):
        bucharest_tz = ZoneInfo("Europe/Bucharest")
        timestamp = datetime.datetime.now(bucharest_tz).strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        self.logs.append(formatted)
        for callback in self._log_observers:
            callback(formatted)

    def add_log_observer(self, callback):
        self._log_observers.append(callback)

    def add_image_observer(self, callback):
        self._image_observers.append(callback)

    def notify_image_update(self):
        for callback in self._image_observers:
            callback(self.processed_stages)
