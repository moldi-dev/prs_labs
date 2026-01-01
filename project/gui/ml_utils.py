import cv2
import numpy as np
import tensorflow as tf
import os


class DigitPredictor:
    def __init__(self, model_path='ean13_digit_model.keras'):
        self.model = None

        if os.path.exists(model_path):
            try:
                self.model = tf.keras.models.load_model(model_path)
                print(f"[ML] Model loaded successfully from {model_path}")
            except Exception as e:
                print(f"[ML] Error loading model: {e}")
        else:
            print(f"[ML] Warning: Model file not found at {model_path}")

    def predict(self, img_crop):
        if self.model is None or img_crop is None:
            return "?"

        try:
            # 1. Grayscale
            if len(img_crop.shape) == 3:
                gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_crop.copy()

            # 2. Add some padding
            # Neural Networks trained on MNIST-style data expect the digit
            # to be centered with black space around it
            rows, cols = gray.shape

            # Make it square
            if rows > cols:
                diff = rows - cols
                pad_l = diff // 2
                pad_r = diff - pad_l
                gray = cv2.copyMakeBorder(gray, 0, 0, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
            else:
                diff = cols - rows
                pad_t = diff // 2
                pad_b = diff - pad_t
                gray = cv2.copyMakeBorder(gray, pad_t, pad_b, 0, 0, cv2.BORDER_CONSTANT, value=0)

            # Add extra uniform border (4 pixels)
            gray = cv2.copyMakeBorder(gray, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)

            # 3. Resize to 28x28
            resized = cv2.resize(gray, (28, 28))

            # 4. Normalize and reshape
            norm = resized.astype('float32') / 255.0
            input_tensor = np.expand_dims(norm, axis=-1)
            input_tensor = np.expand_dims(input_tensor, axis=0)

            # 5. Predict
            prediction = self.model.predict(input_tensor, verbose=0)
            digit_class = np.argmax(prediction)
            confidence = np.max(prediction)

            # Filter low confidence guesses
            if confidence < 0.5:
                return "?"

            return str(digit_class)
        except Exception as e:
            print(f"Prediction Error: {e}")
            return "E"


def segment_digits(barcode_crop):
    if barcode_crop is None:
        return [], None

    h, w = barcode_crop.shape[:2]

    # 1. Crop bottom 30% (to catch tall digits)
    bottom_strip = barcode_crop[int(h * 0.70):h, 0:w]

    # 2. Grayscale
    gray = cv2.cvtColor(bottom_strip, cv2.COLOR_BGR2GRAY)

    # 3. Inverse threshold (text is white, background is black) using Otsu
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Compute how much white is in the image
    # If more than 15%, the text is likely bold or bleeding
    white_ratio = np.count_nonzero(thresh) / thresh.size

    # Perform an erosion for bleeding text to thin it
    if white_ratio > 0.15:
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.erode(thresh, kernel, iterations=1)

    # 4. Find the contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digit_candidates = []

    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)

        # Height check: digits should be reasonably tall (>20% of strip height)
        if ch < 0.2 * bottom_strip.shape[0]:
            continue

        # Aspect ratio check: digits are usually blocky and guard bars (||) are very thin
        aspect_ratio = cw / float(ch)

        # Filter thin lines (bars) and wide blobs (merged digits)
        if aspect_ratio < 0.30:
            continue

        if aspect_ratio > 1.2:
            continue

        digit_candidates.append((x, y, cw, ch))

    # Sort left to right
    digit_candidates.sort(key=lambda b: b[0])

    # Extract the images using the thresholded version for cleaner input
    digit_crops = []
    for (x, y, cw, ch) in digit_candidates:
        roi = thresh[y:y + ch, x:x + cw]
        digit_crops.append(roi)

    return digit_crops, thresh


def validate_ean13(code_str):
    if len(code_str) != 13 or not code_str.isdigit():
        return False, -1

    digits = [int(d) for d in code_str]
    check_digit = digits[-1]
    data_digits = digits[:-1]

    total_sum = 0
    for i, d in enumerate(data_digits):
        # Weight: 1 for even indices (0, 2...), 3 for odd indices (1, 3...)
        weight = 1 if i % 2 == 0 else 3
        total_sum += d * weight

    nearest_ten = (total_sum + 9) // 10 * 10
    calculated_check = nearest_ten - total_sum

    return check_digit == calculated_check, calculated_check