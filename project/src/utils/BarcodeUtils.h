#ifndef PRSLAB2_BARCODEUTILS_H
#define PRSLAB2_BARCODEUTILS_H

#include <opencv2/opencv.hpp>

namespace BarcodeUtils {
    /**
     * Step 1: Prepares the image for stable edge detection
     * Converts to grayscale, applies CLAHE, and smooths noise
     */
    void preprocess_image(const cv::Mat& input, cv::Mat& output);

    /**
     * Step 2: Detects vertical edges to highlight barcode zones
     * Uses Sobel filters, thresholding, and morphological closing
     */
    void detect_edges(const cv::Mat& input, cv::Mat& output);

    /**
     * Step 4: Find the barcode location using contours.
     * Uses findContours and filters by area/aspect ratio.
     */
    cv::RotatedRect get_barcode_region(const cv::Mat& edgeMask);

    /**
     * Step 4: Cut out and flatten the barcode from the original image
     * Uses warpPerspective to handle rotation and skew
     */
    cv::Mat extract_barcode(const cv::Mat& original, const cv::RotatedRect& rect);
}

#endif