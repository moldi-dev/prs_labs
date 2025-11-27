#include "BarcodeUtils.h"

namespace BarcodeUtils {
    void preprocess_image(const cv::Mat& input, cv::Mat& output) {
        // 1. Convert to grayscale
        cv::Mat gray;

        if (input.channels() == 3) {
            cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
        }

        else {
            gray = input.clone();
        }

        // 2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        // Clip limit of 2.0 and tile grid size of 8 * 8 are standard starting points
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        cv::Mat claheImg;
        clahe->apply(gray, claheImg);

        // 3. Apply smoothing to reduce random noise
        // Using a Gaussian Blur with a 5 * 5 kernel
        cv::GaussianBlur(claheImg, output, cv::Size(5, 5), 0);
    }

    void detect_edges(const cv::Mat& input, cv::Mat& output) {
        // 1. Use Sobel filters to find vertical edges
        cv::Mat gradX;
        // ddepth = CV_16S to avoid overflow, dx = 1, dy = 0 for vertical edges
        cv::Sobel(input, gradX, CV_16S, 1, 0, 3);

        cv::Mat absGradX;
        cv::convertScaleAbs(gradX, absGradX);

        // 2. Threshold to keep only strong vertical transitions
        // Using Otsu's binarization to automatically find the optimal threshold
        cv::Mat thresh;
        cv::threshold(absGradX, thresh, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        // 3. Morphological Operations
        // Use a flatter kernel (21 * 3) to connect bars horizontally
        // without grabbing the numbers below
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21, 3));
        cv::morphologyEx(thresh, output, cv::MORPH_CLOSE, kernel);

        // Perform standard Erosion (3 * 3) to remove noise and disconnect
        // the barcode from the text labels if they are barely touching
        // This helps isolate the barcode blob
        cv::Mat erodeKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::erode(output, output, erodeKernel, cv::Point(-1, -1), 4);

        // Dilate slightly to restore the volume of the main barcode block
        // after the erosion
        cv::Mat dilateKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::dilate(output, output, dilateKernel, cv::Point(-1, -1), 4);
    }

    cv::RotatedRect get_barcode_region(const cv::Mat& edgeMask) {
        // 1. Find the contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(edgeMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::RotatedRect bestRect;
        double maxArea = 0;

        for (const auto& c : contours) {
            // Use cv::minAreaRect to compute bounding box
            cv::RotatedRect rect = cv::minAreaRect(c);

            float width = rect.size.width;
            float height = rect.size.height;

            // Handle rotation (ensure width is the long side for calculation)
            if (width < height) {
                std::swap(width, height);
            }

            // Filter candidates by aspect ratio
            // EAN-13 is typically wide => we expect width > height
            double area = width * height;
            double aspectRatio = width / height;

            // Simple heuristic filters:
            // - Area must be significant
            // - Aspect ratio is usually > 2.0 for barcodes
            if (area > maxArea && aspectRatio > 2.0) {
                maxArea = area;
                bestRect = rect;
            }
        }

        return bestRect;
    }

    cv::Mat extract_barcode(const cv::Mat& original, const cv::RotatedRect& rect) {
        if (rect.size.area() < 10) {
            return cv::Mat(); // Return empty if no barcode found
        }

        // Apply warp Perspective to correct skew

        // 1. Determine target size (flattened)
        // We ensure the output is always horizontal (width > height)
        float width = rect.size.width;
        float height = rect.size.height;

        // If the rect is vertical (angle near 90), swap width/height to make output horizontal
        if (rect.angle < -45.0) {
            std::swap(width, height);
        }

        // 2. Get the 4 corners of the rotated rect
        cv::Point2f rectPoints[4];
        rect.points(rectPoints);

        // 3. Sort points to correspond to top-left, top-right, bottom-right, bottom-left
        // This is a simplified sort for standard rotations;
        std::vector<cv::Point2f> srcPts(4);

        // Destination points (straight rectangle)
        std::vector<cv::Point2f> dstPts = {
            {0, height - 1},
            {0, 0},
            {width - 1, 0},
            {width - 1, height - 1}
        };

        for (int i = 0; i < 4; i++) {
            srcPts[i] = rectPoints[i];
        }

        // 4. Compute Perspective Matrix and Warp
        cv::Mat M = cv::getPerspectiveTransform(srcPts, dstPts);
        cv::Mat warped;
        cv::warpPerspective(original, warped, M, cv::Size(width, height));

        return warped;
    }
}