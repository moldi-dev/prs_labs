#ifndef BARCODE_UTILS_H
#define BARCODE_UTILS_H

#include <opencv2/opencv.hpp>

class BarcodeDetector {
public:
    explicit BarcodeDetector(bool verbose = false);
    cv::Mat scan(const cv::Mat& input);

private:
    bool verbose;

    void pre_process_image(const cv::Mat& input, cv::Mat& output);
    void detect_edges(const cv::Mat& input, cv::Mat& output);
    cv::RotatedRect get_barcode_region(const cv::Mat& edgeMask);
    cv::Mat extract_barcode(const cv::Mat& original, const cv::RotatedRect& rect);
};

#endif