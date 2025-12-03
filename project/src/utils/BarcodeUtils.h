#ifndef BARCODE_UTILS_H
#define BARCODE_UTILS_H

#include <opencv2/opencv.hpp>

class BarcodeDetector {
public:
    explicit BarcodeDetector(bool verbose = false);
    cv::Mat scan(const cv::Mat& input);

private:
    bool verbose;
    std::string decodedText = "";
    const std::string resultDir = "./assets/results/";

    void save_debug(const std::string& name, const cv::Mat& img);
    void pre_process_image(const cv::Mat& input, cv::Mat& output);
    void detect_edges(const cv::Mat& input, cv::Mat& output);
    cv::RotatedRect get_barcode_region(const cv::Mat& edgeMask);
    cv::Mat extract_barcode(const cv::Mat& original, const cv::RotatedRect& rect);
    std::string decode_scanline(const cv::Mat& barcodeCrop, int row);
    std::pair<int, char> lookup_digit(int r1, int r2, int r3, int r4, bool isRightSide);
    bool validate_checksum(const std::string& ean);
};

#endif