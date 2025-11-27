#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "src/utils/BarcodeUtils.h"

using namespace std;
using namespace cv;

// Helper to parse CLI arguments manually
string get_cmd_option(char** begin, char** end, const string & option) {
    for (char** it = begin; it != end; ++it) {
        string arg = *it;

        if (arg.find(option) == 0) {
            size_t pos = arg.find("=");

            if (pos != string::npos) {
                return arg.substr(pos + 1);
            }
        }
    }

    return "";
}

int main(int argc, char* argv[]) {
    // 1. Parse Arguments
    string imagePath = get_cmd_option(argv, argv + argc, "--image_path");
    string verboseStr = get_cmd_option(argv, argv + argc, "--verbose");

    // Simple validation
    if (imagePath.empty()) {
        cerr << "Usage: " << argv[0] << " --image_path=<path> [--verbose=true|false]" << endl;
        return -1;
    }

    bool verbose = (verboseStr == "true" || verboseStr == "1");

    // 2. Load Image
    Mat src = imread(imagePath);
    if (src.empty()) {
        cerr << "Error: Could not open image at " << imagePath << endl;
        return -1;
    }

    // 3. Instantiate Detector and Run
    BarcodeDetector detector(verbose);
    Mat finalResult = detector.scan(src);

    // 4. Output Logic
    if (finalResult.empty()) {
        cerr << "Failed to detect barcode." << endl;
    }

    else {
        namedWindow("Result", WINDOW_KEEPRATIO);
        imshow("Result", finalResult);
        cout << "Barcode detected successfully." << endl;
    }

    waitKey(0);

    return 0;
}