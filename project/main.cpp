#include <fstream>
#include <limits>
#include <random>
#include <opencv2/opencv.hpp>
#include "src/common/common.h"
#include "src/slider/slider.h"
#include "src/common/logger/logger.h"
#include "src/utils/BarcodeUtils.h"

using namespace cv;
using namespace std;
using namespace BarcodeUtils;

int main() {
    cout << "Starting EAN-13 Barcode Detection Pipeline... \n";

    // Step 0: Load an image containing a barcode
    string imagePath = "./assets/sample_images/image_2.jpg";
    Mat src = imread(imagePath);

    if (src.empty()) {
        cerr << "Error: Could not open or find the image at " << imagePath << endl;
        return -1;
    }

    // Step 1: Preprocessing
    Mat preprocessed;
    preprocess_image(src, preprocessed);

    // Step 2: Edge Detection
    Mat edges;
    detect_edges(preprocessed, edges);

    // Step 3: Find the Region (Get coordinates from the blob)
    RotatedRect barcodeRect = get_barcode_region(edges);

    // Step 4: Crop and Restore (Apply coordinates to the ORIGINAL image)
    Mat finalBarcode = extract_barcode(src, barcodeRect);

    // Display results for debugging
    namedWindow("Original image", WINDOW_KEEPRATIO);
    imshow("Original image", src);

    namedWindow("Preprocessed (Gray + CLAHE + Blur)", WINDOW_KEEPRATIO);
    imshow("Preprocessed (Gray + CLAHE + Blur)", preprocessed);

    namedWindow("Edges (Sobel + Morph)", WINDOW_KEEPRATIO);
    imshow("Edges (Sobel + Morph)", edges);

    namedWindow("Barcode image", WINDOW_KEEPRATIO);
    imshow("Barcode image", finalBarcode);

    waitKey(0);

    return 0;
}