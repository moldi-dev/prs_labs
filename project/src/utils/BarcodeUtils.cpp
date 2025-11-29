#include "BarcodeUtils.h"
#include <vector>
#include <string>
#include <filesystem>

using namespace cv;
using namespace std;
using namespace filesystem;

BarcodeDetector::BarcodeDetector(bool verbose) {
    this->verbose = verbose;

    if (!exists(resultDir)) {
        create_directory(resultDir);
    }
}

void BarcodeDetector::save_debug(const string& name, const Mat& img) {
    if (verbose && !img.empty()) {
        string path = resultDir + name + ".jpg";
        imwrite(path, img);
        cout << "[Disk] Saved " << name << endl;
    }
}

Mat BarcodeDetector::scan(const Mat& input) {
    if (verbose) {
        cout << "[Step 0] Starting the barcode scanner pipeline..." << endl;
        cout << "   > Input Resolution: " << input.cols << " x " << input.rows << endl;

        save_debug("0_original", input);
    }

    // 1. Preprocess
    Mat preprocessed;
    pre_process_image(input, preprocessed);

    // 2. Edges
    Mat edges;
    detect_edges(preprocessed, edges);

    // 3. Localization
    RotatedRect rect = get_barcode_region(edges);

    // 4. Extraction
    Mat finalCrop = extract_barcode(input, rect);

    if (finalCrop.empty()) {
        cout << "Warning: No barcode detected" << endl;
    }

    else {
        save_debug("4_final_crop", finalCrop);
    }

    return finalCrop;
}

void BarcodeDetector::pre_process_image(const Mat& input, Mat& output) {
    Mat gray;

    if (input.channels() == 3) {
        cvtColor(input, gray, COLOR_BGR2GRAY);
    }

    else {
        gray = input.clone();
    }

    // 1. Apply CLAHE
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
    Mat claheImg;
    clahe->apply(gray, claheImg);

    // 2. Apply smoothing
    GaussianBlur(claheImg, output, Size(5, 5), 0);

    if (verbose) {
        cout << "[Step 1] Preprocessing complete" << endl;
        save_debug("1_preprocessed", output);
    }
}

void BarcodeDetector::detect_edges(const Mat& input, Mat& output) {
    // 1. Calculate Gradients in BOTH directions
    // We need both because a rotated barcode has X and Y gradient components
    Mat gradX, gradY;
    Sobel(input, gradX, CV_16S, 1, 0, 3);
    Sobel(input, gradY, CV_16S, 0, 1, 3);

    Mat absGradX, absGradY;
    convertScaleAbs(gradX, absGradX);
    convertScaleAbs(gradY, absGradY);

    // 2. Combine to get Gradient Magnitude
    // This allows us to see edges regardless of rotation angle
    Mat gradient;
    addWeighted(absGradX, 0.5, absGradY, 0.5, 0, gradient);

    // 3. Threshold
    Mat thresh;
    threshold(gradient, thresh, 0, 255, THRESH_BINARY | THRESH_OTSU);

    // 4. Morphological Closing
    // When a barcode is rotated ~30-45 degrees, the gap between bars becomes diagonal
    // A taller kernel ensures we can reach the neighbor bar vertically to fuse them
    Mat kernel = getStructuringElement(MORPH_RECT, Size(21, 11));
    morphologyEx(thresh, output, MORPH_CLOSE, kernel);

    // 5. Erosion (Remove Noise/Text)
    Mat erodeKernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    erode(output, output, erodeKernel, Point(-1, -1), 4);

    // 6. Dilation (Restore Volume)
    Mat dilateKernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(output, output, dilateKernel, Point(-1, -1), 4);

    if (verbose) {
        cout << "[Step 2] Edge detection complete" << endl;
        save_debug("2_edges", output);
    }
}

RotatedRect BarcodeDetector::get_barcode_region(const Mat& edgeMask) {
    vector<vector<Point>> contours;
    findContours(edgeMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    RotatedRect bestRect;
    double maxArea = 0;

    for (const auto& c : contours) {
        RotatedRect rect = minAreaRect(c);

        float width = rect.size.width;
        float height = rect.size.height;

        // Ensure width is the long side
        if (width < height) {
            swap(width, height);
        }

        double area = width * height;
        double aspectRatio = width / height;

        if (area > maxArea && aspectRatio > 1.6) {
            maxArea = area;
            bestRect = rect;
        }
    }

    if (verbose) {
        cout << "[Step 3] Region search complete. Found candidate data:" << endl;

        if (maxArea > 0) {
            cout << "   > Center: " << bestRect.center << endl;
            cout << "   > Size: " << bestRect.size << endl;
            cout << "   > Angle: " << bestRect.angle << endl;

            // Draw the visualization on a temporary image
            Mat debugImg;
            cvtColor(edgeMask, debugImg, COLOR_GRAY2BGR);
            Point2f v[4];
            bestRect.points(v);

            for (int i = 0; i < 4; i++) {
                line(debugImg, v[i], v[(i + 1) % 4], Scalar(0, 0, 255), 3);
            }

            save_debug("3_bounding_box", debugImg);
        }

        else {
            cout << "   > No valid barcode region found" << endl;
        }
    }

    return bestRect;
}

Mat BarcodeDetector::extract_barcode(const Mat& original, const RotatedRect& rect) {
    if (rect.size.area() < 10) {
        return Mat();
    }

    float angle = rect.angle;
    Size2f size = rect.size;
    Point2f center = rect.center;

    // Orientation correction
    if (size.width < size.height) {
        angle += 90.0f;
        swap(size.width, size.height);
    }

    if (verbose) {
        cout << "[Step 4] Extracting the barcode" << endl;
        cout << "   > Corrected Angle: " << angle << endl;
        cout << "   > Target Size: " << size << endl;
    }

    Mat M = getRotationMatrix2D(center, angle, 1.0);
    Mat rotated, cropped;

    warpAffine(original, rotated, M, original.size(), INTER_CUBIC);
    getRectSubPix(rotated, size, center, cropped);

    // Content-based orientation check (Sobel gradients)
    Mat gray, gradX, gradY;

    if (cropped.channels() == 3) {
        cvtColor(cropped, gray, COLOR_BGR2GRAY);
    }

    else {
        gray = cropped.clone();
    }

    Sobel(gray, gradX, CV_16S, 1, 0);
    Sobel(gray, gradY, CV_16S, 0, 1);

    double sumX = sum(abs(gradX))[0];
    double sumY = sum(abs(gradY))[0];

    if (sumY > sumX) {
        if (verbose) {
            cout << "   > Detected horizontal bars. Rotating 90 degrees" << endl;
        }

        rotate(cropped, cropped, ROTATE_90_CLOCKWISE);
    }

    else {
        if (verbose) {
            cout << "   > Orientation confirmed correct" << endl;
        }
    }

    return cropped;
}