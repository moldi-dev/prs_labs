#include "BarcodeUtils.h"
#include <vector>
#include <string>
#include <filesystem>

using namespace cv;
using namespace std;
using namespace filesystem;

// EAN-13 structure:
// Left guard (101) + 6 digits (L/G patterns) + middle guard (01010) + 6 digits (R patterns) + right guard (101)

// Dictionary for L-codes (odd parity):
// Normalized widths (space-bar-space-bar) sum to 7
// Key: {w1, w2, w3, w4} => digit
// G-codes are the reverse of L-codes
const int L_CODES[10][4] = {
    {3, 2, 1, 1}, // 0
    {2, 2, 2, 1}, // 1
    {2, 1, 2, 2}, // 2
    {1, 4, 1, 1}, // 3
    {1, 1, 3, 2}, // 4
    {1, 2, 3, 1}, // 5
    {1, 1, 1, 4}, // 6
    {1, 3, 1, 2}, // 7
    {1, 2, 1, 3}, // 8
    {3, 1, 1, 2}  // 9
};

// Parity table for the first implicit digit (based on L/G sequence of first 6 digits):
// Index: digit 0 - 9
// Value: pattern string
const string PARITY_PATTERNS[10] = {
    "LLLLLL", // 0
    "LLGLGG", // 1
    "LLGGLG", // 2
    "LLGGGL", // 3
    "LGLLGG", // 4
    "LGGLLG", // 5
    "LGGGLL", // 6
    "LGLGLG", // 7
    "LGLGGL", // 8
    "LGGLGL"  // 9
};

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
        cout << "[Error] No barcode region detected." << endl;
        return input.clone();
    }

    save_debug("4_final_crop", finalCrop);

    if (verbose) {
        cout << "[Step 5] Attempting decoding..." << endl;
    }

    // Try multiple scanlines: center, then +/- offsets
    int centerY = finalCrop.rows / 2;
    vector<int> scanRows = {centerY, centerY - 5, centerY + 5, centerY - 10, centerY + 10};

    string code = "";

    for (int y : scanRows) {
        if (y < 0 || y >= finalCrop.rows) {
            continue;
        }

        code = decode_scanline(finalCrop, y);

        if (!code.empty()) {
            break;
        }
    }

    Mat resultImg = finalCrop.clone();

    // 6. Visualization
    if (!code.empty()) {
        cout << "[Success] DECODED EAN-13: " << code << endl;
        this->decodedText = code;

        // Draw text on image
        copyMakeBorder(resultImg, resultImg, 0, 60, 0, 0, BORDER_CONSTANT, Scalar(255, 255, 255));

        putText(resultImg, "EAN-13: " + code, Point(10, resultImg.rows - 20),
                FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 200), 2);

        line(resultImg, Point(0, centerY), Point(resultImg.cols, centerY), Scalar(0, 255, 0), 1);
    }

    else {
        cout << "[Failed] Could not decode digits." << endl;
        this->decodedText = "Decoding Failed";
        putText(resultImg, "Decode Failed", Point(10, resultImg.rows - 20),
                FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
    }

    save_debug("5_decoded_result", resultImg);
    return resultImg;
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

string BarcodeDetector::decode_scanline(const Mat& crop, int row) {
    // 1. Extract Scanline
    Mat gray;

    if (crop.channels() == 3) {
        cvtColor(crop, gray, COLOR_BGR2GRAY);
    }

    else {
        gray = crop.clone();
    }

    const uchar* ptr = gray.ptr<uchar>(row);
    int width = gray.cols;

    // 2. Binarize
    // We can't use global Otsu because the crop might have black borders
    // The simple approach is to use hard threshold or adaptive threshold on the line
    // I'll use a simple mid-point of min/max
    int minVal = 255, maxVal = 0;

    for (int i = 0; i < width; i++) {
        minVal = min(minVal, (int)ptr[i]);
        maxVal = max(maxVal, (int)ptr[i]);
    }

    int thresh = (minVal + maxVal) / 2;

    vector<int> bitstream; // 0 for space, 1 for bar
    for (int i = 0; i < width; i++) {
        bitstream.push_back(ptr[i] < thresh ? 1 : 0); // Ink is dark (< thresh) => 1
    }

    // 3. Run-Length Encoding
    vector<int> rle;
    if (bitstream.empty()) {
        return "";
    }

    int currentVal = bitstream[0];
    int count = 0;

    for (int b : bitstream) {
        if (b == currentVal) {
            count++;
        }

        else {
            rle.push_back(count); // Store length
            currentVal = b;
            count = 1;
        }
    }
    rle.push_back(count);

    // 4. Find start guard (101 => bar-space-bar)
    // RLE pattern should look like: [QuietZone], bar, space, bar, ...
    // The QuietZone is 0 (space) => we look for the sequence space(N), bar(1), space(1), bar(1)

    int startIndex = -1;

    for (int i = 0; i < rle.size() - 3; i++) {
        // bitstream[0] tells us if rle[0] is bar or space
        // We assume rle[i] is the quiet zone (space), rle[i + 1] is bar(1)

        // If bitstream starts with space (0), then odd indices are bars
        bool isSpace = bitstream[0] == 0 ? i % 2 == 0 : i % 2 != 0;

        if (!isSpace) {
            // rle[i] must be space (quiet zone)
            continue;
        }

        // Candidates for guard
        int b1 = rle[i+1];
        int s1 = rle[i+2];
        int b2 = rle[i+3];

        // Check if they are roughly equal (1:1:1 ratio)
        float avg = (b1 + s1 + b2) / 3.0f;
        float tolerance = 0.5f;

        if (abs(b1 - avg) < avg * tolerance &&
            abs(s1 - avg) < avg * tolerance &&
            abs(b2 - avg) < avg * tolerance
        ) {

            startIndex = i + 4; // Start of first digit
            break;
        }
    }

    if (startIndex == -1) {
        return ""; // Start guard not found
    }

    // 5. Decode left group (6 digits)
    string leftParity = "";
    string digits = "";
    int idx = startIndex;

    for (int d = 0; d < 6; d++) {
        if (idx + 3 >= rle.size()) {
            return "";
        }

        int r1 = rle[idx], r2 = rle[idx + 1], r3 = rle[idx + 2], r4 = rle[idx + 3];
        auto result = lookup_digit(r1, r2, r3, r4, false); // false = left side

        if (result.second == '?') {
            return ""; // Decode error
        }

        digits += to_string(result.first);
        leftParity += result.second; // L or G

        idx += 4;
    }

    // 6. Check middle guard (01010 => space-bar-space-bar-space) => 1-1-1-1-1
    // Just skip it
    idx += 5;

    // 7. Decode right group (6 digits)
    for (int d = 0; d < 6; d++) {
        if (idx + 3 >= rle.size()) {
            return "";
        }

        int r1 = rle[idx], r2 = rle[idx + 1], r3 = rle[idx + 2], r4 = rle[idx + 3];
        auto result = lookup_digit(r1, r2, r3, r4, true); // true = Right side

        if (result.second == '?') return "";

        digits += to_string(result.first);
        // Right side is always R, no parity info needed
        idx += 4;
    }

    // 8. Determine the first digit
    char firstDigit = '?';

    for (int i = 0; i < 10; i++) {
        if (PARITY_PATTERNS[i] == leftParity) {
            firstDigit = (char)('0' + i);
            break;
        }
    }

    if (firstDigit == '?') {
        return "";
    }

    string fullEan = firstDigit + digits;

    // 9. Checksum Validation
    if (validate_checksum(fullEan)) {
        return fullEan;
    }

    return "";
}

pair<int, char> BarcodeDetector::lookup_digit(int r1, int r2, int r3, int r4, bool isRightSide) {
    // Normalize widths to sum to 7
    int total = r1 + r2 + r3 + r4;

    // Simple rounding normalization
    // e.g. if total is 14 pixels, and r1 is 2 pixels: (2 / 14) * 7 = 1.0
    float scale = 7.0f / total;
    int n1 = round(r1 * scale);
    int n2 = round(r2 * scale);
    int n3 = round(r3 * scale);
    int n4 = round(r4 * scale);

    // Fix rounding errors (ensure sum is 7)
    int nSum = n1 + n2 + n3 + n4;

    if (nSum != 7) {
        // Adjust largest element to correct sum
        int diff = 7 - nSum;

        if (n1 >= n2 && n1 >= n3 && n1 >= n4) {
            n1 += diff;
        }

        else if (n2 >= n1 && n2 >= n3 && n2 >= n4) {
            n2 += diff;
        }

        else if (n3 >= n1 && n3 >= n2 && n3 >= n4) {
            n3 += diff;
        }

        else {
            n4 += diff;
        }
    }

    // Match against L-Code Table
    // Left Side: L-codes (odd) and G-codes (reverse L)
    // Right Side: R-codes (L-codes but treated as bar-space-bar-space)

    // Important on RLE:
    // Left side digits always start with SPACE (white)
    // So r1 is Space, r2 Bar, r3 Space, r4 Bar
    // The L-code table is based on this space-bar-space-bar widths

    for (int d = 0; d < 10; d++) {
        const int* code = L_CODES[d];

        // Check L-Code (direct match)
        if (code[0] == n1 && code[1] == n2 && code[2] == n3 && code[3] == n4) {
            return {d, isRightSide ? 'R' : 'L'};
        }

        // Check G-Code (reverse match)
        // G-codes are reverse of L-codes
        if (!isRightSide && code[3] == n1 && code[2] == n2 && code[1] == n3 && code[0] == n4) {
            return {d, 'G'};
        }
    }

    return {-1, '?'};
}

bool BarcodeDetector::validate_checksum(const string& ean) {
    if (ean.length() != 13) {
        return false;
    }

    int sum = 0;

    for (int i = 0; i < 12; i++) {
        int digit = ean[i] - '0';
        sum += (i % 2 == 0) ? digit : digit * 3;
    }

    int nearest10 = ceil(sum / 10.0) * 10;
    int checkDigit = nearest10 - sum;

    return checkDigit == (ean[12] - '0');
}