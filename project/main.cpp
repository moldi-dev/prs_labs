#define CVUI_IMPLEMENTATION
#include "src/ui/cvui.h"

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "src/utils/BarcodeUtils.h"

using namespace cv;
using namespace std;
using namespace filesystem;
using namespace cvui;

class ConsoleBuffer {
private:
    stringstream buffer;
    streambuf* old;
    vector<string> lines;

public:
    ConsoleBuffer() {
        old = cout.rdbuf(buffer.rdbuf());
    }

    ~ConsoleBuffer() {
        cout.rdbuf(old);
    }

    void update() {
        if (buffer.rdbuf()->in_avail() > 0) {
            string text = buffer.str();
            buffer.str("");
            buffer.clear();

            stringstream ss(text);
            string line;

            while (getline(ss, line)) {
                if (!line.empty()) {
                    lines.push_back(line);
                }
            }
        }
    }

    const vector<string>& getLines() const {
        return lines;
    }

    size_t size() const {
        return lines.size();
    }

    void clear() {
        lines.clear();
    }
};

int main() {
    string WINDOW_NAME = "EAN-13 Barcode Scanner";
    init(WINDOW_NAME);

    Mat frame = Mat(800, 1200, CV_8UC3);

    // Application State
    bool verbose = true;
    int currentTab = 0;
    int logScrollPos = 0;

    // UI Interaction States
    bool isDraggingScrollbar = false;
    string imageName = "image_1.jpg";
    bool inputFocus = false;

    Mat imgOriginal, imgPre, imgEdges, imgBox, imgResult;
    bool hasRun = false;

    ConsoleBuffer console;
    cout << "System Ready." << endl;
    cout << "Input Directory: ./assets/sample_images/" << endl;

    while (true) {
        frame = Scalar(49, 52, 57);
        console.update();

        window(frame, 20, 20, 220, 760, "Settings");

        int viewX = 260; int viewY = 60;
        int viewW = 900; int viewH = 500;
        int consoleX = 260; int consoleY = 600;
        int consoleW = 920; int consoleH = 180;

        // Input Field
        text(frame, 35, 60, "Image Filename:");
        unsigned int borderColor = inputFocus ? 0xFF0000 : 0x666666;
        rect(frame, 35, 80, 190, 30, 0x333333, borderColor);
        text(frame, 40, 87, imageName);

        // Manual Hit Test for Input Field
        if (mouse(CLICK)) {
            Point p = mouse();
            Rect inputRect(35, 80, 190, 30);

            if (inputRect.contains(p)) {
                inputFocus = true;
            }

            else {
                inputFocus = false;
            }
        }

        // Load Button
        if (button(frame, 35, 120, 190, 40, "Load Image")) {
            string path = "./assets/sample_images/" + imageName;
            cout << "\n[IO] Attempting to load: " << path << endl;

            if (exists(path)) {
                imgOriginal = imread(path);

                if (!imgOriginal.empty()) {
                    cout << "[IO] Success: Loaded " << imageName << " (" << imgOriginal.cols << "x" << imgOriginal.rows << ")" << endl;
                    hasRun = false;
                    currentTab = 0;
                }

                else {
                    cout << "[Error] File exists but OpenCV could not decode it." << endl;
                }
            }

            else {
                cout << "[Error] File not found: " << path << endl;
            }
        }

        checkbox(frame, 35, 180, "Verbose Mode", &verbose);

        // Run Pipeline
        if (button(frame, 35, 220, 190, 40, "Run Pipeline")) {
            if (imgOriginal.empty()) {
                cout << "[Error] Please load an image first." << endl;
            }

            else {
                cout << "\n--- Running Pipeline ---" << endl;
                BarcodeDetector detector(verbose);
                imgResult = detector.scan(imgOriginal);
                hasRun = true;

                if (verbose) {
                    cout << "[GUI] Reloading debug layers from ./assets/results/ ..." << endl;
                    imgPre = imread("./assets/results/1_preprocessed.jpg");
                    imgEdges = imread("./assets/results/2_edges.jpg");
                    imgBox = imread("./assets/results/3_bounding_box.jpg");
                    imgResult = imread("./assets/results/4_final_crop.jpg");
                }

                currentTab = verbose ? 4 : 0;
                logScrollPos = 99999;
            }
        }

        // OCR Button
        if (button(frame, 35, 280, 190, 40, "Run OCR")) {
             cout << "\n[OCR] Initializing Tesseract..." << endl;
             cout << "[OCR] Error: OCR module not fully implemented in this build." << endl;
             logScrollPos = 99999;
        }

        // Clear Log
        if (button(frame, 35, 720, 190, 30, "Clear Log")) {
            console.clear();
            logScrollPos = 0;
        }

        // Image Viewer Area
        if (hasRun && verbose) {
            int tabW = 100;

            if (button(frame, viewX,20, tabW, 30, "Original")) {
                currentTab = 0;
            }

            if (button(frame, viewX + 110, 20, tabW, 30, "Preprocessed")) {
                currentTab = 1;
            }

            if (button(frame, viewX + 220, 20, tabW, 30, "Edges")) {
                currentTab = 2;
            }

            if (button(frame, viewX + 330, 20, tabW, 30, "Region")) {
                currentTab = 3;
            }

            if (button(frame, viewX + 440, 20, tabW, 30, "Barcode")) {
                currentTab = 4;
            }
        }

        Mat* displayPtr = nullptr;

        if (hasRun || !imgOriginal.empty()) {
            if (!hasRun) {
                displayPtr = &imgOriginal;
            }

            else if (!verbose) {
                displayPtr = imgResult.empty() ? &imgOriginal : &imgResult;
            }

            else {
                switch (currentTab) {
                    case 0: displayPtr = &imgOriginal; break;
                    case 1: displayPtr = &imgPre; break;
                    case 2: displayPtr = &imgEdges; break;
                    case 3: displayPtr = &imgBox; break;
                    case 4: displayPtr = &imgResult; break;
                }
            }
        }

        if (displayPtr && !displayPtr->empty()) {
            float scaleX = (float)viewW / displayPtr->cols;
            float scaleY = (float)viewH / displayPtr->rows;
            float scale = min(scaleX, scaleY);

            if (scale > 1.0f) {
                scale = 1.0f;
            }

            Mat viz;
            resize(*displayPtr, viz, Size(), scale, scale);

            int offsetX = (viewW - viz.cols) / 2;
            int offsetY = (viewH - viz.rows) / 2;

            image(frame, viewX + offsetX, viewY + offsetY, viz);
            string info = "Res: " + to_string(displayPtr->cols) + "x" + to_string(displayPtr->rows);
            text(frame, viewX, viewY + viewH + 5, info, 0.5, 0xBBBBBB);
        }

        else {
            rect(frame, viewX, viewY, viewW, viewH, 0x444444, 0x444444);
            text(frame, viewX + viewW/2 - 50, viewY + viewH/2, "No Image Data");
        }

        // Scrollable Console
        window(frame, consoleX, consoleY, consoleW, consoleH, "Console Log");

        int lineHeight = 15;
        int maxVisibleLines = (consoleH - 40) / lineHeight;
        size_t totalLines = console.size();

        int maxScrollIndex = 0;

        if (totalLines > maxVisibleLines) {
            maxScrollIndex = (int)totalLines - maxVisibleLines;
        }

        // Clamp existing scroll
        if (logScrollPos > maxScrollIndex) {
            logScrollPos = maxScrollIndex;
        }

        if (logScrollPos < 0) {
            logScrollPos = 0;
        }

        if (totalLines > maxVisibleLines) {
            // Define geometry
            int barX = consoleX + consoleW - 20;
            int barY = consoleY + 30;
            int barW = 15;
            int barH = consoleH - 40;
            int thumbH = 30;

            Point mouseP = mouse();
            bool isMouseDown = mouse(DOWN);

            // 1. Check if we just started dragging
            // condition: Mouse is down AND inside the bar rect
            Rect barRect(barX, barY, barW, barH);

            if (isMouseDown) {
                if (isDraggingScrollbar) {
                    // CASE A: CONTINUING A DRAG
                    // We don't care if mouse is inside rect, just calculate pos
                    int relativeY = mouseP.y - barY - (thumbH/2);
                    float ratio = (float)relativeY / (barH - thumbH);

                    if (ratio < 0) {
                        ratio = 0;
                    }

                    if (ratio > 1) {
                        ratio = 1;
                    }

                    logScrollPos = (int)(ratio * maxScrollIndex);

                }

                else if (barRect.contains(mouseP)) {
                    // CASE B: STARTING A DRAG
                    isDraggingScrollbar = true;
                    // Initial jump to mouse pos
                    int relativeY = mouseP.y - barY - (thumbH/2);
                    float ratio = (float)relativeY / (barH - thumbH);

                    if (ratio < 0) {
                        ratio = 0;
                    }

                    if (ratio > 1) {
                        ratio = 1;
                    }

                    logScrollPos = (int)(ratio * maxScrollIndex);
                }
            }

            else {
                // Mouse released
                isDraggingScrollbar = false;
            }

            // Draw Track
            rect(frame, barX, barY, barW, barH, 0x333333, 0x555555);

            // Draw Thumb
            // Recalculate thumbY based on potentially updated logScrollPos
            float currentRatio = (float)logScrollPos / maxScrollIndex;
            int thumbY = barY + (int)(currentRatio * (barH - thumbH));

            unsigned int thumbColor = isDraggingScrollbar ? 0xAAAAAA : 0x888888;
            rect(frame, barX, thumbY, barW, thumbH, thumbColor, thumbColor);
        }

        int textY = consoleY + 30;
        int drawnCount = 0;
        const vector<string>& allLines = console.getLines();

        for (size_t i = logScrollPos; i < totalLines && drawnCount < maxVisibleLines; i++) {
            text(frame, consoleX + 10, textY, allLines[i], 0.4, 0x00ff00);
            textY += lineHeight;
            drawnCount++;
        }

        update();
        cv::imshow(WINDOW_NAME, frame);

        int key = waitKey(20);

        if (key == 27) {
            break;
        }

        if (inputFocus && key != -1) {
            if (key == 8 || key == 127) {
                if (!imageName.empty()) {
                    imageName.pop_back();
                }
            }

            else if (key == 13) {
                inputFocus = false;
            }

            else if (  (key >= 'a' && key <= 'z') ||
                       (key >= 'A' && key <= 'Z') ||
                       (key >= '0' && key <= '9') ||
                       key == '.' ||
                       key == '_' ||
                       key == '-'
            ) {
                if (imageName.length() < 30) {
                    imageName += (char)key;
                }
            }
        }
    }

    return 0;
}