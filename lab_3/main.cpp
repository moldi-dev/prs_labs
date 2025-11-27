#include <fstream>
#include <limits>
#include <opencv2/opencv.hpp>
#include "src/common/common.h"
#include "src/slider/slider.h"
#include "src/common/logger/logger.h"

using namespace cv;
using namespace std;

struct peak {
    int theta, ro, hval;

    bool operator < (const peak& o) const {
        return hval > o.hval;
    }
};

void perform_hough_algorithm(Mat_<uchar> edgeImg, int windowSize, int k);

int main() {
    // Step 1: read the image
    Mat_<uchar> img = imread("assets/images_Hough/edge_simple.bmp", IMREAD_GRAYSCALE);

    namedWindow("Original Image", WINDOW_KEEPRATIO);
    imshow("Original Image", img);

    perform_hough_algorithm(img, 3, 7);

    waitKey(0);

    return 0;
}

void perform_hough_algorithm(Mat_<uchar> edgeImg, int windowSize, int k) {
    // Step 2: initialize the Hough accumulator
    int width = edgeImg.cols;
    int height = edgeImg.rows;

    int diagonal = (int)round(sqrt(width * width + height * height));
    Mat hough = Mat::zeros(diagonal + 1, 360, CV_32SC1);

    // Step 3: fill in the accumulator
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (edgeImg(y, x) == 255) {
                for (int theta = 0; theta < 360; theta++) {
                    double thetaRad = theta * CV_PI / 180;
                    int ro = round(x * cos(thetaRad) + y * sin(thetaRad));

                    if (ro >= 0 && ro <= diagonal) {
                        hough.at<int>(ro, theta)++;
                    }
                }
            }
        }
    }

    // Step 4: normalize and display the accumulator
    double maxHoughValue;
    minMaxLoc(hough, 0, &maxHoughValue);
    Mat houghImg;
    hough.convertTo(houghImg, CV_8UC1, 255.0 / maxHoughValue);

    namedWindow("Hough Accumulator", WINDOW_KEEPRATIO);
    imshow("Hough Accumulator", houghImg);

    // Step 5: detect the local maxima
    vector<peak> peaks;
    int roDim = hough.rows;
    int thetaDim = hough.cols;
    int halfWindow = windowSize / 2;

    for (int ro = halfWindow; ro < roDim - halfWindow; ro++) {
        for (int theta = halfWindow; theta < thetaDim - halfWindow; theta++) {
            int currentVal = hough.at<int>(ro, theta);

            if (currentVal == 0) {
                continue;
            }

            bool isLocalMax = true;

            for (int i = -halfWindow; i <= halfWindow; i++) {
                for (int j = -halfWindow; j <= halfWindow; j++) {
                    if (hough.at<int>(ro + i, theta + j) > currentVal) {
                        isLocalMax = false;
                        break;
                    }
                }

                if (!isLocalMax) {
                    break;
                }
            }

            if (isLocalMax) {
                peaks.push_back({ theta, ro, currentVal });
            }
        }
    }

    sort(peaks.begin(), peaks.end());

    if (peaks.size() > k) {
        peaks.resize(k);
    }

    // Step 6: draw the lines on the image and display the results
    Mat detectedLines;
    cvtColor(edgeImg, detectedLines, COLOR_GRAY2BGR);

    for (peak peak : peaks) {
        double ro = peak.ro;
        double thetaRad = peak.theta * CV_PI / 180.0;

        double a = cos(thetaRad);
        double b = sin(thetaRad);

        double x0 = a * ro;
        double y0 = b * ro;

        Point2d pt1;
        Point2d pt2;

        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));

        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));

        line(detectedLines, pt1, pt2, Scalar(0, 255, 0), 1);
    }

    namedWindow("Detected Lines", WINDOW_KEEPRATIO);
    imshow("Detected Lines", detectedLines);
}
