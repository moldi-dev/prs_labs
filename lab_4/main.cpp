#include <fstream>
#include <limits>
#include <opencv2/opencv.hpp>
#include "src/common/common.h"
#include "src/slider/slider.h"
#include "src/common/logger/logger.h"

using namespace cv;
using namespace std;

int di[9] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
int dj[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
int weight[9] = {3, 2, 3, 2, 0, 2, 3, 2, 3};

Mat_<uchar> perform_chamfer_DT(Mat_<uchar> src);
double compute_matching_score(Mat_<uchar> dt, Mat_<uchar> object);

int main() {
    Mat_<uchar> img = imread("assets/images_DT_PM/PatternMatching/template.bmp", IMREAD_GRAYSCALE);
    Mat_<uchar> object1 = imread("assets/images_DT_PM/PatternMatching/template.bmp", IMREAD_GRAYSCALE);
    Mat_<uchar> object2 = imread("assets/images_DT_PM/PatternMatching/unknown_object1.bmp", IMREAD_GRAYSCALE);
    Mat_<uchar> object3 = imread("assets/images_DT_PM/PatternMatching/unknown_object2.bmp", IMREAD_GRAYSCALE);

    Mat dt = perform_chamfer_DT(img);

    namedWindow("Original Image", WINDOW_KEEPRATIO);
    imshow("Original Image", img);

    namedWindow("Distance Transform Image", WINDOW_KEEPRATIO);
    imshow("Distance Transform Image", dt);

    namedWindow("Object Image 1", WINDOW_KEEPRATIO);
    imshow("Object Image 1", object1);

    namedWindow("Object Image 2", WINDOW_KEEPRATIO);
    imshow("Object Image 2", object2);

    namedWindow("Object Image 3", WINDOW_KEEPRATIO);
    imshow("Object Image 3", object3);

    double score1 = compute_matching_score(dt, object1);
    cout << "Matching score 1: " << score1 << endl;

    double score2 = compute_matching_score(dt, object2);
    cout << "Matching score 2: " << score2 << endl;

    double score3 = compute_matching_score(dt, object3);
    cout << "Matching score 3: " << score3<< endl;

    waitKey(0);

    return 0;
}

Mat_<uchar> perform_chamfer_DT(Mat_<uchar> src) {
    // Step 1: initialize the DT map
    Mat_<uchar> dt = src.clone();
    int height = dt.rows;
    int width = dt.cols;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (src(i, j) == 0) {
                dt(i, j) = 0;
            }

            else {
                dt(i, j) = 255;
            }
        }
    }

    // Step 2: scan top-down and left-right
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int minDist = dt(i, j);

            for (int k = 0; k < 4; k++) {
                int I = i + di[k];
                int J = j + dj[k];

                if (I >= 0 && I < height && J >= 0 && J < width) {
                    int neighborDist = dt(I, J) + weight[k];
                    minDist = min(minDist, neighborDist);
                }
            }

            dt(i,j) = (uchar)min(minDist, 255);
        }
    }

    // Step 3: scan bottom-up and right-left
    for (int i = height - 1; i >= 0; i--) {
        for (int j = width - 1; j >= 0; j--) {
            int minDist = dt(i, j);

            for (int k = 4; k < 9; k++) {
                int I = i + di[k];
                int J = j + dj[k];

                if (I >= 0 && I < height && J >= 0 && J < width) {
                    int neighborDist = dt(I, J) + weight[k];
                    minDist = min(minDist, neighborDist);
                }
            }

            dt(i, j) = (uchar)min(minDist, 255);
        }
    }

    return dt;
}

double compute_matching_score(Mat_<uchar> dt, Mat_<uchar> object) {
    double total = 0;
    int contourPointsCounter = 0;
    int height = object.rows;
    int width = object.cols;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (object(i, j) == 0) {
                total += dt(i, j);
                contourPointsCounter++;
            }
        }
    }

    return contourPointsCounter > 0 ? total / contourPointsCounter : 255;
}
