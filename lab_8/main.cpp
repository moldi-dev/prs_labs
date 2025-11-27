#include <fstream>
#include <limits>
#include <random>
#include <opencv2/opencv.hpp>
#include "src/common/common.h"
#include "src/slider/slider.h"
#include "src/common/logger/logger.h"

using namespace cv;
using namespace std;

const int nrClasses = 6;
char classes[nrClasses][10] = {
    "beach", "city", "desert", "forest", "landscape", "snow"
};
int nrBins = 8;
int featureDim = nrBins * 3;

void compute_histogram(Mat& img, vector<float>& hist);
int classify_KNN(Mat& X, Mat& y, vector<float>& feat, int K);

void compute_histogram(Mat& img, vector<float>& hist) {
    hist.assign(nrBins * 3, 0);

    int binSize = 256 / nrBins;

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            Vec3b p = img.at<Vec3b>(i, j);

            int b = p[0] / binSize;
            int g = p[1] / binSize;
            int r = p[2] / binSize;

            if (b >= nrBins) {
                b = nrBins - 1;
            }

            if (g >= nrBins) {
                g = nrBins - 1;
            }

            if (r >= nrBins) {
                r = nrBins - 1;
            }

            hist[b]++;
            hist[g + nrBins]++;
            hist[r + 2 * nrBins]++;
        }
    }
}

int classify_KNN(Mat& X, Mat& y, vector<float>& feat, int K) {
    vector<pair<float, int>> dist;

    for (int i = 0; i < X.rows; i++) {
        float d = 0;

        for (int j = 0; j < featureDim; j++) {
            float diff = feat[j] - X.at<float>(i, j);
            d += diff * diff;
        }

        dist.push_back({ sqrt(d), y.at<uchar>(i, 0) });
    }

    sort(dist.begin(), dist.end(), [](auto& a, auto& b) { return a.first < b.first; });

    vector<int> votes(nrClasses, 0);

    for (int k = 0; k < K; k++) {
        votes[dist[k].second]++;
    }

    int bestClass = 0;

    for (int c = 1; c < nrClasses; c++) {
        if (votes[c] > votes[bestClass]) {
            bestClass = c;
        }
    }

    return bestClass;
}

int main() {
    vector<string> trainFolders = {"./assets/images_KNN/train/"};
    vector<string> testFolders  = {"./assets/images_KNN/test/"};

    vector<vector<string>> trainList(nrClasses);
    vector<vector<string>> testList(nrClasses);

    vector<float> hist;
    vector<float> feat;

    char buf[256];
    int nrTrain = 0;
    int row = 0;
    int pred;

    float correct = 0;
    float total = 0;

    Mat C = Mat::zeros(nrClasses, nrClasses, CV_32SC1);
    Mat img;

    for (int c = 0; c < nrClasses; c++) {
        for (int i = 0; true; i++) {
            sprintf(buf, "%s%s/%06d.jpeg", trainFolders[0].c_str(), classes[c], i);

            img = imread(buf);

            if (img.empty()) {
                break;
            }

            trainList[c].push_back(buf);
        }
    }

    for (int c = 0; c < nrClasses; c++) {
        nrTrain += trainList[c].size();
    }

    Mat X(nrTrain, featureDim, CV_32FC1);
    Mat y(nrTrain, 1, CV_8UC1);

    for (int c = 0; c < nrClasses; c++) {
        for (string f : trainList[c]) {
            img = imread(f);

            compute_histogram(img, hist);

            for (int d = 0; d < featureDim; d++) {
                X.at<float>(row, d) = hist[d];
            }

            y.at<uchar>(row) = c;
            row++;
        }
    }

    cout << "Loaded training set: " << nrTrain << " images\n";

    for (int c = 0; c < nrClasses; c++) {
        for (int i = 0; true; i++) {
            sprintf(buf, "%s%s/%06d.jpeg", testFolders[0].c_str(), classes[c], i);

            img = imread(buf);

            if (img.empty()) {
                break;
            }

            compute_histogram(img, feat);

            pred = classify_KNN(X, y, feat, 6);

            C.at<int>(pred, c)++;
        }
    }

    cout << "\nConfusion matrix:\n";
    cout << C << "\n\n";

    for (int i = 0; i < nrClasses; i++) {
        for (int j = 0; j < nrClasses; j++) {
            total += C.at<int>(i, j);

            if (i == j) {
                correct += C.at<int>(i, j);
            }
        }
    }

    cout << "Accuracy = " << (correct / total) * 100.0 << "%\n";

    return 0;
}