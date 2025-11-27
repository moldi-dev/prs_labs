#include <fstream>
#include <limits>
#include <opencv2/opencv.hpp>
#include "src/common/common.h"
#include "src/slider/slider.h"
#include "src/common/logger/logger.h"

using namespace cv;
using namespace std;

Mat read_data(string filePath);
pair<Mat, Mat> subtract_mean(Mat X);
Mat compute_covariance(Mat XzeroMean);
void eigen_decomposition(Mat& C, Mat& eigenValues, Mat& eigenVectors);
void print_eigenvalues(Mat eigenValues);
Mat compute_pca_coefficients(Mat XzeroMean, Mat Q);
Mat reconstruct_k(Mat Xcoef, Mat Q, Mat meanRow, int k);
double mean_abs_diff(Mat X, Mat Xk);
void min_max_by_column(Mat Xcoef, Mat& mins, Mat& maxs);
void plot_2d_points(Mat Xcoef, string windowName);
void plot_3d_grayscale(Mat Xcoef, string windowName);

int main() {
    // Step 1: read input file path
    string filePath = "./assets/data_PCA/pca3d.txt";
    Mat X = read_data(filePath);

    // Step 2: compute mean and zero-mean data
    auto [meanRow, XzeroMean] = subtract_mean(X);

    // Step 3: covariance matrix
    Mat C = compute_covariance(XzeroMean);

    // Step 4: eigen-decomposition
    Mat eigenValues, Q;
    eigen_decomposition(C, eigenValues, Q);

    // Step 5: print eigenvalues
    print_eigenvalues(eigenValues);

    // Step 6: compute PCA coefficients and kth approximation
    int k = 1;
    Mat Xcoef = compute_pca_coefficients(XzeroMean, Q);
    Mat Xk = reconstruct_k(Xcoef, Q, meanRow, k);

    // Step 7: mean absolute difference between X and Xk
    double meanAbsoluteDiff = mean_abs_diff(X, Xk);
    cout << "Mean absolute difference with k = " << k << ": " << meanAbsoluteDiff << "\n";

    // Step 8: min/max per column of Xcoef
    Mat mins;
    Mat maxs;
    min_max_by_column(Xcoef, mins, maxs);

    cout << "Xcoef column mins: ";
    for (int j = 0; j < mins.cols; j++) {
        cout << mins.at<double>(0, j) << (j + 1 < mins.cols? ", ":"\n");
    }

    cout << "Xcoef column maxs: ";
    for (int j = 0; j < maxs.cols; j++) {
        cout << maxs.at<double>(0, j) << (j + 1 < maxs.cols? ", ":"\n");
    }

    // Step 9: if d <= 2, plot 2D using first 2 coefficients
    if (X.cols <= 2) {
        plot_2d_points(Xcoef, "PCA 2D");
        waitKey(0);
    }

    // Step 10: if d > 2, plot grayscale image using first 3 coefficients
    else if (X.cols > 2) {
        plot_3d_grayscale(Xcoef, "PCA 3D");
        waitKey(0);
    }

    return 0;
}

// Step 1: read the list of data points from file "n d" then n rows with d values
Mat read_data(string filePath) {
    ifstream fin(filePath);
    int nPoints = 0;
    int dims = 0;
    fin >> nPoints >> dims;

    Mat X(nPoints, dims, CV_64FC1);

    for (int i = 0; i < nPoints; i++) {
        for (int j = 0; j < dims; j++) {
            double val;
            fin >> val;

            X.at<double>(i, j) = val;
        }
    }

    return X;
}

// Step 2: compute the mean vector and subtract it from the data points
pair<Mat, Mat> subtract_mean(Mat X) {
    Mat meanRow(1, X.cols, CV_64FC1);

    for (int j = 0; j < X.cols; j++) {
        meanRow.at<double>(0, j) = mean(X.col(j))[0];
    }

    Mat Xzm = X.clone();

    for (int i = 0;i < X.rows; i++) {
        for (int j = 0;j < X.cols; j++) {
            Xzm.at<double>(i, j) -= meanRow.at<double>(0, j);
        }
    }

    return {meanRow, Xzm};
}

// Step 3: calculate the covariance matrix as a matrix product
Mat compute_covariance(Mat XzeroMean) {
    Mat C = (XzeroMean.t() * XzeroMean) / (XzeroMean.rows - 1);
    return C;
}

// Step 4: perform eigenvalue decomposition on the covariance matrix
void eigen_decomposition(Mat& C, Mat& eigenValues, Mat& eigenVectors) {
    eigen(C, eigenValues, eigenVectors);
    eigenVectors = eigenVectors.t();
}

// Step 5: print the eigenvalues
void print_eigenvalues(Mat eigenValues) {
    cout << "Eigenvalues: ";

    for (int i = 0; i < eigenValues.rows; i++) {
        cout << eigenValues.at<double>(i, 0) << (i + 1 < eigenValues.rows? ", ":"\n");
    }
}

// Step 6: calculate PCA coefficients Xcoef = XzeroMean * Q and build kth approximation Xk
Mat compute_pca_coefficients(Mat XzeroMean, Mat Q) {
    return XzeroMean * Q;
}

Mat reconstruct_k(Mat Xcoef, Mat Q, Mat meanRow, int k) {
    Mat Qk = Q.colRange(0, k);
    Mat Xk = Xcoef.colRange(0, k) * Qk.t();
    Mat XkWithMean = Xk.clone();

    for (int i = 0; i < XkWithMean.rows; i++) {
        for (int j = 0; j < XkWithMean.cols; j++) {
            XkWithMean.at<double>(i, j) += meanRow.at<double>(0, j);
        }
    }

    return XkWithMean;
}

// Step 7: evaluate mean absolute difference between original points and their k-dim approximation
double mean_abs_diff(Mat X, Mat Xk) {
    double sumAbs = 0.0;

    for (int i = 0; i < X.rows; i++) {
        for (int j = 0; j < X.cols; j++) {
            sumAbs += abs(X.at<double>(i, j) - Xk.at<double>(i, j));
        }
    }

    return sumAbs / (X.rows * X.cols);
}

// Step 8: find min and max values along each column of coefficient matrix
void min_max_by_column(Mat Xcoef, Mat& mins, Mat& maxs) {
    mins.create(1, Xcoef.cols, CV_64FC1);
    maxs.create(1, Xcoef.cols, CV_64FC1);

    for (int j = 0; j < Xcoef.cols; j++) {
        double min;
        double max;

        minMaxLoc(Xcoef.col(j), &min, &max);

        mins.at<double>(0, j) = min;
        maxs.at<double>(0, j) = max;
    }
}

// Step 9: if d>=2, plot 2D using first 2 coefficients
void plot_2d_points(Mat Xcoef, string windowName) {
    int n = Xcoef.rows;
    Mat mins;
    Mat maxs;

    min_max_by_column(Xcoef, mins, maxs);

    Mat coords = Xcoef.colRange(0,2).clone();

    for (int i = 0; i < n; i++) {
        coords.at<double>(i, 0) -= mins.at<double>(0, 0);
        coords.at<double>(i, 1) -= mins.at<double>(0, 1);
    }

    double maxX = maxs.at<double>(0, 0) - mins.at<double>(0, 0);
    double maxY = maxs.at<double>(0, 1) - mins.at<double>(0, 1);

    int width = (int)ceil(max(1.0, maxX)) + 10;
    int height= (int)ceil(max(1.0, maxY)) + 10;

    Mat img(height, width, CV_8UC1, Scalar(255));

    for (int i = 0; i < n; i++) {
        int x = (int)round(coords.at<double>(i, 0));
        int y = (int)round(coords.at<double>(i, 1));

        if (0 <= x && x < width && 0 <= y && y < height) {
            circle(img, Point(x, y), 1, Scalar(0), FILLED);
        }
    }

    namedWindow(windowName, WINDOW_KEEPRATIO);
    imshow(windowName, img);
}

// Step 10: if d>=3, plot grayscale image using first 3 coefficients
void plot_3d_grayscale(Mat Xcoef, string windowName) {
    int n = Xcoef.rows;

    Mat mins, maxs;
    min_max_by_column(Xcoef, mins, maxs);

    Mat coords = Xcoef.colRange(0,2).clone();

    for (int i = 0; i < n; i++) {
        coords.at<double>(i, 0) -= mins.at<double>(0, 0);
        coords.at<double>(i, 1) -= mins.at<double>(0, 1);
    }

    double maxX = maxs.at<double>(0, 0) - mins.at<double>(0, 0);
    double maxY = maxs.at<double>(0, 1) - mins.at<double>(0, 1);

    int width = (int)ceil(max(1.0, maxX)) + 10;
    int height= (int)ceil(max(1.0, maxY)) + 10;

    Mat img(height, width, CV_8UC1, Scalar(255));

    double minI = mins.at<double>(0, 2), maxI = maxs.at<double>(0, 2);
    double denom = (maxI - minI == 0 ? 1.0 : (maxI - minI));

    for (int i = 0; i < n; i++) {
        int x = (int)round(coords.at<double>(i, 0));
        int y = (int)round(coords.at<double>(i, 1));
        int intensity = (int)round(255.0 * (Xcoef.at<double>(i, 2) - minI) / denom);

        if (0 <= x && x < width && 0 <= y && y < height) {
            img.at<uchar>(y, x) = saturate_cast<uchar>(intensity);
        }
    }

    namedWindow(windowName, WINDOW_KEEPRATIO);
    imshow(windowName, img);
}