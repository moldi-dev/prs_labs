#include <fstream>
#include <limits>
#include <random>
#include <opencv2/opencv.hpp>
#include "src/common/common.h"
#include "src/slider/slider.h"
#include "src/common/logger/logger.h"

using namespace cv;
using namespace std;

Mat_<int> convert_image_to_points_2d(Mat img);
void apply_k_means(Mat_<int> points, int k, Mat src);
void initialize(Mat_<int> points, int k, Mat_<int>& centroids);
void display_centroids(Mat_<int> centroids, Mat src, string name);
double find_euclidean_distance(Mat_<int> p1, Mat_<int> p2);

int main() {
    Mat img = imread("./assets/images_Kmeans/points4.bmp", IMREAD_GRAYSCALE);
    Mat_<int> points = convert_image_to_points_2d(img);

    namedWindow("Initial image", WINDOW_KEEPRATIO);
    imshow("Initial image", img);

    apply_k_means(points, 3, img);

    waitKey(0);

    return 0;
}

Mat_<int> convert_image_to_points_2d(Mat img) {
    vector<Point> points;

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (img.at<uchar>(i, j) == 0) {
                points.push_back(Point(j, i));
            }
        }
    }

    Mat_<int> matPoints(points.size(), 2);

    for (int i = 0; i < points.size(); i++) {
        matPoints(i, 0) = points[i].x;
        matPoints(i, 1) = points[i].y;
    }

    return matPoints;
}

void initialize(Mat_<int> points, int k, Mat_<int>& centroids) {
    default_random_engine gen;
    uniform_int_distribution<int> distribution(0, points.rows - 1);

    for (int i = 0; i < k; i++) {
        int idx = distribution(gen);
        points.row(idx).copyTo(centroids.row(i));
    }
}

void display_centroids(Mat_<int> centroids, Mat src, string name) {
    for (int i = 0; i < centroids.rows; i++) {
        Point center(centroids(i, 0), centroids(i, 1));
        circle(src, center, 6, 0, -1);
    }

    namedWindow(name, WINDOW_KEEPRATIO);
    imshow(name, src);
}

double find_euclidean_distance(Mat_<int> p1, Mat_<int> p2) {
    double x1 = p1(0, 0);
    double y1 = p1(0, 1);
    double x2 = p2(0, 0);
    double y2 = p2(0, 1);

    double dx = x1 - x2;
    double dy = y1 - y2;

    return sqrt(dx * dx + dy * dy);
}

void apply_k_means(Mat_<int> points, int k, Mat src) {
    int d = points.cols;
    Mat_<int> centroids(k,d);
    centroids.setTo(0);

    initialize(points, k, centroids);
    display_centroids(centroids, src.clone(), "Centroids after initialization");

    bool change = true;
    int maxIterations = 100;
    int iteration = 0;
    Mat_<int> labels(points.rows, 1);

    while (change && iteration < maxIterations) {
        change = false;

        for (int i = 0; i < points.rows; i++) {
            double minDist = DBL_MAX;
            int bestCluster = -1;

            for (int j = 0; j < k; j++) {
                double dist = find_euclidean_distance(points.row(i), centroids.row(j));

                if (dist < minDist) {
                    minDist = dist;
                    bestCluster = j;
                }
            }

            if (labels(i, 0) != bestCluster) {
                labels(i, 0) = bestCluster;
                change = true;
            }
        }

        Mat_<int> newCentroids(k, d);
        Mat_<int> counts(k, 1);
        newCentroids.setTo(0);
        counts.setTo(0);

        for (int i = 0; i < points.rows; i++) {
            int cluster = labels(i, 0);
            newCentroids.row(cluster) += points.row(i);
            counts(cluster, 0)++;
        }

        for (int j = 0; j < k; j++) {
            if (counts(j, 0) > 0) {
                newCentroids.row(j) /= counts(j, 0);
            }
        }

        centroids = newCentroids;
        iteration++;

        string name = "Centroids after iteration " + to_string(iteration);

        display_centroids(centroids, src.clone(), name);
    }

    Mat clustered(src.rows, src.cols, CV_8UC3, Scalar(255, 255, 255));

    vector<Vec3b> colors(k);
    RNG rng(12345);

    for (int i = 0; i < k; i++) {
        colors[i] = Vec3b(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    }

    for (int i = 0; i < points.rows; i++) {
        int x = points(i, 0);
        int y = points(i, 1);
        int lbl = labels(i, 0);
        clustered.at<Vec3b>(y, x) = colors[lbl];
    }

    for (int i = 0; i < k; i++) {
        Point c(centroids(i, 0), centroids(i, 1));
        circle(clustered, c, 6, Scalar(0, 0, 0), -1);
    }

    namedWindow("Clustered Image", WINDOW_KEEPRATIO);
    imshow("Clustered Image", clustered);
}