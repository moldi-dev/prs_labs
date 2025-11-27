#include <fstream>
#include <limits>
#include <opencv2/opencv.hpp>
#include "src/common/common.h"
#include "src/slider/slider.h"
#include "src/common/logger/logger.h"

using namespace cv;
using namespace std;

vector<int> ransac_algorithm(int s, vector<Point2d> points, double t, int T, int N);
Mat_<uchar> draw_line(Mat_<uchar> input_image, vector<int> params);

int main() {
    srand(time(nullptr));

    // 1. Open the input image and construct the input point set by finding
    // the positions of all black points
    Mat_<uchar> input_image = imread("assets/points_RANSAC/points1.bmp", IMREAD_GRAYSCALE);

    vector<Point2d> points;

    for (int i = 0; i < input_image.rows; i++) {
        for (int j = 0; j < input_image.cols; j++) {
            if (input_image(i, j) == 0) {
                points.push_back(Point2d(j, i));
            }
        }
    }

    // 2 and 3. Calculate the parameters 𝑁 and 𝑇 starting from the recommended values:
    // t = 10, p = 0.99, q = 0.7 and s = 2. For points1.bmp use q = 0.3
    double t = 10.0; // inlier distance threshold (pixels)
    double p = 0.99; // desired success probability
    double q = 0.3; // inlier ratio estimate for points1.bmp
    int s = 2; // minimal sample size for a line

    int N = log(1.0 - p) / log(1.0 - pow(q, s));
    int T = q * points.size();

    cout << "N = " << N << endl;
    cout << "T = " << T << endl;

    // 4. Apply the RANSAC method
    vector<int> params = ransac_algorithm(s, points, t, T, N);

    // 7. Draw the optimal line found by the method
    namedWindow("RANSAC Algorithm", WINDOW_KEEPRATIO);
    imshow("RANSAC Algorithm", draw_line(input_image, params));

    waitKey(0);

    return 0;
}

vector<int> ransac_algorithm(int s, vector<Point2d> points, double t, int T, int N) {
    vector<int> result(3, 0);

    if (points.size() < 2) {
        return result;
    }

    int best_inliers = -1;
    double best_a = 0.0, best_b = 0.0, best_c = 0.0;

    // 5. Write the correct termination conditions based on the size of the
    // consensus set and the maximum number of iterations
    for (int i = 0; i < N && best_inliers < T; i++) {

        // 4.a. Choose two different points;
        int i1 = rand() % points.size();
        int i2 = rand() % points.size();

        while (i2 == i1) {
            i2 = rand() % points.size();
        }

        Point2d p1 = points[i1];
        Point2d p2 = points[i2];

        if (p1.x == p2.x && p1.y == p2.y) {
            continue;
        }

        // 4.b. Determine the equation of the line passing through the selected points
        double a = p1.y - p2.y;
        double b = p2.x - p1.x;
        double c = p1.x * p2.y - p2.x * p1.y;

        double denom = sqrt(a * a + b * b);

        if (denom == 0.0) {
            continue;
        }

        int inliers = 0;

        // 4.c. Find the distances of each point to the line;
        // 4.d. Count the number of inliers
        for (int j = 0; j < points.size(); j++) {
            Point2d p = points[j];

            double d = abs(a * p.x + b * p.y + c) / denom;

            if (d <= t) {
                inliers++;
            }
        }

        // 4.e. Save the line parameters (a, b, c) if the current line has
        // the highest number of inliers so far
        if (inliers > best_inliers) {
            best_inliers = inliers;

            best_a = a;
            best_b = b;
            best_c = c;
        }
    }

    if (best_inliers > 0) {
        result[0] = (int)round(best_a);
        result[1] = (int)round(best_b);
        result[2] = (int)round(best_c);
    }

    return result;
}

Mat_<uchar> draw_line(Mat_<uchar> input_image, vector<int> params) {
    Mat_<uchar> result = input_image.clone();

    int a = params[0];
    int b = params[1];
    int c = params[2];

    int W = input_image.cols - 1;

    Point2d p1(0, -c / b);
    Point2d p2(W, (-a * W - c) / b);

    line(result, p1, p2, Scalar(0, 0, 255));

    return result;
}
