#include <fstream>
#include <limits>
#include <random>
#include <opencv2/opencv.hpp>
#include <iomanip>
#include "src/common/common.h"
#include "src/slider/slider.h"
#include "src/common/logger/logger.h"

using namespace cv;
using namespace std;

struct Dataset {
    Mat x; // Augmented feature vector [1, x1, x2]
    int y; // Class label (-1 or +1)
};

vector<Dataset> build_data_set(Mat img);
Mat train_online_perceptron(Mat img, vector<Dataset> trainingSet, int maxIter, double eLimit);
Mat draw_decision(Mat img, Mat w);

int main() {
    // Read the input image
    string filename = "./assets/images_Perceptron/test00.bmp";
    Mat img = imread(filename, IMREAD_COLOR);

    if (img.empty()) {
        cout << "Error loading image! Ensure " << filename << " exists" << endl;
        return -1;
    }

    // Build the training set
    vector<Dataset> trainingSet = build_data_set(img);
    cout << "Training set size: " << trainingSet.size() << endl;

    if (trainingSet.empty()) {
        cout << "No red or blue points found. Exiting..." << endl;
        return -1;
    }

    // Train the classifier
    // Parameters: eta = 10^-4, Elimit = 10^-5, max_iter = 10^5
    train_online_perceptron(img, trainingSet, 100000, 0.00001);

    return 0;
}

vector<Dataset> build_data_set(Mat img) {
    vector<Dataset> result;

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            Vec3b pixel = img.at<Vec3b>(i, j);
            Dataset d;

            // Red pixel
            if (pixel[2] > 200 && pixel[0] < 50 && pixel[1] < 50) {
                d.x = (Mat_<double>(1, 3) << 1.0, (double)j, (double)i); // [1, col, row]
                d.y = 1;
                result.push_back(d);
            }

            // Blue pixel
            else if (pixel[0] > 200 && pixel[2] < 50 && pixel[1] < 50) {
                d.x = (Mat_<double>(1, 3) << 1.0, (double)j, (double)i); // [1, col, row]
                d.y = -1;
                result.push_back(d);
            }
        }
    }

    return result;
}

Mat train_online_perceptron(Mat img, vector<Dataset> trainingSet, int maxIter, double eLimit) {
    // Initialize the augmented weight vector w = [w0, w1, w2]
    Mat w = (Mat_<double>(1, 3) << 0.1, 0.1, 0.1);

    // Learning rates
    double eta = 0.0001; // Standard learning rate
    double etaBias = 0.01; // Larger learning rate for bias (w0)

    double e; // Error rate
    int errorCount;
    double z;
    char name[256];
    Mat bigImg;
    Mat result;

    cout << fixed << setprecision(6);
    cout << "Learning rate (features) = " << eta << ", (bias) = " << etaBias << endl;

    for (int iter = 0; iter < maxIter; iter++) {
        cout << "Iteration " << iter << endl;
        errorCount = 0;

        for (int i = 0; i < trainingSet.size(); i++) {
            // Calculate z = w^T * x
            z = w.dot(trainingSet[i].x);

            // Extract values
            double w0 = w.at<double>(0, 0);
            double w1 = w.at<double>(0, 1);
            double w2 = w.at<double>(0, 2);

            double x0 = trainingSet[i].x.at<double>(0, 0);
            double x1 = trainingSet[i].x.at<double>(0, 1);
            double x2 = trainingSet[i].x.at<double>(0, 2);

            int y = trainingSet[i].y;

            cout << "i=" << i << ": w=[" << w0 << " " << w1 << " " << w2 << "] "
                 << "xi=[" << (int)x0 << " " << (int)x1 << " " << (int)x2 << "] "
                 << "yi = " << y << " zi=" << z << endl;

            // Update the weights in case of misclassification
            if (z * y <= 0) {
                cout << "wrong" << endl;

                string op = (y > 0) ? " + " : " - ";

                cout << "update "
                     << "w0 = w0" << op << etaBias << "*dddd" << (int)x0 << ", "
                     << "w1 = w1" << op << eta << "*" << (int)x1 << ", "
                     << "w2 = w2" << op << eta << "*" << (int)x2 << endl;

                // Update features w1, w2 using eta
                w.at<double>(0, 1) += eta * x1 * y;
                w.at<double>(0, 2) += eta * x2 * y;

                // Update bias w0 with specialized learning rate etaBias
                w.at<double>(0, 0) += etaBias * x0 * y;

                errorCount++;
            }
        }

        // Normalize the error
        e = (double)errorCount / trainingSet.size();

        // Check the stopping condition
        if (e < eLimit) {
            cout << "Converged at iteration: " << iter << endl;
            break;
        }

        // Visualize the results
        result = draw_decision(img, w);

        // Resizing needs to be done on Fedora 42 because OpenCV for some reason displays the 40 * 40 image and it is very small
        resize(result, bigImg, Size(), 10.0, 10.0, INTER_NEAREST);

        sprintf(name, "Iteration %d", iter);

        namedWindow(name, WINDOW_KEEPRATIO);
        imshow(name, bigImg);

        waitKey(0);
        destroyWindow(name);
    }

    cout << "Final error rate: " << e << endl;
    cout << "Final weights: " << w << endl;

    resize(result, bigImg, Size(), 10.0, 10.0, INTER_NEAREST);

    namedWindow("Final image", WINDOW_KEEPRATIO);
    imshow("Final image", bigImg);

    waitKey(0);
    destroyWindow(name);

    return w;
}

Mat draw_decision(Mat img, Mat w) {
    Mat result = img.clone();

    double w0 = w.at<double>(0, 0);
    double w1 = w.at<double>(0, 1);
    double w2 = w.at<double>(0, 2);

    // Ensure no division by 0 if the line is vertical
    if (abs(w2) < 1e-6) {
        cout << "Warning: Vertical line, cannot draw using y = mx + b method" << endl;
        return result;
    }

    // w0 + w1 * x + w2 * y = 0  =>  y = -(w0 + w1 * x) / w2
    Point p1, p2;

    // Point at left edge (x = 0)
    p1.x = 0;
    p1.y = (int)((-w0 - w1 * 0) / w2);

    // Point at right edge (x = width)
    p2.x = img.cols;
    p2.y = (int)((-w0 - w1 * img.cols) / w2);

    // Draw a green line
    line(result, p1, p2, Scalar(0, 255, 0), 1);

    return result;
}
