#include <fstream>
#include <limits>
#include <random>
#include <opencv2/opencv.hpp>
#include "src/common/common.h"
#include "src/slider/slider.h"
#include "src/common/logger/logger.h"

using namespace cv;
using namespace std;

const int NUM_CLASSES = 10; // MNIST digits 0-9
const int IMG_WIDTH = 28;
const int IMG_HEIGHT = 28;
const int NUM_FEATURES = IMG_WIDTH * IMG_HEIGHT;

struct Dataset {
    Mat X; // feature matrix: N x 784, CV_8UC1 (values 0 or 255)
    Mat y; // label vector: N x 1, CV_32SC1
};

Dataset load_images(const string& rootFolder, int maxImagesPerClass);
void train_naive_bayes(const Dataset& trainData, Mat& priors, Mat& likelihoods);
int classify_naive_bayes(const Mat& imgRow, const Mat& priors, const Mat& likelihoods);

int main() {
    // 1. Load Training Data
    cout << "[Step 1] Loading the training data" << endl;
    Dataset trainingData = load_images("./assets/images_Bayes/train", 1000);

    if (trainingData.X.empty()) {
        cerr << "Error: No training images loaded. Check the directory structure (train/0/*.png)" << endl;
        return -1;
    }

    // 2. Train Model
    // Compute priors and likelihoods using ONLY training data [cite: 1028, 1032, 1034]
    Mat priors, likelihoods;
    cout << "[Step 2] Training the Naive Bayes Classifier..." << endl;
    train_naive_bayes(trainingData, priors, likelihoods);

    // 3. Load Test Data
    cout << "[Step 3] Loading the test data" << endl;
    Dataset testData = load_images("./assets/images_Bayes/test", 800);

    if (testData.X.empty()) {
        cerr << "Error: No test images loaded. Check the directory structure (test/0/*.png)" << endl;
        return -1;
    }

    // 4. Evaluate on Test Set
    cout << "[Step 4] Evaluating on the test data..." << endl;

    Mat confusionMatrix = Mat::zeros(NUM_CLASSES, NUM_CLASSES, CV_32S);
    int correct = 0;
    int total = testData.X.rows; // Use the size of the test set

    for (int i = 0; i < total; ++i) {
        // Get the test sample and its true label
        Mat sample = testData.X.row(i);
        int trueLabel = testData.y.at<int>(i);

        // Predict using the model trained in Step 2
        int predictedLabel = classify_naive_bayes(sample, priors, likelihoods);

        // Update stats
        if (predictedLabel == trueLabel) {
            correct++;
        }

        // Update Confusion Matrix
        confusionMatrix.at<int>(trueLabel, predictedLabel)++;
    }

    // 5. Results
    // The error rate is the fraction of misclassified test instances
    double accuracy = (double)correct / total * 100.0;
    double errorRate = 100.0 - accuracy;

    cout << "Total Test Images: " << total << endl;
    cout << "Accuracy: " << accuracy << "%" << endl;
    cout << "Error Rate: " << errorRate << "%" << endl;
    cout << "Confusion Matrix (Row=Real, Col=Pred):" << endl;
    cout << confusionMatrix << endl;

    return 0;
}

Dataset load_images(const string& rootFolder, int maxImagesPerClass) {
    Dataset data;
    cout << "Loading images..." << endl;

    for (int c = 0; c < NUM_CLASSES; c++) {
        int loadedCount = 0;

        for (int index = 0; index < maxImagesPerClass; index++) {
            // Format string like "train/0/000001.png"
            char filepath[256];
            sprintf(filepath, "%s/%d/%06d.png", rootFolder.c_str(), c, index);

            // Load as grayscale
            Mat img = imread(filepath, IMREAD_GRAYSCALE);

            if (img.empty()) {
                continue;
            }

            // Binarize the image
            Mat binaryImg;
            threshold(img, binaryImg, 127, 255, THRESH_BINARY);

            // Reshape to 1 row, NUM_FEATURES columns
            Mat featureRow = binaryImg.reshape(1, 1);

            // Add to dataset
            data.X.push_back(featureRow);
            data.y.push_back(c);

            loadedCount++;
        }

        cout << "Loaded " << loadedCount << " images for class " << c << endl;
    }

    // Convert X to CV_8U and y to CV_32S for consistency
    data.X.convertTo(data.X, CV_8U);
    data.y.convertTo(data.y, CV_32S);

    return data;
}

void train_naive_bayes(const Dataset& trainData, Mat& priors, Mat& likelihoods) {
    int n = trainData.X.rows; // Total number of instances

    // Initialize priors (1 x C) and likelihoods (C x d)
    priors = Mat::zeros(NUM_CLASSES, 1, CV_64F);
    likelihoods = Mat::zeros(NUM_CLASSES, NUM_FEATURES, CV_64F);

    // Temp array to count instances per class (n_i)
    vector<int> classCounts(NUM_CLASSES, 0);

    // 1. Accumulate counts
    for (int i = 0; i < n; i++) {
        int label = trainData.y.at<int>(i);
        classCounts[label]++;

        // Scan features
        const uchar* rowPtr = trainData.X.ptr<uchar>(i);

        for (int j = 0; j < NUM_FEATURES; j++) {
            // If feature is 255 (white pixel), increment count for (class, feature)
            if (rowPtr[j] == 255) {
                likelihoods.at<double>(label, j) += 1.0;
            }
        }
    }

    // 2. Compute Probabilities
    for (int c = 0; c < NUM_CLASSES; c++) {
        // Calculate Prior P(C=c) = n_c / n
        if (n > 0) {
            priors.at<double>(c) = (double)classCounts[c] / n;
        }

        // Calculate Likelihoods with Laplace Smoothing
        // p(x_j=255 | C=c) = (count + 1) / (n_c + |C|)
        double denominator = classCounts[c] + NUM_CLASSES;

        for (int j = 0; j < NUM_FEATURES; j++) {
            double count = likelihoods.at<double>(c, j);
            double prob = (count + 1.0) / denominator;
            likelihoods.at<double>(c, j) = prob;
        }
    }
}

int classify_naive_bayes(const Mat& imgRow, const Mat& priors, const Mat& likelihoods) {
    double maxLogPosterior = -DBL_MAX; // Initialize with lowest double
    int bestClass = -1;

    // Loop through all classes to find the best fit
    for (int c = 0; c < NUM_CLASSES; c++) {
        // Start with log(Prior)
        double logPosterior = log(priors.at<double>(c));

        // Sum log likelihoods for features
        const uchar* pixelPtr = imgRow.ptr<uchar>(0);

        for (int j = 0; j < NUM_FEATURES; j++) {
            double probWhite = likelihoods.at<double>(c, j);

            // Check pixel value in test image (T_j)
            if (pixelPtr[j] == 255) {
                // If pixel is white, add log(P(x=255|c))
                logPosterior += log(probWhite);
            }

            else {
                // If pixel is black (0), add log(P(x=0|c))
                // P(x=0|c) = 1 - P(x=255|c)
                logPosterior += log(1.0 - probWhite);
            }
        }

        // Find argmax
        if (logPosterior > maxLogPosterior) {
            maxLogPosterior = logPosterior;
            bestClass = c;
        }
    }

    return bestClass;
}