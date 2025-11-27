#include <fstream>
#include <limits>
#include <opencv2/opencv.hpp>
#include "src/common/common.h"
#include "src/slider/slider.h"
#include "src/common/logger/logger.h"

using namespace cv;
using namespace std;

vector<Point2d> readPointsFile(string filePath);
void drawCross(Mat img, int cx, int cy, int halfSize = 3, int thickness = 1, uchar color = 0);
Mat drawPointsImage(vector<Point2d> points);

int main() {
    Logger::init();

    // Choose which file to open
    string filepath = "assets/points_LeastSquares/points1.txt";
    vector<Point2d> points = readPointsFile(filepath);

    if (points.empty()) {
        ERROR("No points to display");
        Logger::destroy();
        return -1;
    }

    Mat result = drawPointsImage(points);

    // Show result
    imshow("Points", result);
    waitKey(0);

    Logger::destroy();
    return 0;
}

vector<Point2d> readPointsFile(string filepath) {
    ifstream fin(filepath);
    vector<Point2d> pts;

    if (!fin) {
        ERROR("Failed to open points file: {}", filepath);
        return pts;
    }

    int n = 0;
    fin >> n;

    if (!fin || n <= 0) {
        ERROR("Invalid or empty first line in points file: {}", filepath);
        return pts;
    }

    pts.reserve(n);

    float x, y;
    int read = 0;

    while (fin >> x >> y) {
        pts.emplace_back(x, y);
        read++;
        if (read >= n) break;
    }

    if (read != n) {
        WARN("Expected {} points, read {}", n, read);
    }

    INFO("Loaded {} point(s) from {}", pts.size(), filepath);
    return pts;
}

void drawCross(Mat img, int cx, int cy, int halfSize, int thickness, uchar color) {
    line(img, Point(cx - halfSize, cy), Point(cx + halfSize, cy), color, thickness, LINE_AA);
    line(img, Point(cx, cy - halfSize), Point(cx, cy + halfSize), color, thickness, LINE_AA);
}

Mat drawPointsImage(vector<Point2d> points) {
    int W = 500, H = 500, pad = 20;
    Mat canvas(H, W, CV_8UC1, Scalar(255));

    // Find min/max for normalization
    double minX = numeric_limits<double>::max();
    double minY = numeric_limits<double>::max();
    double maxX = numeric_limits<double>::lowest();
    double maxY = numeric_limits<double>::lowest();

    for (auto p : points) {
        minX = min(minX, p.x);
        minY = min(minY, p.y);
        maxX = max(maxX, p.x);
        maxY = max(maxY, p.y);
    }

    double spanX = maxX - minX;
    double spanY = maxY - minY;
    if (spanX == 0) spanX = 1;
    if (spanY == 0) spanY = 1;

    double scaleX = (W - 2.0 * pad) / spanX;
    double scaleY = (H - 2.0 * pad) / spanY;
    double scale = min(scaleX, scaleY);

    // Draw each point
    for (auto p : points) {
        int xImg = (int)(pad + (p.x - minX) * scale);
        int yImg = (int)(H - pad - (p.y - minY) * scale); // flip y

        if (xImg >= 0 && xImg < W && yImg >= 0 && yImg < H) {
            drawCross(canvas, xImg, yImg, 3, 1, 0);
        }
    }

    return canvas;
}