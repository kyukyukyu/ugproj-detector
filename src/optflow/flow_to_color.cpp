#include "flow_to_color.hpp"
#include "colorcode.h"

#include <opencv2/opencv.hpp>

#include <cmath>
#include <limits>

#define UNKNOWN_FLOW_THRESHOLD 1e9
#define INIT_VAL_MAX -999
#define INIT_VAL_MIN 999

using namespace std;
using namespace opticalflow;

void ugproj::flowToColor(MCImageDoubleX &vx, MCImageDoubleX &vy,
                         cv::Mat& img) {
    typedef Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> ArrayDXX;

    const int width = vx.width();
    const int height = vy.height();

    // copy array
    ArrayDXX u(height, width), v(height, width);
    for (int c = 0; c < width; ++c) {
        for (int r = 0; r < height; ++r) {
            u.coeffRef(r, c) = vx(c, r, 0);
            v.coeffRef(r, c) = vy(c, r, 0);
        }
    }

    // fix unknown flow
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> idxUnknown(height, width);
    idxUnknown = (u.abs() > UNKNOWN_FLOW_THRESHOLD) ||
                 (v.abs() > UNKNOWN_FLOW_THRESHOLD);
    for (int c = 0; c < height; ++c) {
        for (int r = 0; r < width; ++r) {
            if (idxUnknown.coeffRef(r, c))
                u.coeffRef(r, c) = v.coeffRef(r, c) = 0;
        }
    }

    // get mix/max value (only for debug)
    double maxU = INIT_VAL_MAX, maxV = INIT_VAL_MAX;
    double minU = INIT_VAL_MIN, minV = INIT_VAL_MIN;
    maxU = u.maxCoeff();
    minU = u.minCoeff();

    maxV = v.maxCoeff();
    minV = v.minCoeff();

    // get maximum norm
    double maxRad = -1;
    ArrayDXX rad = (u.square() + v.square()).sqrt();
    maxRad = rad.maxCoeff();

    // normalize vectors
    double eps = numeric_limits<double>::epsilon();
    u /= maxRad + eps;
    v /= maxRad + eps;

    // initialize flow image matrix
    img.create(height, width, CV_8UC3);

    // fill flow image
    cv::Scalar color;
    cv::Point point;
    uchar pix[3];
    for (int c = 0; c < width; ++c) {
        for (int r = 0; r < height; ++r) {
            if (idxUnknown.coeffRef(r, c)) {
                // unknown flow
                color = CV_RGB(0, 0, 0);
            } else {
                computeColor(u.coeffRef(r, c), v.coeffRef(r, c), pix);
                color = CV_RGB(pix[2], pix[1], pix[0]);
            }
            point = cv::Point(c, r);
            cv::line(img, point, point, color);
        }
    }
}
