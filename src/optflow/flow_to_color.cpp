#include "flow_to_color.hpp"
#include "colorcode.h"

#include <cmath>
#include <limits>

#define UNKNOWN_FLOW_THRESHOLD 1e9
#define INIT_VAL_MAX -999
#define INIT_VAL_MIN 999

using namespace std;
using namespace opticalflow;

void ugproj::flowToColor(MCImageDoubleX &vx, MCImageDoubleX &vy,
                         cv::Mat& img) {
    const int width = vx.width();
    const int height = vy.height();
    
    Eigen::Matrix2d u = vx;
    Eigen::Matrix2d v = vy;
    double maxU = INIT_VAL_MAX, maxV = INIT_VAL_MAX;
    double minU = INIT_VAL_MIN, minV = INIT_VAL_MIN;
    double maxRad = -1;
    
    // fix unknown flow
    Eigen::ArrayXXi idxUnknown = ((u.array().abs() > UNKNOWN_FLOW_THRESHOLD) ||
                                  (v.array().abs() > UNKNOWN_FLOW_THRESHOLD)).cast<int>();
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if (idxUnknown.coeffRef(i, j) == 0)
                u.coeffRef(i, j) = v.coeffRef(i, j) = 0;
        }
    }
    
    maxU = u.maxCoeff();
    minU = u.minCoeff();
    
    maxV = v.maxCoeff();
    minV = v.minCoeff();
    
    Eigen::Matrix2d rad = (u.array().square() + v.array().square()).cwiseSqrt();
    maxRad = rad.maxCoeff();
    
    double eps = numeric_limits<double>::epsilon();
    u /= maxRad + eps;
    v /= maxRad + eps;
    
    cv::Scalar color;
    uchar pix[3];
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if (idxUnknown.coeffRef(i, j) == 0) {
                // unknown flow
                color = CV_RGB(0, 0, 0);
            } else {
                // compute color
                computeColor(u.coeffRef(i, j), v.coeffRef(i, j), pix);
                color = CV_RGB(pix[0], pix[1], pix[2]);
            }
            img.at<cv::Scalar>(i, j) = color;
        }
    }
}