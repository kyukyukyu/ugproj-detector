#ifndef UGPROJ_OPTFLOW_FLOW2COLOR_HEADER
#define UGPROJ_OPTFLOW_FLOW2COLOR_HEADER

#include "../structure.hpp"

#include <opencv2/opencv.hpp>

namespace ugproj {
    void flowToColor(ugproj::OpticalFlow& vx,
                     ugproj::OpticalFlow& vy,
                     cv::Mat& colorMat);
}

#endif
