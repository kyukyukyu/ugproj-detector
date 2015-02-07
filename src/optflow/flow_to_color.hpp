#ifndef UGPROJ_OPTFLOW_FLOW2COLOR_HEADER
#define UGPROJ_OPTFLOW_FLOW2COLOR_HEADER

#include "../celiu-optflow/optical_flow.h"

#include <opencv2/opencv.hpp>

namespace ugproj {
    void flowToColor(opticalflow::MCImageDoubleX& vx,
                     opticalflow::MCImageDoubleX& vy,
                     cv::Mat& colorMat);
}

#endif