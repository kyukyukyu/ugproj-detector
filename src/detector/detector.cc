#include "detector.h"

#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;
namespace ugproj {

void FaceDetector::detectFaces(const Mat& frame, vector<Rect>& rects) {
    const cv::Scalar& lowerBound = this->cfg_.skin_lower;
    const cv::Scalar& upperBound = this->cfg_.skin_upper;
    Mat ycrcb;
    Mat mask;
    Mat gray;

    cvtColor(frame, ycrcb, CV_BGR2YCrCb);
    inRange(ycrcb, lowerBound, upperBound, mask);

    cvtColor(frame, gray, COLOR_BGR2GRAY);
    equalizeHist(gray, gray);
    gray &= mask;

    vector<Rect> facesInGray;
    cascade.detectMultiScale(
            gray,
            facesInGray,
            this->cfg_.scale,
            2,
            0 | CASCADE_SCALE_IMAGE,
            Size(30, 30));

    for (vector<Rect>::const_iterator r = facesInGray.begin();
         r != facesInGray.end();
         ++r) {
        const cv::Rect& roi = *r;
        Mat croppedMask(mask, roi);
        double m = norm( mean(croppedMask) );
        if (m/256 < 0.8)
            continue;
        rects.push_back(roi);
    }
}

} // ugproj
