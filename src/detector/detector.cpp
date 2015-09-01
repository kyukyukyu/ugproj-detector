#include "detector.hpp"

#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;
namespace ugproj {

void FaceDetector::detectFaces(const Mat& frame, vector<Rect>& rects, const float scale) {
    const cv::Scalar& lowerBound = this->cfg_.skin_lower;
    const cv::Scalar& upperBound = this->cfg_.skin_upper;
    Mat ycrcb;
    Mat mask;
    Mat gray;
    Mat smallImg(cvRound(frame.rows * scale),
                 cvRound(frame.cols * scale),
                 CV_8UC1);

    cvtColor(frame, ycrcb, CV_BGR2YCrCb);
    inRange(ycrcb, lowerBound, upperBound, mask);

    cvtColor(frame, gray, COLOR_BGR2GRAY);
    equalizeHist(gray, gray);
    gray &= mask;
    resize(gray, smallImg, smallImg.size());

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
        int sourceX = r->x / scale;
        int sourceY = r->y / scale;
        int sourceWidth = r->width / scale;
        int sourceHeight = r->height / scale;
        Mat croppedMask(mask,
                Range(sourceY, sourceY + sourceHeight),
                Range(sourceX, sourceX + sourceWidth));
        double m = norm( mean(croppedMask) );
        if (m/256 < 0.8)
            continue;
        Rect new_r(sourceX, sourceY, sourceWidth, sourceHeight);
        rects.push_back(new_r);
    }
}

} // ugproj
