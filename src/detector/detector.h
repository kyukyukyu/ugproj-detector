#ifndef UGPROJ_DETECTOR_HEADER
#define UGPROJ_DETECTOR_HEADER

#include <opencv2/opencv.hpp>
#include <vector>
#include "../structure.h"

namespace ugproj {

class FaceDetector {
  public:
    FaceDetector(cv::CascadeClassifier& cascade,
                 const Configuration::DetectionSection& cfg) :
        cascade(cascade), cfg_(cfg) {};
    void detectFaces(const cv::Mat& frame, std::vector<cv::Rect>& rects);
  private:
    cv::CascadeClassifier& cascade;
    const Configuration::DetectionSection& cfg_;
};

} // ugproj

#endif
