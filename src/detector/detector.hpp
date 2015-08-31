#ifndef UGPROJ_DETECTOR_HEADER
#define UGPROJ_DETECTOR_HEADER

#include <opencv2/opencv.hpp>
#include <vector>

namespace ugproj {

class FaceDetector {
  public:
    FaceDetector(cv::CascadeClassifier& cascade) : cascade(cascade) {};
    void detectFaces(const cv::Mat& frame, std::vector<cv::Rect>& rects,
                     const float scale);
  private:
    cv::CascadeClassifier& cascade;
};

} // ugproj

#endif
