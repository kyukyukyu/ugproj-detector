#ifndef UGPROJ_DETECTOR_HEADER
#define UGPROJ_DETECTOR_HEADER

#include <opencv2/opencv.hpp>
#include <vector>

namespace ugproj {
    class FaceDetector {
        private:
            cv::CascadeClassifier& cascade;
        public:
            FaceDetector(cv::CascadeClassifier& cascade):
                cascade(cascade) {};
            void detectFaces(const cv::Mat& frame, std::vector<cv::Rect>& rects, const float scale);
    };
} // ugproj

#endif
