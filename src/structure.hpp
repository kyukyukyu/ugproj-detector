#ifndef UGPROJ_STRUCTURE_HEADER
#define UGPROJ_STRUCTURE_HEADER

#include "celiu-optflow/optical_flow.h"

#include <opencv2/opencv.hpp>
#include <vector>

namespace ugproj {

    typedef opticalflow::MCImageDoubleX OptFlowArray;

    class FaceCandidate {
        public:
            const unsigned long framePos;
            const cv::Rect rect;
            const cv::Mat image;

            FaceCandidate(unsigned long& framePos, const cv::Rect& rect, cv::Mat& image):
                framePos(framePos), rect(rect), image(image) {};
    };

    class Face {
        private:
            std::vector<FaceCandidate> candidates;

        public:
            typedef unsigned int id_type;
            const id_type id;
            Face(id_type id): id(id) {};
            Face(id_type id, FaceCandidate& candidate);
            void addCandidate(FaceCandidate& candidate);
    };

} // ugproj

#endif
