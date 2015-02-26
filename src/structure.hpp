#ifndef UGPROJ_STRUCTURE_HEADER
#define UGPROJ_STRUCTURE_HEADER

#include "celiu-optflow/optical_flow.h"

#include <opencv2/opencv.hpp>
#include <vector>

namespace ugproj {

    typedef unsigned long temp_idx_t;
    typedef unsigned int face_id_t;
    typedef opticalflow::MCImageDoubleX OptFlowArray;

    class FaceCandidate {
        public:
            const temp_idx_t frameIndex;
            const cv::Rect rect;
            const cv::Mat image;
            face_id_t faceId;

            FaceCandidate(temp_idx_t& frameIndex, const cv::Rect& rect, cv::Mat& image):
                frameIndex(frameIndex), rect(rect), image(image), faceId(0) {};
    };

    class Face {
        private:
            std::vector<FaceCandidate> candidates;

        public:
            typedef face_id_t id_type;
            const id_type id;
            Face(id_type id): id(id) {};
            Face(id_type id, FaceCandidate& candidate);
            void addCandidate(FaceCandidate& candidate);
    };

} // ugproj

#endif
