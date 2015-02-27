#ifndef UGPROJ_OPTFLOW_MANAGER_HEADER
#define UGPROJ_OPTFLOW_MANAGER_HEADER

#include "../structure.hpp"

#include <opencv2/opencv.hpp>
#include <vector>

namespace ugproj {
    class OpticalFlowManager {
        public:
            typedef std::pair<OptFlowArray*, OptFlowArray*> flow_t;

        private:
            std::vector<flow_t*> flowVect;
        public:
            int flowWidth;
            int flowHeight;

            OpticalFlowManager(int flowWidth, int flowHeight);
            ~OpticalFlowManager();
            void append(flow_t* flow);
            const cv::Vec2d getFlowAt(const temp_idx_t startTempIndex, const temp_idx_t endTempIndex, int x, int y) const;
    };
} // ugproj

#endif
