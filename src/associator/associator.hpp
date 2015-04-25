#ifndef UGPROJ_ASSOCIATOR_HEADER
#define UGPROJ_ASSOCIATOR_HEADER

#define UGPROJ_ASSOCIATOR_SIFT_TRIAL_COUNT 10

#include "../structure.hpp"
#include "../optflow/manager.hpp"

#include <opencv2/opencv.hpp>
#include <vector>

namespace ugproj {
    class FaceAssociator {
        public:
            typedef std::vector<FaceCandidate*> fc_v;

        protected:
            std::vector<Face>& faces;
            fc_v& prevCandidates;
            fc_v& nextCandidates;
            double **prob;
            double threshold;

            void matchCandidates();

        public:
            FaceAssociator(
                    std::vector<Face>& faces,
                    fc_v& prevCandidates,
                    fc_v& nextCandidates,
                    double threshold):
                faces(faces),
                prevCandidates(prevCandidates),
                nextCandidates(nextCandidates),
                threshold(threshold) {
                    // probability array allocation
                    fc_v::size_type rows, cols;
                    rows = prevCandidates.size();
                    cols = nextCandidates.size();

                    prob = new double *[rows];
                    while (rows--) {
                        prob[rows] = new double[cols];
                    }
                }
            virtual ~FaceAssociator() {
                fc_v::size_type row = prevCandidates.size();
                while (row--) {
                    delete[] prob[row];
                }
                delete[] prob;
            }
            void associate();
            virtual void calculateProb() = 0;
    };

    class IntersectionFaceAssociator : public FaceAssociator {
        public:
            IntersectionFaceAssociator(
                    std::vector<Face>& faces,
                    fc_v& prevCandidates,
                    fc_v& nextCandidates,
                    double threshold):
                FaceAssociator(faces, prevCandidates, nextCandidates,
                               threshold) {};
            void calculateProb();
    };

    class OpticalFlowFaceAssociator : public FaceAssociator {
        private:
            OpticalFlowManager& flowManager;
            const temp_idx_t prevFramePos;
            const temp_idx_t nextFramePos;

        public:
            OpticalFlowFaceAssociator(
                    std::vector<Face>& faces,
                    fc_v& prevCandidates,
                    fc_v& nextCandidates,
                    OpticalFlowManager& flowManager,
                    const temp_idx_t prevFramePos,
                    const temp_idx_t nextFramePos,
                    double threshold);
            void calculateProb();
    };

    class SiftFaceAssociator : public FaceAssociator {
        private:
            const cv::Mat& prevFrame;
            const cv::Mat& nextFrame;
            std::vector<cv::DMatch> matches;
            std::vector<cv::KeyPoint> keypointsA;
            std::vector<cv::KeyPoint> keypointsB;
            cv::Mat descA;
            cv::Mat descB;
            std::vector<cv::Rect> bestFitBoxes;
            void computeBestFitBox(fc_v::size_type queryIdx,
                                   cv::Rect& bestFitBox,
                                   std::vector<cv::DMatch>* bestMatches);
            void computeMatchMask(const cv::Rect& beforeRect, cv::Mat& matchMask);
            void computeFitBox(const cv::DMatch& match1,
                               const cv::DMatch& match2,
                               const std::vector<cv::KeyPoint>& keypointsA,
                               const std::vector<cv::KeyPoint>& keypointsB,
                               const cv::Rect& beforeRect,
                               cv::Rect& fitBox) const;

        public:
            SiftFaceAssociator(std::vector<Face>& faces,
                               fc_v& prevCandidates,
                               fc_v& nextCandidates,
                               const cv::Mat& prevFrame,
                               const cv::Mat& nextFrame,
                               double threshold);
            void calculateProb();
            void visualize(cv::Mat& img);
    };
} // ugproj

#endif
