#ifndef UGPROJ_ASSOCIATOR_HEADER
#define UGPROJ_ASSOCIATOR_HEADER

#define UGPROJ_ASSOCIATOR_SIFT_TRIAL_COUNT 70
#define UGPROJ_ASSOCIATOR_SIFT_SCALE_THRESHOLD 1.25
#define UGPROJ_ASSOCIATOR_SIFT_RADIUS_SMALL_THRESHOLD 60
#define UGPROJ_ASSOCIATOR_SIFT_RADIUS_BIG_THRESHOLD 30
#define UGPROJ_ASSOCIATOR_SIFT_RADIUS_THRESHOLD 30
#define UGPROJ_ASSOCIATOR_SIFT_INLIER_THRESHOLD 0.03

#define PI  3.14159265
#define LINEAR_TRANSFORM    1
#define SIMILARITY_TRANSFORM    2

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
            struct Fit {
                cv::Rect queryBox;
                cv::Rect box;
                cv::RotatedRect rotatedBox;
                cv::Point q1;
                cv::Point q2;
                cv::Point t1;
                cv::Point t2;
                Eigen::VectorXd matL; // linear transformation matrix
                Eigen::VectorXd matS; // similarity transformation matrix
                std::vector<cv::DMatch> matches;
                int num_inlier;
                double inlier_ratio;
                inline int num_keyp_former() const {
                    return this->matches.size();
                }
                inline double get_inlier_ratio() const {
                    return inlier_ratio;
                }
            };
            const cv::Mat& prevFrame;
            const cv::Mat& nextFrame;
            std::vector<cv::KeyPoint> keypointsA;
            std::vector<cv::KeyPoint> keypointsB;
            cv::Mat descA;
            cv::Mat descB;
            std::vector<Fit> bestFits;
            int transformation;
            void computeBestFitBox(fc_v::size_type queryIdx,
                                   Fit* bestFit);
            void computeMatchMask(const cv::Rect& beforeRect, cv::Mat& matchMask);
            bool computeFitBox(const cv::DMatch& match1,
                               const cv::DMatch& match2,
                               const std::vector<cv::KeyPoint>& keypointsA,
                               const std::vector<cv::KeyPoint>& keypointsB,
                               const cv::Rect& beforeRect,
                               Fit* fitCandidate) const;
            double calculateInlierRatio(Fit& fitCandidate,
                                        const std::vector<cv::DMatch>& matches,
                                        const fc_v::size_type fit_index);
            inline cv::Scalar color_for(const fc_v::size_type cdd_index);
            void draw_fit_candidate(const std::vector<cv::DMatch>& matches, cv::Point* center, Fit& fitCandidate, const fc_v::size_type cdd_index);
            void draw_best_fit(const fc_v::size_type cdd_index,
                               cv::Mat* match_img);
            void draw_next_candidates(const fc_v::size_type cdd_index,
                cv::Mat* next_frame);
            void draw_inlier_edge(cv::Mat* next_frame, const std::vector<cv::DMatch>& matches, cv::Point* center, cv::Point* matched, int radius);

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
