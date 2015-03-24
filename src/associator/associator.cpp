#include "associator.hpp"
#include "../optflow/manager.hpp"

#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/Dense>
#include <Eigen/LU>

#include <boost/random.hpp>

#include <cstring>
#include <cstdlib>
#include <time.h>
#include <utility>

using namespace ugproj;
using namespace std;
using Eigen::MatrixXd;
using namespace Eigen;
using namespace boost;

int result_cnt = 0;

void FaceAssociator::matchCandidates() {
    typedef fc_v::size_type size_type;

    const size_type
        prevSize = prevCandidates.size(), nextSize = nextCandidates.size();

    for (size_type j = 0; j < nextSize; ++j) {
        double max = -1;
        size_type maxRow;

        for (size_type i = 0; i < prevSize; ++i) {
            if (prob[i][j] > max && prob[i][j] > threshold) {
                max = prob[i][j];
                maxRow = i;
            }
        }

        vector<Face>::size_type faceId;
        if (max > 0) {
            faceId = prevCandidates[maxRow]->faceId;
        } else {
            faceId = faces.size() + 1;
            faces.push_back(Face(faceId));
        }
        nextCandidates[j]->faceId = faceId;
        faces[faceId - 1].addCandidate(*nextCandidates[j]);
    }
}

void FaceAssociator::associate() {
    calculateProb();
    matchCandidates();
}

void IntersectionFaceAssociator::calculateProb() {
    typedef fc_v::size_type size_type;

    const size_type
        prevSize = prevCandidates.size(), nextSize = nextCandidates.size();

    for (size_type i = 0; i < prevSize; ++i) {
        const cv::Rect& rectI = prevCandidates[i]->rect;
        for (size_type j = 0; j < nextSize; ++j) {
            const cv::Rect& rectJ = nextCandidates[j]->rect;
            cv::Rect intersect = rectI & rectJ;
            int intersectArea = intersect.area();
            int unionArea =
                rectI.area() + rectJ.area() - intersectArea;
            prob[i][j] = (double)intersectArea / unionArea;
        }
    }
}

OpticalFlowFaceAssociator::OpticalFlowFaceAssociator(
        std::vector<Face>& faces,
        fc_v& prevCandidates,
        fc_v& nextCandidates,
        OpticalFlowManager& flowManager,
        const temp_idx_t prevFramePos,
        const temp_idx_t nextFramePos,
        double threshold):
    FaceAssociator(
            faces,
            prevCandidates,
            nextCandidates,
            threshold),
    flowManager(flowManager),
    prevFramePos(prevFramePos),
    nextFramePos(nextFramePos) {
    }

void OpticalFlowFaceAssociator::calculateProb() {
    typedef fc_v::size_type size_type;

    const size_type
        prevSize = prevCandidates.size(), nextSize = nextCandidates.size();

    // calculate probability
    for (size_type i = 0; i < prevSize; ++i) {
        FaceCandidate* prevC = prevCandidates[i];
        vector<int> pc(nextSize, 0);
        const cv::Point tl = prevC->rect.tl();
        const int rectWidth = prevC->rect.width;
        const int rectHeight = prevC->rect.height;
        const int rectArea = prevC->rect.area();

        for (int x = 0; x < rectWidth; ++x) {
            for (int y = 0; y < rectHeight; ++y) {
                const cv::Point p = tl + cv::Point(x, y);
                const cv::Vec2d v = flowManager.getFlowAt(
                        prevFramePos,
                        nextFramePos,
                        p.x,
                        p.y);
                const cv::Point2d pInDouble = p;
                const cv::Point2d dest = pInDouble + cv::Point2d(v);
                for (size_type j = 0; j < nextSize; ++j) {
                    if (nextCandidates[j]->rect.contains(dest)) {
                        ++pc[j];
                    }
                }
            }
        }

        for (size_type j = 0; j < nextSize; ++j) {
            prob[i][j] = (double)pc[j] / (double)rectArea;
        }
    }
}

SiftFaceAssociator::SiftFaceAssociator(std::vector<Face>& faces,
                                       fc_v& prevCandidates,
                                       fc_v& nextCandidates,
                                       const cv::Mat& prevFrame,
                                       const cv::Mat& nextFrame,
                                       double threshold):
    FaceAssociator(faces, prevCandidates, nextCandidates, threshold),
    prevFrame(prevFrame), nextFrame(nextFrame) {
    cv::Mat imgA, imgB;
    cv::SIFT sift = cv::SIFT();

    cv::cvtColor(this->prevFrame, imgA, CV_BGR2GRAY);
    cv::cvtColor(this->nextFrame, imgB, CV_BGR2GRAY);
    sift(imgA, cv::Mat(), this->keypointsA, this->descA);
    sift(imgB, cv::Mat(), this->keypointsB, this->descB);
}

void SiftFaceAssociator::calculateProb() {
    fc_v::size_type prevCddsSize, nextCddsSize;
    prevCddsSize = this->prevCandidates.size();
    nextCddsSize = this->nextCandidates.size();

    // set random seed for random-picking matches later
    srand(time(NULL));

    // find max rect from each prev candidates
    for (fc_v::size_type i = 0; i < prevCddsSize; ++i) {
        cv::Rect bestFitBox;
        this->computeBestFitBox(i, bestFitBox);
        this->bestFitBoxes.push_back(bestFitBox);

        for (fc_v::size_type j = 0; j < nextCddsSize; ++j) {
            const cv::Rect& afterCddBox = this->nextCandidates[j]->rect;
            cv::Rect intersection = bestFitBox & afterCddBox;
            const int intersectArea = intersection.area();
            this->prob[i][j] =
                (double) intersectArea /
                (double) (bestFitBox.area() +
                          afterCddBox.area() -
                          intersectArea);
        }
    }
}

void findSAB(Eigen::MatrixXd& matA, Eigen::VectorXd& matB, Eigen::VectorXd& matX)
{
    Eigen::MatrixXd matAT = matA.transpose();
    matX = (matAT * matA).inverse() * (matAT * matB);
}

void SiftFaceAssociator::computeBestFitBox(fc_v::size_type queryIdx,
                                           cv::Rect& bestFitBox) {
    const cv::Rect& queryBox = this->prevCandidates[queryIdx]->rect;
    cv::Ptr<cv::DescriptorMatcher> matcher =
        cv::DescriptorMatcher::create("BruteForce");
    vector<cv::DMatch> matches;
    cv::Mat matchMask;

    this->computeMatchMask(queryBox, matchMask);
    matcher->match(descA, descB, matches, matchMask);

    vector<cv::Rect> fitBoxes;
    for (int cnt_it = 0;
         cnt_it < UGPROJ_ASSOCIATOR_SIFT_TRIAL_COUNT;
         cnt_it++) {

        // random-pick two matches
        int idx_m1, idx_m2;
        idx_m1 = rand() % matches.size();
        idx_m2 = rand() % matches.size();

        cv::Rect fitBox;
        this->computeFitBox(matches[idx_m1], matches[idx_m2],
                            keypointsA, keypointsB,
                            queryBox, fitBox);
        fitBoxes.push_back(fitBox);
    }

    // find the best fit box
    double maxInlierRatio = -1.0f;
    vector<cv::Rect>::const_iterator it;

    for (it = fitBoxes.cbegin(); it != fitBoxes.cend(); ++it) {
        const cv::Rect fitBox = *it;

        // count the number of keypoints in fitBox
        int cnt_all = 0;
        vector<cv::KeyPoint>::const_iterator itKeypB;

        for (itKeypB = keypointsB.cbegin(); itKeypB != keypointsB.cend(); ++itKeypB) {
            const cv::KeyPoint& keypoint = *itKeypB;
            if (fitBox.contains(keypoint.pt)) {
                ++cnt_all;
            }
        }

        // count the number of keypoints matched with ones from queryBox
        int cnt_match = 0;
        vector<cv::DMatch>::const_iterator itMatches;

        for (itMatches = matches.cbegin();
             itMatches != matches.cend();
             ++itMatches) {
            const cv::DMatch& match = *itMatches;
            const cv::KeyPoint& keypoint = this->keypointsB[match.trainIdx];
            if (fitBox.contains(keypoint.pt)) {
                ++cnt_match;
            }
        }

        // compute inlier ratio and compare to maxInlierRatio
        double inlierRatio = (double) cnt_match / (double) cnt_all;
        if (inlierRatio > maxInlierRatio) {
            maxInlierRatio = inlierRatio;
            bestFitBox = fitBox;
        }
    }
}

void SiftFaceAssociator::computeMatchMask(const cv::Rect& beforeRect,
                                          cv::Mat& matchMask) {
    const vector<cv::KeyPoint>& keypointsA = this->keypointsA;
    const vector<cv::KeyPoint>& keypointsB = this->keypointsB;
    matchMask = cv::Mat::zeros(keypointsA.size(), keypointsB.size(), CV_8UC1);

    for (vector<cv::KeyPoint>::const_iterator it = keypointsA.cbegin();
         it != keypointsA.cend();
         ++it) {
        const cv::KeyPoint& kpA = *it;
        vector<cv::KeyPoint>::size_type i = it - keypointsA.cbegin();
        if (beforeRect.contains(kpA.pt)) {
            matchMask.row(i).setTo(1);
        }
    }
}

void SiftFaceAssociator::computeFitBox(
        const cv::DMatch& match1,
        const cv::DMatch& match2,
        const std::vector<cv::KeyPoint>& keypointsA,
        const std::vector<cv::KeyPoint>& keypointsB,
        const cv::Rect& beforeRect,
        cv::Rect& fitBox) const {
    // keypoint indices of random-picked matches
    // bp is for 'before (key)point', ap is for 'after (key)point'
    int idx_bp1, idx_bp2, idx_ap1, idx_ap2;
    idx_bp1 = match1.queryIdx;
    idx_bp2 = match2.queryIdx;
    idx_ap1 = match1.trainIdx;
    idx_ap2 = match2.trainIdx;

    int x1, y1, x2, y2;
    x1 = keypointsA[idx_bp1].pt.x;
    y1 = keypointsA[idx_bp1].pt.y;
    x2 = keypointsA[idx_bp2].pt.x;
    y2 = keypointsA[idx_bp2].pt.y;

    int sx1, sy1, sx2, sy2;
    sx1 = keypointsB[idx_ap1].pt.x;
    sy1 = keypointsB[idx_ap1].pt.y;
    sx2 = keypointsB[idx_ap2].pt.x;
    sy2 = keypointsB[idx_ap2].pt.y;

    // compute pseudo inverse
    Eigen::MatrixXd matA(4, 3);
    Eigen::VectorXd matB(4);
    Eigen::VectorXd matX;
    double s, a, b;
    matA << x1, 1, 0,
            x2, 1, 0,
            y1, 0, 1,
            y2, 0, 1;
    matB << sx1, sx2, sy1, sy2;
    findSAB(matA, matB, matX);
    s = matX[0];
    a = matX[1];
    b = matX[2];

    int bef_x1 = beforeRect.x;
    int bef_y1 = beforeRect.y;
    int bef_x2 = beforeRect.x + beforeRect.width - 1;
    int bef_y2 = beforeRect.y + beforeRect.height - 1;

    // when pseudo inverse
    int aft_x1 = (int)(s * (double)bef_x1 + a);
    int aft_y1 = (int)(s * (double)bef_y1 + b);
    int aft_x2 = (int)(s * (double)bef_x2 + a);
    int aft_y2 = (int)(s * (double)bef_y2 + b);

    // keep in boundary
    cv::Size frameSize = this->prevFrame.size();
    aft_x1 = std::min( std::max(0, aft_x1), frameSize.width - 1 );
    aft_y1 = std::min( std::max(0, aft_y1), frameSize.height - 1 );
    aft_x2 = std::min( std::max(0, aft_x2), frameSize.width - 1 );
    aft_y2 = std::min( std::max(0, aft_y2), frameSize.height - 1 );

    fitBox.x = aft_x1;
    fitBox.y = aft_y1;
    fitBox.width = aft_x2 - aft_x1;
    fitBox.height = aft_y2 - aft_y1;
}

static cv::Scalar colorPreset[] = {
    CV_RGB(0, 255, 0),
    CV_RGB(255, 0, 0),
    CV_RGB(0, 0, 255),
    CV_RGB(255, 255, 0),
    CV_RGB(255, 0, 255),
    CV_RGB(0, 255, 255)
};
// compute the number of preset colors
static const int nColorPreset = sizeof(colorPreset) / sizeof(cv::Scalar);

void SiftFaceAssociator::visualize(cv::Mat& img) {
    // clone prevFrame and nextFrame
    cv::Mat _prevFrame = this->prevFrame.clone();
    cv::Mat _nextFrame = this->nextFrame.clone();

    fc_v::size_type i;
    for (i = 0; i < this->prevCandidates.size(); ++i) {
        // set color
        cv::Scalar& color = colorPreset[i % nColorPreset];

        // draw candidate box on _prevFrame
        const cv::Rect& cddBox = this->prevCandidates[i]->rect;
        cv::rectangle(_prevFrame,
                      cddBox.tl(), cddBox.br(),
                      color);

        // draw best fit box on _nextFrame
        const cv::Rect& fitBox = this->bestFitBoxes[i];
        cv::rectangle(_nextFrame,
                      fitBox.tl(), fitBox.br(),
                      color);
    }

    // draw matches
    cv::drawMatches(_prevFrame,
                    this->keypointsA,
                    _nextFrame,
                    this->keypointsB,
                    this->matches,
                    img);
}
