#include "associator.hpp"
#include "../optflow/manager.hpp"

#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/Dense>
#include <Eigen/LU>

#include <boost/random.hpp>

#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <sstream>
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
        Fit bestFit;
        this->computeBestFitBox(i, &bestFit);
        this->bestFits.push_back(bestFit);

        const cv::Rect& bestFitBox = bestFit.box;
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

void SiftFaceAssociator::computeBestFitBox(fc_v::size_type queryIdx,
                                           Fit* bestFit) {
    const cv::Rect& queryBox = this->prevCandidates[queryIdx]->rect;
    cv::Ptr<cv::DescriptorMatcher> matcher =
        cv::DescriptorMatcher::create("BruteForce");
    vector<cv::DMatch> matches;
    cv::Mat matchMask;

    this->computeMatchMask(queryBox, matchMask);
    matcher->match(descA, descB, matches, matchMask);

    vector<cv::Rect> fitBoxes;
    int i = 0;
    while (i < UGPROJ_ASSOCIATOR_SIFT_TRIAL_COUNT) {
        // random-pick two matches
        int idx_m1, idx_m2;
        idx_m1 = rand() % matches.size();
        idx_m2 = rand() % matches.size();
        if (idx_m1 == idx_m2 && matches.size() > 1) {
            // same match picked: try again
            continue;
        }

        cv::Rect fitBox;
        this->computeFitBox(matches[idx_m1], matches[idx_m2],
                            keypointsA, keypointsB,
                            queryBox, fitBox);
        fitBoxes.push_back(fitBox);

        ++i;
    }

    // find the best fit box
    double maxInlierRatio = -1.0f;
    Fit best_fit;
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
            best_fit.box = fitBox;
            best_fit.matches = matches;
            best_fit.num_inlier = cnt_match;
        }
    }

    *bestFit = best_fit;
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
    // The top-left point of beforeRect is needed to set this as origin for
    // computation
    cv::Point origin = beforeRect.tl();

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

    // solve linear system
    Eigen::MatrixXd matA(4, 3);
    Eigen::Vector4d matB;
    Eigen::VectorXd matX;
    double s, a, b;
    matA << x1 - origin.x, 1, 0,
            x2 - origin.x, 1, 0,
            y1 - origin.y, 0, 1,
            y2 - origin.y, 0, 1;
    matB << sx1 - origin.x,
            sx2 - origin.x,
            sy1 - origin.y,
            sy2 - origin.y;
    matX = matA.colPivHouseholderQr().solve(matB);
    s = matX[0];
    a = matX[1];
    b = matX[2];

    int fitbox_l = beforeRect.x + a;
    int fitbox_t = beforeRect.y + b;
    int fitbox_r = (int)(fitbox_l + s * beforeRect.width);
    int fitbox_b = (int)(fitbox_t + s * beforeRect.height);

    // keep in boundary
    cv::Size frameSize = this->prevFrame.size();
    fitbox_l = std::min( std::max(0, fitbox_l), frameSize.width);
    fitbox_t = std::min( std::max(0, fitbox_t), frameSize.height);
    fitbox_r = std::min( std::max(0, fitbox_r), frameSize.width);
    fitbox_b = std::min( std::max(0, fitbox_b), frameSize.height);

    fitBox.x = fitbox_l;
    fitBox.y = fitbox_t;
    fitBox.width = fitbox_r - fitbox_l;
    fitBox.height = fitbox_b - fitbox_t;
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
        const cv::Scalar color = this->color_for(i);

        // draw candidate box on _prevFrame
        const cv::Rect& cddBox = this->prevCandidates[i]->rect;
        cv::rectangle(_prevFrame,
                      cddBox.tl(), cddBox.br(),
                      color);
    }
    // write next candidates
    for (i = 0; i < this->nextCandidates.size(); ++i) {
        this->draw_next_candidates(i, &_nextFrame);
    }

    // consolidate matches
    vector<cv::DMatch> matches;
    vector<Fit>::const_iterator it;
    for (it = this->bestFits.cbegin(); it != this->bestFits.cend(); ++it) {
        const Fit& f = *it;
        matches.insert(matches.end(), f.matches.begin(), f.matches.end());
    }

    // draw matches
    cv::drawMatches(_prevFrame,
                    this->keypointsA,
                    _nextFrame,
                    this->keypointsB,
                    matches,
                    img,
                    cv::Scalar::all(-1),    // random colors for matchColor
                    cv::Scalar::all(-1),    // random colors for singlePointColor
                    std::vector<char>(),    // empty matchMask
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    for (i = 0; i < this->prevCandidates.size(); ++i) {
        this->draw_best_fit(i, &img);
    }
}

inline cv::Scalar SiftFaceAssociator::color_for(const fc_v::size_type cdd_index) {
    return colorPreset[cdd_index % nColorPreset];
}

void SiftFaceAssociator::draw_best_fit(const fc_v::size_type cdd_index,
                                       cv::Mat* match_img) {
    // offset to draw best fit on latter frame
    cv::Point offset_x = cv::Point(this->nextFrame.cols, 0);

    const Fit& bestFit = this->bestFits[cdd_index];

    // draw best fit box
    const cv::Rect& fitBox = bestFit.box;
    const cv::Scalar color = this->color_for(cdd_index);
    cv::rectangle(*match_img,
                  fitBox.tl() + offset_x, fitBox.br() + offset_x,
                  color);

    // compute the scale and draw this and inlier information
    // below the best fit box
    const cv::Rect& cddBox = this->prevCandidates[cdd_index]->rect;
    const double scale = (double) fitBox.width / (double) cddBox.width;

    // generate text to draw
    stringstream ss;
    string text_1, text_2;
    ss << "s: " << scale << " (1/" << (1 / scale) << ")";
    text_1 = ss.str();
    ss.str("");
    ss << "# of inliers: " << bestFit.num_inlier << "("
       << bestFit.inlier_ratio() * 100 << "%)";
    text_2 = ss.str();

    // compute text offset from box
    cv::Point offset_text;
    int baseline;
    const cv::Size text_1_size =
            cv::getTextSize(text_1, CV_FONT_HERSHEY_PLAIN, 1.0, 1, &baseline);
    const cv::Size text_2_size =
            cv::getTextSize(text_2, CV_FONT_HERSHEY_PLAIN, 1.0, 1, &baseline);
    const int text_width = std::max(text_1_size.width, text_2_size.width);
    const int text_height = text_1_size.height + 4 + text_2_size.height + 4;
    const cv::Point tl_box = fitBox.tl() + offset_x;
    if (tl_box.y - text_height < 0) {
        // Text will overflow if it is placed above the box.
        // Hence, place it over the box.
        offset_text = cv::Point(0, fitBox.height + text_height);
    } else {
        offset_text = cv::Point(0, -4);
    }
    if (tl_box.x + text_width > 2 * this->prevFrame.cols) {
        // Text will overflow if it is left-aligned to the box.
        // Hence, align to right.
        offset_text.x -= text_width - fitBox.width;
    }

    // draw text
    cv::putText(*match_img,
                text_2,
                tl_box + offset_text,
                CV_FONT_HERSHEY_PLAIN,
                1.0,
                color);
    cv::putText(*match_img,
                text_1,
                tl_box + offset_text
                - cv::Point(0, 4 + text_2_size.height),
                CV_FONT_HERSHEY_PLAIN,
                1.0,
                color);
}

void SiftFaceAssociator::draw_next_candidates(const fc_v::size_type cdd_index, cv::Mat* next_frame){
    // set color
    const cv::Scalar color = this->color_for(cdd_index);

    // draw candidate box on _prevFrame
    const cv::Rect& cddBox = this->nextCandidates[cdd_index]->rect;
    cv::rectangle(*next_frame,
        cddBox.tl(), cddBox.br(),
        color);

    // generate text to draw
    stringstream ss;
    string text;
    ss << "next " << cdd_index;
    text = ss.str();

    // compute text offset from box
    cv::Point offset_text;
    int baseline;
    const cv::Size text_size =
        cv::getTextSize(text, CV_FONT_HERSHEY_PLAIN, 1.0, 1, &baseline);
    const int text_width = text_size.width;
    const int text_height = text_size.height + 4;
    const cv::Point tl_box = cddBox.tl();
    if (tl_box.y - text_height < 0) {
        // Text will overflow if it is placed above the box.
        // Hence, place it over the box.
        offset_text = cv::Point(0, cddBox.height + text_height);
    }
    else {
        offset_text = cv::Point(0, 12 + cddBox.height);
    }
    if (tl_box.x + text_width > 2 * this->prevFrame.cols) {
        // Text will overflow if it is left-aligned to the box.
        // Hence, align to right.
        offset_text.x -= text_width - cddBox.width;
    }

    // draw text
    cv::putText(*next_frame,
        text,
        tl_box + offset_text,
        CV_FONT_HERSHEY_PLAIN,
        1.0,
        color);
}
