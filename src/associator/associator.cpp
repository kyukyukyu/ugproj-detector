#include "associator.hpp"
#include "../optflow/manager.hpp"

#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/Dense>

#include <cstring>

using namespace ugproj;
using namespace std;

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


void SiftFaceAssociator::calculateProb() {
    int prevCddsSize, nextCddsSize;
    prevCddsSize = this->prevCandidates.size();
    nextCddsSize = this->nextCandidates.size();

    cv::SIFT sift = cv::SIFT();
    vector<cv::KeyPoint> keypointsA, keypointsB;
    cv::Mat descA, descB;
    cv::Mat imgA, imgB;

    cv::cvtColor(this->prevFrame, imgA, CV_BGR2GRAY);
    cv::cvtColor(this->nextFrame, imgB, CV_BGR2GRAY);

    sift(imgA, cv::Mat(), keypointsA, descA);
    sift(imgB, cv::Mat(), keypointsB, descB);

    cv::Ptr<cv::DescriptorMatcher> matcher =
        cv::DescriptorMatcher::create("BruteForce");
    vector<cv::DMatch> matches;

    matcher->match(descA, descB, matches);


    typedef Eigen::Matrix<int,
                          Eigen::Dynamic,
                          Eigen::Dynamic,
                          Eigen::RowMajor> PcMatrix;
    PcMatrix pc = PcMatrix::Zero(prevCddsSize, nextCddsSize);

	vector<char> matchesMask(matches.size(), 0);

    for (vector<cv::DMatch>::const_iterator it = matches.cbegin();
         it != matches.cend();
         ++it) {
        const cv::DMatch& match = *it;
        const int indexA = match.queryIdx;
        const int indexB = match.trainIdx;
        const cv::KeyPoint& kpA = keypointsA[indexA];
        const cv::KeyPoint& kpB = keypointsB[indexB];
        vector<int> selectedIdxsA = vector<int>();
        vector<int> selectedIdxsB = vector<int>();
		int flag = 0;
        for (int i = 0; i < prevCddsSize; ++i) {
            if (prevCandidates[i]->rect.contains(kpA.pt)) {
				flag = 1;
				matchesMask[indexA] = 1;
				selectedIdxsA.push_back(i);
            }
        }
		if (flag == 0)
			matchesMask[indexA] = 0;

        for (int j = 0; j < nextCddsSize; ++j) {
            if (nextCandidates[j]->rect.contains(kpB.pt)) {
                selectedIdxsB.push_back(j);
            }
        }

        for (vector<int>::const_iterator itI = selectedIdxsA.cbegin();
             itI != selectedIdxsA.cend();
             ++itI) {
            for (vector<int>::const_iterator itJ = selectedIdxsB.cbegin();
                 itJ != selectedIdxsB.cend();
                 ++itJ) {
                const int i = *itI;
                const int j = *itJ;
                pc(i, j) += 1;
            }
        }
    }

	// drawing the results
	cv::namedWindow("matches", 1);
	cv::Mat img_matches;
	drawMatches(imgA, keypointsA, imgB, keypointsB, matches, img_matches, CV_RGB(0, 255, 0), CV_RGB(0, 0, 255), matchesMask);
	imshow("matches", img_matches);
	cv::waitKey(1000);

    int nRows = pc.rows();
    for (int rowIndex = 0; rowIndex < nRows; ++rowIndex) {
        const Eigen::RowVectorXi& _row = pc.row(rowIndex);
        int nDest = _row.sum();

        Eigen::RowVectorXd row = _row.cast<double>();
        row /= nDest;

        memcpy(this->prob[rowIndex], row.data(), sizeof(double) * row.cols());
    }
}
