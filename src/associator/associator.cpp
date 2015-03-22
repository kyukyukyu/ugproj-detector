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

void SiftFaceAssociator::calculateProb() {

	calculateNextRect();

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
	   for (int i = 0; i < prevCddsSize; ++i) {
            if (prevCandidates[i]->rect.contains(kpA.pt)) {
				selectedIdxsA.push_back(i);
            }
        }

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

    int nRows = pc.rows();
    for (int rowIndex = 0; rowIndex < nRows; ++rowIndex) {
        const Eigen::RowVectorXi& _row = pc.row(rowIndex);
        int nDest = _row.sum();

        Eigen::RowVectorXd row = _row.cast<double>();
        row /= nDest;

        memcpy(this->prob[rowIndex], row.data(), sizeof(double) * row.cols());
    }
}



void findSAB(Eigen::MatrixXd& matA, Eigen::VectorXd& matB, Eigen::VectorXd& matX)
{
    Eigen::MatrixXd matAT = matA.transpose();
    matX = (matAT * matA).inverse() * (matAT * matB);
}

void findSxSyAB(int N, int M, vector <double> &A, vector <double> &B, vector <double> &X)
{
	// find Sx, Sy, A, B by inverse
	Eigen::MatrixXd _A(N, M);
	Eigen::MatrixXd _B(N, 1);
	Eigen::MatrixXd _X(M, 1);

	for (int i = 0; i < N; i++) {
		_B(i, 0) = B[i];
		for (int j = 0; j < M; j++) {
			_A(i, j) = A[i*M + j];
		}
	}
	_X = _A.inverse()*_B;

	for (int i = 0; i < M; i++)
		X[i] = _X(i, 0);
}

void SiftFaceAssociator::calculateNextRect() {

	int prevCddsSize, nextCddsSize;
	prevCddsSize = this->prevCandidates.size();
	nextCddsSize = this->nextCandidates.size();

	cv::Scalar colorPreset[] = {
		CV_RGB(0, 255, 0),
		CV_RGB(255, 0, 0),
		CV_RGB(0, 0, 255),
		CV_RGB(255, 255, 0),
		CV_RGB(255, 0, 255),
		CV_RGB(0, 255, 255)
	};

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

	vector<cv::Mat> matchMasks; // match masks for each prevCandidates
	vector<int> cnt;
	cv::Mat matchMask;

	printf("prevCddsSize is %d", prevCddsSize);

	for (int i = 0; i < prevCddsSize; ++i) {
		// matchMask initialize
		matchMask = cv::Mat::zeros(keypointsA.size(), keypointsB.size(), CV_8UC1);
		matchMasks.push_back(matchMask.clone());
		cnt.push_back(0);
	}

	int cnt_k = 0; // cnt for keypoints

	// classify each rect's feature with making match mask
	for (vector<cv::KeyPoint>::const_iterator it = keypointsA.cbegin();
		it != keypointsA.cend();
		++it, ++cnt_k){
		const cv::KeyPoint& kpA = keypointsA[cnt_k];
		int j;
		for (int i = 0; i < prevCddsSize; ++i) {
			if (prevCandidates[i]->rect.contains(kpA.pt)) {
				for (j = 0; j < keypointsB.size(); j++){
					matchMasks[i].at<uchar>(cnt_k, j) = 1;
					cnt[i]++;
				}
			}
			else{
				for (j = 0; j < keypointsB.size(); j++){
					matchMasks[i].at<uchar>(cnt_k, j) = 0;
				}
			}
		}
	}
	
	// drawing the results
	cv::Mat img_matches;
	
	vector< vector<cv::DMatch> > matches_list; // vector for all matches from each prev candidates

	cv::Mat next = nextFrame.clone();

	// find max rect from each prev candidates
	for (int i = 0; i < prevCddsSize; ++i) {
		vector<cv::DMatch> matches;
		matches_list.push_back(matches);

		// split each prevCandidates' matches with match mask
		matcher->match(descA, descB, matches, matchMasks[i]);
	
		vector< vector<cv::Rect> > eachRectCandidates; // vector for all rect candidates
		vector<cv::Rect> rectCandidates; // vector for 10 rect candidates from each prev Candidates

		srand(time(NULL));
		for (int cnt_it = 0;
             cnt_it < UGPROJ_ASSOCIATOR_SIFT_TRIAL_COUNT;
             cnt_it++) {

			// pick random match pointer
			int idx_m1, idx_m2, idx_bp1, idx_bp2, idx_ap1, idx_ap2;

			idx_m1 = rand() % matches.size();
			idx_m2 = rand() % matches.size();

			idx_bp1 = matches[idx_m1].queryIdx;
			idx_bp2 = matches[idx_m2].queryIdx;
			idx_ap1 = matches[idx_m1].trainIdx;
			idx_ap2 = matches[idx_m2].trainIdx;

			int x1, y1, x2, y2, ax1, ay1, ax2, ay2;

			x1 = keypointsA[idx_bp1].pt.x;
			y1 = keypointsA[idx_bp1].pt.y;
			x2 = keypointsA[idx_bp2].pt.x;
			y2 = keypointsA[idx_bp2].pt.y;

			ax1 = keypointsB[idx_ap1].pt.x;
			ay1 = keypointsB[idx_ap1].pt.y;
			ax2 = keypointsB[idx_ap2].pt.x;
			ay2 = keypointsB[idx_ap2].pt.y;

			// calculate pseudo inverse or inverse
            Eigen::MatrixXd matA(4, 3);
            Eigen::VectorXd matB(4);
            Eigen::VectorXd matX;

			matA << x1, 1, 0,
				    x2, 1, 0,
				    y1, 0, 1,
				    y2, 0, 1;
			matB << ax1, ax2, ay1, ay2;

			findSAB(matA, matB, matX);

			cv::Rect calculatedRect;
			calculatedRect = prevCandidates[i]->rect;
			int bef_x1 = calculatedRect.x;
			int bef_y1 = calculatedRect.y;
			int bef_x2 = calculatedRect.x + calculatedRect.width - 1;
			int bef_y2 = calculatedRect.y + calculatedRect.height - 1;

			// when pseudo inverse
			int aft_x1 = (int)(matX[0] * (double)bef_x1 + matX[1]);
			int aft_y1 = (int)(matX[0] * (double)bef_y1 + matX[2]);
			int aft_x2 = (int)(matX[0] * (double)bef_x2 + matX[1]);
			int aft_y2 = (int)(matX[0] * (double)bef_y2 + matX[2]);

			// cut boundry
            aft_x1 = std::min( std::max(0, aft_x1), imgA.size().width - 1 );
            aft_y1 = std::min( std::max(0, aft_y1), imgA.size().height - 1 );
            aft_x2 = std::min( std::max(0, aft_x2), imgA.size().width - 1 );
            aft_y2 = std::min( std::max(0, aft_y2), imgA.size().height - 1 );

			calculatedRect.x = aft_x1;
			calculatedRect.y = aft_y1;
			calculatedRect.width = aft_x2 - aft_x1;
			calculatedRect.height = aft_y2 - aft_y1;

			rectCandidates.push_back(calculatedRect);

		}
		eachRectCandidates.push_back(rectCandidates);

		// count which is included to each rect Candidates among current prevCandidate's match keypointsB
		printf("\ncalculate inlier ratio\n");
		vector<int> cnt_match(UGPROJ_ASSOCIATOR_SIFT_TRIAL_COUNT, 0); // matched feature with each prev Candidate
		vector<int> cnt_all(UGPROJ_ASSOCIATOR_SIFT_TRIAL_COUNT, 0); // all matched feature

		// calculate cnt_all
        vector<cv::KeyPoint>::const_iterator itKeypB;
        for (itKeypB = keypointsB.cbegin(); itKeypB != keypointsB.cend(); ++itKeypB) {
			const cv::KeyPoint& target = *itKeypB;
			for (int cnt_rc = 0;
                 cnt_rc < UGPROJ_ASSOCIATOR_SIFT_TRIAL_COUNT;
                 ++cnt_rc) {
				if (rectCandidates[cnt_rc].contains(target.pt)) {
					cnt_all[cnt_rc]++;
				}
			}
		}

		// calculate cnt_match
		for (vector<cv::DMatch>::const_iterator it = matches.cbegin();
			it != matches.cend();
			++it){
            const cv::DMatch& match = *it;
			const cv::KeyPoint& target = keypointsB[match.trainIdx];
			for (int cnt_rc = 0;
                 cnt_rc < UGPROJ_ASSOCIATOR_SIFT_TRIAL_COUNT;
                 ++cnt_rc) {
				if (rectCandidates[cnt_rc].contains(target.pt)) {
					cnt_match[cnt_rc]++;
				}
			}
		}

		// find max ratio box
		cv::Rect maxRect;
		double max = 0;
		printf("find Max Ratio Box\n");
		for (int cnt_rc = 0;
             cnt_rc < UGPROJ_ASSOCIATOR_SIFT_TRIAL_COUNT;
             ++cnt_rc) {
            double inlierRatio = (double) cnt_match[cnt_rc] /
                                 (double) cnt_all[cnt_rc];
			if (inlierRatio > max) {
				max = inlierRatio;
				maxRect = rectCandidates[cnt_rc];
			}
		}

		printf("Max Rect (%d,%d) - (%d,%d)\n", maxRect.x, maxRect.y, maxRect.x + maxRect.width - 1, maxRect.y + maxRect.height);
		rectangle(next,
			cvPoint(maxRect.x, maxRect.y),
			cvPoint(maxRect.x + maxRect.width - 1, maxRect.y + maxRect.height),
			colorPreset[i]);

		// draw matches
		drawMatches(imgA, keypointsA, next, keypointsB, matches, img_matches, colorPreset[i], cv::Scalar::all(-1), vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	}

	char filename[100];
	sprintf(filename, "output/result%d.jpg", result_cnt++);
	imwrite(filename, img_matches);

}

