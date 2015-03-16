#include "associator.hpp"
#include "../optflow/manager.hpp"

#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/Dense>
#include <Eigen/LU>

#include <cstring>
#include <cstdlib>
#include <time.h>

using namespace ugproj;
using namespace std;
using Eigen::MatrixXd;
using namespace Eigen;

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


/*
void pinv(MatrixType& pinvmat){
	eigen_assert(m_isInitialized && "SVD is not initialized.");
	double  pinvtoler = 1.e-6; // choose your tolerance wisely!
	SingularValuesType singularValues_inv = m_singularValues;
	for (long i = 0; i<m_workMatrix.cols(); ++i) {
		if (m_singularValues(i) > pinvtoler)
			singularValues_inv(i) = 1.0 / m_singularValues(i);
		else singularValues_inv(i) = 0;
	}
	pinvmat = (m_matrixV*singularValues_inv.asDiagonal()*m_matrixU.transpose());
}
*/

void findSAB(int N, int M, vector <double> &A, vector <double> &B, vector <double> &X)
{
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

	vector<cv::Mat> matchMasks;
	vector<int> cnt;

	cv::Mat matchMask;

	printf("prevCddsSize is %d", prevCddsSize);

	for (int i = 0; i < prevCddsSize; ++i) {
		matchMask = cv::Mat::zeros(keypointsA.size(), keypointsB.size(), CV_8UC1);
		matchMasks.push_back(matchMask.clone());
		cnt.push_back(0);
	}

	int cnt_k=0;

	// classify each rect's feature with making match mask
	for (vector<cv::KeyPoint>::const_iterator it = keypointsA.cbegin();
		it != keypointsA.cend();
		++it,++cnt_k){
		const cv::KeyPoint& kpA = keypointsA[cnt_k];
		int j;
		for (int i = 0; i < prevCddsSize; ++i) {
			if (prevCandidates[i]->rect.contains(kpA.pt)) {
				for ( j = 0; j < keypointsB.size(); j++){
					matchMasks[i].at<uchar>(cnt_k, j) = 1;
					cnt[i]++;
				}
			}
			else{
				for ( j = 0; j < keypointsB.size(); j++){
					matchMasks[i].at<uchar>(cnt_k, j) = 0;
				}
			}
		}
	}
	cv::Scalar colorPreset[] = {
		CV_RGB(0, 255, 0),
		CV_RGB(255, 0, 0),
		CV_RGB(0, 0, 255),
		CV_RGB(255, 255, 0),
		CV_RGB(255, 0, 255),
		CV_RGB(0, 255, 255)
	};

	// drawing the results
	//cv::namedWindow("matches", 1);
	cv::Mat img_matches;
	vector<cv::DMatch> merged_matches;

	vector<vector<cv::DMatch>> matches_list;
	cv::Mat next = nextFrame.clone();

	// split each rect's matches with match mask
	for (int i = 0; i < prevCddsSize; ++i) {
		vector < cv::DMatch > matches_temp;
		matches_list.push_back(matches_temp);

		matcher->match(descA, descB, matches_list[i], matchMasks[i]);
		printf("\nprevCdd %d's match size is %d", i, matches_list[i].size());

		
		
		// pick random match pointer
		int idx_m1, idx_m2, idx_bp1, idx_bp2, idx_ap1, idx_ap2;
		srand(time(NULL));
		idx_m1 = rand() % matches_list[i].size();
		idx_m2 = rand() % matches_list[i].size();

		idx_bp1 = matches_list[i][idx_m1].queryIdx;
		idx_bp2 = matches_list[i][idx_m2].queryIdx;
		idx_ap1 = matches_list[i][idx_m1].trainIdx;
		idx_ap2 = matches_list[i][idx_m2].trainIdx;

		int x1, y1, x2, y2, ax1, ay1, ax2, ay2;
		x1 = keypointsA[idx_bp1].pt.x;
		y1 = keypointsA[idx_bp1].pt.y;
		x2 = keypointsA[idx_bp2].pt.x;
		y2 = keypointsA[idx_bp2].pt.y;

		ax1 = keypointsA[idx_ap1].pt.x;
		ay1 = keypointsA[idx_ap1].pt.y;
		ax2 = keypointsA[idx_ap2].pt.x;
		ay2 = keypointsA[idx_ap2].pt.y;


		printf("\n1st random pointer is %d (%d, %d) -> (%d, %d)", idx_bp1, x1, y1, ax1, ay1);
		printf("\n2nd random pointer is %d (%d, %d) -> (%d, %d)", idx_bp2, x2, y2, ax2, ay2);

		// print random match pointer
		string temp = to_string(i) + "_bp1";
		cv::putText(imgA,
			temp,
			cvPoint(x1, y1),
			cv::FONT_HERSHEY_PLAIN,
			1.0,
			colorPreset[i]);
		temp = to_string(i) + "_bp2";
		cv::putText(imgA,
			temp,
			cvPoint(x2, y2),
			cv::FONT_HERSHEY_PLAIN,
			1.0,
			colorPreset[i]);
		temp = to_string(i) + "_ap1";
		cv::putText(next,
			temp,
			cvPoint(ax1, ay1),
			cv::FONT_HERSHEY_PLAIN,
			1.0,
			colorPreset[i]);
		temp = to_string(i) + "_ap2";
		cv::putText(next,
			temp,
			cvPoint(ax2, ay2),
			cv::FONT_HERSHEY_PLAIN,
			1.0,
			colorPreset[i]);

		// calculate Rect
		vector <double> A(16);
		vector <double> B(4);
		vector <double> X(4);

		/*
		A =	{	0, 0, 1, 0, 
			1,0, 1, 0,
			0,1, 0, 1,
			0, 0, 0, 1 };
		B = { 1,3,3,1 };
		*/
		
		A = { (double)x1, 0, 1, 0,
			(double)x2, 0, 1, 0,
			0, (double)y1, 0, 1,
			0, (double)y2, 0, 1 };
		B = { (double)ax1, (double)ax2, (double)ay1, (double)ay2 };
		
		findSAB(4, 4, A, B, X);

		printf("\nSx, Sy, a, b = (%lf,%lf,%lf,%lf)\n", X[0], X[1], X[2], X[3]);

		// draw Rect in imgB

		//  drawRect(frame, candidate->faceId, candidate->rect);
		// (Mat& frame, Face::id_type id, const Rect& facePosition)
		cv::Rect calculatedRect;
		calculatedRect = prevCandidates[i]->rect;
		int bef_x1 = calculatedRect.x;
		int bef_y1 = calculatedRect.y;
		int bef_x2 = calculatedRect.x + calculatedRect.width - 1;
		int bef_y2 = calculatedRect.y + calculatedRect.height - 1;

		printf("\nimg size is (%d,%d)\n", imgA.size().width, imgA.size().height);

		printf("\ncalculatedRect before (%d,%d) - (%d,%d)\n", bef_x1,bef_y1, bef_x2, bef_y2);

		int aft_x1 = (int)(X[0] * (double)bef_x1 + X[2]);
		int aft_y1 = (int)(X[1] * (double)bef_y1 + X[3]);
		int aft_x2 = (int)(X[0] * (double)bef_x2 + X[2]);
		int aft_y2 = (int)(X[1] * (double)bef_y2 + X[3]);

		if (aft_x1 >= imgA.size().width)	aft_x1 = imgA.size().width-1;
		else if (aft_x1 < 0) aft_x1 = 0;
		if (aft_y1 >= imgA.size().height)aft_y1 = imgA.size().height-1;
		else if (aft_y1 < 0) aft_y1 = 0;
		if (aft_x2 >= imgA.size().width)	aft_x2 = imgA.size().width-1;
		else if (aft_x2 < 0) aft_x2 = 0;
		if (aft_y2 >= imgA.size().width)	aft_y2 = imgA.size().height-1;
		else if (aft_y2 < 0) aft_y2 = 0;

		printf("\ncalculatedRect (%d,%d) - (%d,%d)\n", aft_x1,aft_y1,aft_x2,aft_y2);

		
		rectangle(next,
			cvPoint(aft_x1, aft_y1),
			cvPoint(aft_x2,	aft_y2),
			colorPreset[i]);

		// draw matches
		drawMatches(imgA, keypointsA, next, keypointsB, matches_list[i], img_matches, colorPreset[i], cv::Scalar::all(-1), vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		
		imshow("match", img_matches);
		cv::waitKey(500);

	}

	// pick 2 points in random and calculate S, a, b
	vector<vector<cv::Rect>> CandidateRects;
	 
	for (int i = 0; i < prevCddsSize; ++i) {
	

	}

	char filename[100];
	sprintf(filename, "output/result%d.jpg", result_cnt++);
	imwrite(filename, img_matches);

}

