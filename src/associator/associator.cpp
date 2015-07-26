#include "associator.hpp"

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
#include <random>
#include <sstream>
#include <utility>

using namespace ugproj;
using namespace std;
using Eigen::MatrixXd;
using namespace Eigen;
using namespace boost;

void FaceAssociator::matchCandidates() {
  typedef fc_v::size_type size_type;

  const size_type
    prevSize = prevCandidates.size(), nextSize = nextCandidates.size();

  for (size_type j = 0; j < nextSize; ++j) {
    double max = -1;
    size_type maxRow;

    for (size_type i = 0; i < prevSize; ++i) {
      printf("prob[%d][%d] = %f\n",i,j,prob[i][j]);
      if (prob[i][j] > max && prob[i][j] > threshold) {
        max = prob[i][j];
        maxRow = i;
      }
    }

    vector<Face>::size_type faceId;
    if (max > 0) {
      faceId = prevCandidates[maxRow].faceId;
    } else {
      std::printf("new face\n");
      faceId = faces.size() + 1;
      faces.push_back(Face(faceId));
    }
    nextCandidates[j].faceId = faceId;
    faces[faceId - 1].addCandidate(nextCandidates[j]);
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
    const cv::Rect& rectI = prevCandidates[i].rect;
    for (size_type j = 0; j < nextSize; ++j) {
      const cv::Rect& rectJ = nextCandidates[j].rect;
      cv::Rect intersect = rectI & rectJ;
      int intersectArea = intersect.area();
      int unionArea =
        rectI.area() + rectJ.area() - intersectArea;
      prob[i][j] = (double)intersectArea / unionArea;
    }
  }
}

#ifndef UGPROJ_SUPPRESS_CELIU
OpticalFlowFaceAssociator::OpticalFlowFaceAssociator(
    std::vector<Face>& faces,
    const fc_v& prevCandidates,
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
    const FaceCandidate& prevC = prevCandidates[i];
    vector<int> pc(nextSize, 0);
    const cv::Point tl = prevC.rect.tl();
    const int rectWidth = prevC.rect.width;
    const int rectHeight = prevC.rect.height;
    const int rectArea = prevC.rect.area();

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
          if (nextCandidates[j].rect.contains(dest)) {
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
#endif

SiftFaceAssociator::SiftFaceAssociator(std::vector<Face>& faces,
                     const fc_v& prevCandidates,
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
      const cv::Rect& afterCddBox = this->nextCandidates[j].rect;
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
  const cv::Rect& queryBox = this->prevCandidates[queryIdx].rect;
  cv::Ptr<cv::DescriptorMatcher> matcher =
    cv::DescriptorMatcher::create("BruteForce");
  vector<cv::DMatch> matches;
  cv::Mat matchMask;

  this->computeMatchMask(queryBox, matchMask);
  matcher->match(descA, descB, matches, matchMask);

  vector<cv::Rect> fitBoxes;
  this->list_fit_boxes(matches, queryBox, &fitBoxes);

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

void SiftFaceAssociator::list_fit_boxes(const vector<cv::DMatch>& matches,
                    const cv::Rect& query_box,
                    vector<cv::Rect>* fit_boxes) {
  const vector<cv::DMatch>::size_type num_matches = matches.size();
  if (num_matches <= 2) {
    const cv::DMatch& match1 = matches[0];
    const cv::DMatch& match2 =
        (num_matches == 1) ? matches[0] : matches[1];
    cv::Rect fit_box;
    this->computeFitBox(match1, match2,
              this->keypointsA, this->keypointsB,
              query_box, fit_box);
    fit_boxes->push_back(fit_box);
    return;
  }

  int i = 0;
  while (i < UGPROJ_ASSOCIATOR_SIFT_TRIAL_COUNT) {
    // random-pick two matches
    int idx_m1, idx_m2;
    idx_m1 = rand() % num_matches;
    idx_m2 = rand() % num_matches;
    if (idx_m1 == idx_m2) {
      // same match picked: try again
      continue;
    }

    cv::Rect fit_box;
    bool is_valid_fitting = this->computeFitBox(
        matches[idx_m1], matches[idx_m2],
        keypointsA, keypointsB,
        query_box, fit_box);
    if (!is_valid_fitting) {
      continue;
    }

    fit_boxes->push_back(fit_box);
    ++i;
  }
}

bool SiftFaceAssociator::computeFitBox(
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

  if (s >= UGPROJ_ASSOCIATOR_SIFT_SCALE_THRESHOLD ||
    1/s >= UGPROJ_ASSOCIATOR_SIFT_SCALE_THRESHOLD) {
    return false;
  }

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

  return true;
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
    const cv::Rect& cddBox = this->prevCandidates[i].rect;
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
          cv::Scalar::all(-1),  // random colors for matchColor
          cv::Scalar::all(-1),  // random colors for singlePointColor
          std::vector<char>(),  // empty matchMask
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
  const cv::Rect& cddBox = this->prevCandidates[cdd_index].rect;
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
  const cv::Rect& cddBox = this->nextCandidates[cdd_index].rect;
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

KltFaceAssociator::KltFaceAssociator(
    std::vector<Face>& faces,
    const FaceCandidateList& prev_candidates,
    FaceCandidateList& next_candidates,
    const temp_idx_t next_index,
    const cv::Mat& next_frame,
    const std::vector<SparseOptflow>& optflows,
    double threshold) :
  FaceAssociator(faces, prev_candidates, next_candidates, threshold),
  next_index_(next_index),
  next_frame_(next_frame),
  frame_size_(next_frame.size()),
  optflows_(optflows) {}

void KltFaceAssociator::associate() {
  FaceAssociator::associate();
  // Add best fits of non-labeled face candidates in previous frame to
  // nextCandidates.
  const auto prev_size = this->prevCandidates.size();
  const auto next_size = this->nextCandidates.size();
  std::printf("in associate, prev size %d next size %d\n",prev_size,next_size);
  for (FaceCandidateList::size_type i = 0; i < prev_size; ++i) {
    unsigned int n_overlapped = 0;
    for (FaceCandidateList::size_type j = 0; j < next_size; ++j) {
      if (this->prob[i][j] > threshold) {
        ++n_overlapped;
      }
    }
    std::printf("prev size %d's overlapped %d\n",i,n_overlapped);
    if (n_overlapped == 0) {
      Fit best_fit = this->best_fits_[i];
      std::printf("prev %d's inlier num is %d\n",i,best_fit.num_inliers);
      if (!best_fit.valid()) {
        continue;
      }
      const cv::Rect& face_rect = best_fit.box;
      std::printf("new fit (%d,%d/%d,%d)\n",face_rect.x,face_rect.y,face_rect.width,face_rect.height); 
      const cv::Mat face_img(this->next_frame_, face_rect);
      const face_id_t face_id = this->prevCandidates[i].faceId;
      FaceCandidate restored(this->next_index_, face_rect, face_img);
      restored.faceId = face_id;
      restored.fitted = 1;
      printf("fitted is %d\n",restored.fitted);
      this->nextCandidates.push_back(restored);
      this->faces[face_id - 1].addCandidate(restored);
    }
  }
}

void KltFaceAssociator::calculateProb() {
  // Set random seed for random-picking matches later.
  srand(time(NULL));

  // Best fits will be saved into `this->best_fits_`.
  this->compute_best_fits();

  // Compute correspondence for each pair of face candidates in two frames.
  FaceCandidateList::size_type i, j, n_prev_cdds, n_next_cdds;
  n_prev_cdds = this->prevCandidates.size();
  n_next_cdds = this->nextCandidates.size();
  for (i = 0; i < n_prev_cdds; ++i) {
    Fit& best_fit = this->best_fits_[i];
    if (!best_fit.valid()) {
      // No best fit available. Set all probability to zero.
      std::printf("%d's prev candidate's best fit is null!\n",i);
      std::fill(prob[i], prob[i] + n_next_cdds, 0.0);
      continue;
    }

    const cv::Rect& fit_box = best_fit.box;
    std::printf("%d's prev candidate's best fit (%d,%d/%d,%d)\n",i,fit_box.x,fit_box.y,fit_box.width,fit_box.height); 
    

    if(n_next_cdds == 0){
      std::printf("next cdd is null!\n"); 
    }
    for (j = 0; j < n_next_cdds; ++j) {
      const cv::Rect& cdd_box = this->nextCandidates[j].rect;
      std::printf("next cdd (%d,%d/%d,%d)\n",cdd_box.x,cdd_box.y,cdd_box.width,cdd_box.height); 
      cv::Rect intersection = fit_box & cdd_box;
      const int intersectArea = intersection.area();
      this->prob[i][j] =
          (double) intersectArea /
          (double) (fit_box.area() + cdd_box.area() - intersectArea);
    }
  }
}

void KltFaceAssociator::compute_best_fits() {
  FaceCandidateList::const_iterator it;
  for (it = this->prevCandidates.cbegin();
       it != this->prevCandidates.cend();
       ++it) {
    std::printf("\nprev %d's candidate\n",it-this->prevCandidates.cbegin());
    const FaceCandidate& prev_cdd = *it;
    const cv::Rect& prev_cdd_box = prev_cdd.rect;
    std::printf("prev rect (%d,%d/%d,%d) fitted %d\n",prev_cdd_box.x,prev_cdd_box.y,prev_cdd_box.width,prev_cdd_box.height,prev_cdd.fitted);
    // Find optical flows whose outgoing point is inside `prev_cdd`.
    const MatchSet outgoing_matches =
        this->find_matches(prev_cdd_box, kOutgoing);

    // Compute fit boxes using RANSAC-based algorithm.
    const std::vector<Fit> fit_boxes =
        this->compute_fit_boxes(outgoing_matches, prev_cdd_box);

    if(!fit_boxes.size()){
        std::printf("fit_boxes is null!!!\n");
        //it = prevCandidates.erase(it);
    }

    // Find the best fit box.
    double max_inlier = 0;
    Fit best_fit;
    vector<Fit>::const_iterator it;
    for (it = fit_boxes.cbegin(); it != fit_boxes.cend(); ++it) {
      const Fit fit_box = *it;

      unsigned int num_inliers = fit_box.num_inliers;
      if (num_inliers > max_inlier) {
          std::printf("best fit is changed\n");
        max_inlier = num_inliers;
        best_fit = fit_box;
      }

    }
    best_fit.num_inliers = max_inlier;
    std::printf("best fit's num inlier is %d\n",best_fit.num_inliers);
    this->best_fits_.push_back(best_fit);
  }
}

KltFaceAssociator::MatchSet KltFaceAssociator::find_matches(
    const cv::Rect& rect,
    const MatchPointSelection point_selection) const {
  MatchSet found_matches;
  std::vector<SparseOptflow>::const_iterator it;
  for (it = this->optflows_.cbegin(); it != this->optflows_.cend(); ++it) {
    const SparseOptflow& optflow = *it;
    bool is_inside = false;
    if (point_selection == kOutgoing) {
      is_inside = rect.contains(optflow.prev_point);
    } else if (point_selection == kIncoming) {
      is_inside = rect.contains(optflow.next_point);
    }
    cv::Point2d diff;
    diff = optflow.next_point - optflow.prev_point;
      
    const cv::Size& frame_size = this->frame_size_;
    int optflow_distance_thres = rect.width / 10 + frame_size.width * 0.02;
    if (is_inside && cv::norm(diff) < optflow_distance_thres) {
      //std::printf("optflow distance %f/%d\n",cv::norm(diff),optflow_distance_thres);
      Match m = std::make_pair<cv::Point2d, cv::Point2d>(optflow.prev_point,
                                                         optflow.next_point);
      found_matches.insert(m);
    }
  }
  printf("found_matches size is %d\n",found_matches.size());
  return found_matches;
}

KltFaceAssociator::MatchSet KltFaceAssociator::find_matches_in_rect(
    const cv::Rect& rect,
    const MatchSet& matches) const {
  MatchSet found_matches;
  MatchSet::const_iterator it;
  for (it = matches.cbegin(); it != matches.cend(); ++it) {
    Match match = *it;
      bool is_inside = false;
      is_inside = rect.contains(match.first);
    if (is_inside) {
      found_matches.insert(match);
    }
  }
  return found_matches;
}

int KltFaceAssociator::compute_inlier(const MatchSet& matches, const Fit& fit_box) const {
   
  MatchSet::const_iterator it;
  int cnt = 0;

  for (it = matches.cbegin(); it != matches.cend(); ++it) {
    Match match = *it;

    cv::Point before = match.first;
    cv::Point after = match.second;
    cv::Point transformed;

    transformed.x = (before.x - fit_box.origin.x) * fit_box.sx + fit_box.a + fit_box.origin.x;
    transformed.y = (before.y - fit_box.origin.y) * fit_box.sy + fit_box.b + fit_box.origin.y;

    cv::Point diff = after - transformed;

    const cv::Size& frame_size = this->frame_size_;
    int inlier_distance_thres = frame_size.width * 0.2;

    if(cv::norm(diff)<inlier_distance_thres)
        cnt++;
  }
  // std::printf("num of inlier is %d/%d\n",cnt,matches.size());
  return cnt;
}

std::vector<KltFaceAssociator::Fit> KltFaceAssociator::compute_fit_boxes(
    const MatchSet& matches,
    const cv::Rect& base_rect) const {
  // Return value for this method. The list of computed fit boxes.
  std::vector<Fit> ret;
  // The number of given matches.
  const MatchSet::size_type num_matches = matches.size();
  // Const-iterators for (random-)picking two matches.
  MatchSet::const_iterator it1, it2;
  
  int num_inlier;
 
  std::printf("ret size is %d\n",ret.size());
  

  const cv::Size& frame_size = this->frame_size_;
  int fit_box_size_thres = frame_size.width * 0.025;

  // If the number of matches is too small for sampling, use all of them and
  // return the single fit box.
  if (num_matches == 2) {
    it1 = matches.cbegin();
    it2 = matches.cend();
    std::advance(it2, -1);
    const Match& match1 = *it1;
    const Match& match2 = *it2;
    Fit fit_box;
    this->compute_fit_box(base_rect, match1, match2, &fit_box);
    if(fit_box.box.width<fit_box_size_thres || fit_box.box.height<fit_box_size_thres){
      std::printf("fit box is too small %d\n",fit_box.box.width);
      return ret;
    }
    std::printf("chk inlier number\n");
    fit_box.num_inliers = this->compute_inlier(matches, fit_box);
    int inlier_thres = base_rect.width / 10 * 0.9;
    if(fit_box.num_inliers>=inlier_thres)
      ret.push_back(fit_box);
  }else if(num_matches<2){
    // No fit box
    std::printf("num matches is less then 2\n");
    return ret;
  }
  else {
    std::vector< std::pair<unsigned int, unsigned int> > idx_pairs =
        KltFaceAssociator::list_index_pairs(num_matches, true);
    ret.reserve(UGPROJ_ASSOCIATOR_SIFT_TRIAL_COUNT);
    for (const auto& idx_pair : idx_pairs) {
      it1 = it2 = matches.cbegin();
      std::advance(it1, idx_pair.first);
      std::advance(it2, idx_pair.second);
      const Match& match1 = *it1;
      const Match& match2 = *it2;
      Fit fit_box;
      if (!this->compute_fit_box(base_rect, match1, match2, &fit_box)) {
        std::printf("compute_fit_box failed\n");
        continue;
      }
    if(fit_box.box.width<fit_box_size_thres || fit_box.box.height<fit_box_size_thres){
        std::printf("fit box is too small %d\n",fit_box.box.width);
        continue;
      }
      // inlier chk
      fit_box.num_inliers = this->compute_inlier(matches, fit_box);
      int inlier_thres = base_rect.width / 10 * 0.9;
      //std::printf("num inliers thres %d/%d\n",fit_box.num_inliers,inlier_thres);
      if(fit_box.num_inliers>=inlier_thres)
        ret.push_back(fit_box);
      if (ret.size() >= UGPROJ_ASSOCIATOR_SIFT_TRIAL_COUNT) {
        break;
      }
      
    }
  }

  return ret;
}

bool KltFaceAssociator::compute_fit_box(const cv::Rect& base_rect,
                                        const Match& match1,
                                        const Match& match2,
                                        Fit* fit_box) const {
  // The top-left point of base_rect is needed to set this as origin for
  // computation.
  cv::Point origin = base_rect.tl();

  fit_box->origin = origin;
  // Point p_i_j is the outgoing (i = 1) / incoming (i = 2) point of the j-th
  // match.
  cv::Point p_1_1 = match1.first;
  cv::Point p_1_2 = match2.first;
  cv::Point p_2_1 = match1.second;
  cv::Point p_2_2 = match2.second;

  double x1, y1, x2, y2;
  x1 = p_1_1.x;
  y1 = p_1_1.y;
  x2 = p_1_2.x;
  y2 = p_1_2.y;

  double sx1, sy1, sx2, sy2;
  sx1 = p_2_1.x;
  sy1 = p_2_1.y;
  sx2 = p_2_2.x;
  sy2 = p_2_2.y;

  // solve linear system
  Eigen::MatrixXd matA(4, 4);
  Eigen::Vector4d matB;
  Eigen::VectorXd matX;
  double sx, sy, a, b;
  matA << x1 - origin.x, 0, 1, 0,
          x2 - origin.x, 0, 1, 0,
          0, y1 - origin.y, 0, 1,
          0, y2 - origin.y, 0, 1;
  matB << sx1 - origin.x,
          sx2 - origin.x,
          sy1 - origin.y,
          sy2 - origin.y;
  matX = matA.colPivHouseholderQr().solve(matB);
  sx = matX[0];
  sy = matX[1];
  a = matX[2];
  b = matX[3];
  
  fit_box->sx = sx;
  fit_box->sy = sy;
  fit_box->a = a;
  fit_box->b = b;

  if (sx >= UGPROJ_ASSOCIATOR_SIFT_SCALE_THRESHOLD ||
    1/sx >= UGPROJ_ASSOCIATOR_SIFT_SCALE_THRESHOLD ||
    sy >= UGPROJ_ASSOCIATOR_SIFT_SCALE_THRESHOLD ||
    1/sy >= UGPROJ_ASSOCIATOR_SIFT_SCALE_THRESHOLD) {
    return false;
  }

  int fitbox_l = base_rect.x + a;
  int fitbox_t = base_rect.y + b;
  int fitbox_r = (int)(fitbox_l + sx * base_rect.width);
  int fitbox_b = (int)(fitbox_t + sy * base_rect.height);

  // keep in boundary
  const cv::Size& frame_size = this->frame_size_;
  fitbox_l = std::min( std::max(0, fitbox_l), frame_size.width);
  fitbox_t = std::min( std::max(0, fitbox_t), frame_size.height);
  fitbox_r = std::min( std::max(0, fitbox_r), frame_size.width);
  fitbox_b = std::min( std::max(0, fitbox_b), frame_size.height);

  fit_box->box.x = fitbox_l;
  fit_box->box.y = fitbox_t;
  fit_box->box.width = fitbox_r - fitbox_l;
  fit_box->box.height = fitbox_b - fitbox_t;

  if(fit_box->box.width<0 ||
     fit_box->box.height<0 ||
     (double)fit_box->box.width/(double)fit_box->box.height > 1.5 ||
     (double)fit_box->box.height/(double)fit_box->box.width > 1.5){
      std::printf("width %f height %f\n",fit_box->box.width,fit_box->box.height);
      return false;
  }

  return true;
}

std::vector< std::pair<unsigned int, unsigned int> >
KltFaceAssociator::list_index_pairs(unsigned int size, bool shuffle) {
  std::vector< std::pair<unsigned int, unsigned int> > ret;
  ret.reserve(size * (size - 1) / 2);
  for (unsigned int i = 0; i < size - 1; ++i) {
    for (unsigned int j = i + 1; j < size; ++j) {
      ret.push_back(std::pair<unsigned int, unsigned int>(i, j));
    }
  }
  if (shuffle) {
    static std::random_device rd;
    std::mt19937 urng(rd());
    std::shuffle(ret.begin(), ret.end(), urng);
  }
  return ret;
}
