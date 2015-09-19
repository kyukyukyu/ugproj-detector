#include "associator.h"

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

void FaceAssociator::match_faces() {
  typedef FaceList::size_type size_type;

  const size_type
      prevSize = prev_faces.size(), nextSize = next_faces.size();

  for (size_type j = 0; j < nextSize; ++j) {
    double max = -1;
    size_type maxRow;

    for (size_type i = 0; i < prevSize; ++i) {
      if (prob[i][j] > max && prob[i][j] > threshold) {
        max = prob[i][j];
        maxRow = i;
      }
    }

    vector<FaceTracklet>::size_type tracklet_id;
    if (max > 0) {
      tracklet_id = prev_faces[maxRow].tracklet_id;
    } else {
      tracklet_id = tracklets.size() + 1;
      tracklets.push_back(FaceTracklet(tracklet_id));
    }
    next_faces[j].tracklet_id = tracklet_id;
    tracklets[tracklet_id - 1].add_face(next_faces[j]);
  }
}

void FaceAssociator::associate() {
  calculateProb();
  match_faces();
}

KltFaceAssociator::KltFaceAssociator(
    std::vector<FaceTracklet>& faces,
    const FaceList& prev_faces,
    FaceList& next_faces,
    const temp_idx_t next_index,
    const cv::Mat& next_frame,
    const std::vector<SparseOptflow>& optflows,
    double threshold) :
  FaceAssociator(faces, prev_faces, next_faces, threshold),
  next_index_(next_index),
  next_frame_(next_frame),
  frame_size_(next_frame.size()),
  optflows_(optflows) {}

void KltFaceAssociator::associate() {
  FaceAssociator::associate();
  // Add best fits of non-labeled faces in previous frame to
  // next_faces.
  const auto prev_size = this->prev_faces.size();
  const auto next_size = this->next_faces.size();
  for (FaceList::size_type i = 0; i < prev_size; ++i) {
    unsigned int n_overlapped = 0;
    for (FaceList::size_type j = 0; j < next_size; ++j) {
      if (this->prob[i][j] > threshold) {
        ++n_overlapped;
      }
    }
    if (n_overlapped == 0) {
      Fit best_fit = this->best_fits_[i];
      if (!best_fit.valid()) {
        continue;
      }
      const cv::Rect& face_rect = best_fit.box;
      const cv::Mat face_img(this->next_frame_, face_rect);
      const tracklet_id_t face_id = this->prev_faces[i].tracklet_id;
      Face restored(this->next_index_, face_rect, face_img);
      restored.tracklet_id = face_id;
      restored.fitted = 1;
      this->next_faces.push_back(restored);
      this->tracklets[face_id - 1].add_face(restored);
    }
  }
}

void KltFaceAssociator::calculateProb() {
  // Set random seed for random-picking matches later.
  srand(time(NULL));

  // Best fits will be saved into `this->best_fits_`.
  this->compute_best_fits();

  // Compute correspondence for each pair of faces in two frames.
  FaceList::size_type i, j, n_prev_cdds, n_next_cdds;
  n_prev_cdds = this->prev_faces.size();
  n_next_cdds = this->next_faces.size();
  for (i = 0; i < n_prev_cdds; ++i) {
    Fit& best_fit = this->best_fits_[i];
    if (!best_fit.valid()) {
      // No best fit available. Set all probability to zero.
      std::fill(prob[i], prob[i] + n_next_cdds, 0.0);
      continue;
    }

    const cv::Rect& fit_box = best_fit.box;
    for (j = 0; j < n_next_cdds; ++j) {
      const cv::Rect& cdd_box = this->next_faces[j].rect;
      cv::Rect intersection = fit_box & cdd_box;
      const int intersectArea = intersection.area();
      this->prob[i][j] =
          (double) intersectArea /
          (double) (fit_box.area() + cdd_box.area() - intersectArea);
    }
  }
}

void KltFaceAssociator::compute_best_fits() {
  FaceList::const_iterator it;
  for (it = this->prev_faces.cbegin();
       it != this->prev_faces.cend();
       ++it) {
    const Face& prev_cdd = *it;
    const cv::Rect& prev_cdd_box = prev_cdd.rect;
    // Find optical flows whose outgoing point is inside `prev_cdd`.
    const MatchSet outgoing_matches =
        this->find_matches(prev_cdd_box, kOutgoing);

    // Compute fit boxes using RANSAC-based algorithm.
    const std::vector<Fit> fit_boxes =
        this->compute_fit_boxes(outgoing_matches, prev_cdd_box);

    // Find the best fit box.
    double max_inlier = 0;
    Fit best_fit;
    vector<Fit>::const_iterator it;
    for (it = fit_boxes.cbegin(); it != fit_boxes.cend(); ++it) {
      const Fit fit_box = *it;

      unsigned int num_inliers = fit_box.num_inliers;
      if (num_inliers > max_inlier) {
        max_inlier = num_inliers;
        best_fit = fit_box;
      }

    }
    best_fit.num_inliers = max_inlier;
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
      Match m = std::make_pair<cv::Point2d, cv::Point2d>(optflow.prev_point,
                                                         optflow.next_point);
      found_matches.insert(m);
    }
  }
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
    transformed.x =
        (before.x - fit_box.origin.x) * fit_box.sx + fit_box.a +
        fit_box.origin.x;
    transformed.y =
        (before.y - fit_box.origin.y) * fit_box.sy + fit_box.b +
        fit_box.origin.y;

    cv::Point diff = after - transformed;

    const cv::Size& frame_size = this->frame_size_;
    int inlier_distance_thres = frame_size.width * 0.2;

    if (cv::norm(diff) < inlier_distance_thres) {
        cnt++;
    }
  }
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
  // Size of frame.
  const cv::Size& frame_size = this->frame_size_;
  // Threshold value for fit box size. Applied to both width and height.
  int fit_box_size_thres = frame_size.width * 0.025;
  if (num_matches < 2) {
    // No fit box can be computed.
    return ret;
  }
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
      // Failed to compute fit box.
      continue;
    }
    if (fit_box.box.width < fit_box_size_thres ||
        fit_box.box.height < fit_box_size_thres) {
      // Fit box is too small.
      continue;
    }
    // Check inliers.
    fit_box.num_inliers = this->compute_inlier(matches, fit_box);
    unsigned int inlier_thres = base_rect.width / 10 * 0.9;
    if (fit_box.num_inliers >= inlier_thres) {
      ret.push_back(fit_box);
    }
    if (ret.size() >= UGPROJ_ASSOCIATOR_SIFT_TRIAL_COUNT) {
      break;
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
  int fitbox_r = (int) (fitbox_l + sx * base_rect.width);
  int fitbox_b = (int) (fitbox_t + sy * base_rect.height);

  // keep in boundary
  const cv::Size& frame_size = this->frame_size_;
  fitbox_l = std::min(std::max(0, fitbox_l), frame_size.width);
  fitbox_t = std::min(std::max(0, fitbox_t), frame_size.height);
  fitbox_r = std::min(std::max(0, fitbox_r), frame_size.width);
  fitbox_b = std::min(std::max(0, fitbox_b), frame_size.height);

  fit_box->box.x = fitbox_l;
  fit_box->box.y = fitbox_t;
  fit_box->box.width = fitbox_r - fitbox_l;
  fit_box->box.height = fitbox_b - fitbox_t;

  if (fit_box->box.width < 0 ||
      fit_box->box.height < 0 ||
      (double) fit_box->box.width / (double) fit_box->box.height > 1.5 ||
      (double) fit_box->box.height / (double) fit_box->box.width > 1.5) {
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
