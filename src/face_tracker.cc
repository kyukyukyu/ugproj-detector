#define UGPROJ_SUPPRESS_CELIU
#include "associator/associator.hpp"
#include "face_tracker.h"

#include <cstdio>

namespace ugproj {

const char* FaceTracker::kVideoKey = "result";
const char* FaceTracker::kVideoFilename = "result.avi";

FaceTracker::FaceTracker() {
  this->input_ = NULL;
  this->writer_ = NULL;
  this->cfg_ = NULL;
}

int FaceTracker::set_input(FileInput* input) {
  this->input_ = input;
  return 0;
}

int FaceTracker::set_writer(FileWriter* writer) {
  this->writer_ = writer;
  return 0;
}

int FaceTracker::set_cfg(const Configuration* cfg) {
  this->cfg_ = cfg;
  return 0;
}

int FaceTracker::track(std::vector<unsigned long>* tracked_positions) {
  // Return value of this method.
  int ret = 0;

  if (!this->cfg_ || !this->input_ || !this->writer_) {
    return 1;
  }

  cv::CascadeClassifier& cascade = this->input_->cascade();
  cv::VideoCapture& video = this->input_->video();
  FaceDetector detector(cascade, this->cfg_->detection);

  // Retrieve properties of input video.
  VideoProperties video_props;
  this->get_properties(&video, &video_props);
  const cv::Size orig_frame_size =
      cv::Size(video_props.frame_width, video_props.frame_height);

  ret = this->writer_->open_video_file(
      this->kVideoKey,
      this->kVideoFilename,
      this->cfg_->scan.target_fps,
      orig_frame_size);
  if (ret) {
    return ret;
  }

  // Compute scale factor for frame width and height by comparing frame
  // size in configuration and the original frame size.
  const cv::Size& cfg_frame_size = this->cfg_->scan.frame_size;
  const cv::Size& scan_frame_size =
      (cfg_frame_size.width > 0 && cfg_frame_size.height > 0) ?
      cfg_frame_size : orig_frame_size;
  const double sx =
      (double) video_props.frame_width / (double) scan_frame_size.width;
  const double sy =
      (double) video_props.frame_height / (double) scan_frame_size.height;

  // The position of current grabbed frame.
  unsigned long pos = 0;
  // The index number of current tracking.
  temp_idx_t curr_index = 0;
  // The last index number of tracking with at lease one detected face
  // before current one.
  temp_idx_t last_index_detected;
  // The matrix representing resized current grabbed frame.
  cv::Mat curr_frame;
  // The matrix representing original current grabbed frame.
  cv::Mat curr_frame_orig;
  // The matrix representing resized previous grabbed frame.
  cv::Mat prev_frame;
  // The list of face candidates detected in current grabbed frame.
  FaceCandidateList* curr_candidates = NULL;
  // The list of face candidates detected in previous grabbed frame.
  FaceCandidateList* prev_candidates = NULL;
  // The list of sparse optical flows computed at current tracking.
  std::vector<SparseOptflow>* curr_optflows = NULL;
  // The list of sparse optical flows computed at previous tracking.
  std::vector<SparseOptflow>* prev_optflows = NULL;

  while (pos < video_props.frame_count) {
    if (!video.grab()) {
      // Failed to grab next frame
      ret = 1;
      break;
    }

    // If target fps is set to zero, every frame will be tracked.
    const double target_fps = this->cfg_ ? this->cfg_->scan.target_fps : 0.0;
    const double mod = target_fps == 0.0 ?
        0.0 : std::fmod(pos, video_props.fps / target_fps);
    static const double epsilon = std::numeric_limits<double>::epsilon();
    if (mod - 1.0 <= -epsilon) {
      video.retrieve(curr_frame_orig);
      if (curr_frame_orig.empty()) {
        // Something went wrong on retrieving frame.
        ret = 1;
        break;
      }
      cv::resize(curr_frame_orig, curr_frame, scan_frame_size,
                 0, 0,              // dsize is used instead of fx, fy.
                 cv::INTER_AREA);   // Preferred method for image decimation.

      // `tracked_positions->at(curr_index)` becomes `pos`
      tracked_positions->push_back(pos);

      // Track current grabbed frame.
      std::printf("Tracking faces for frame #%ld...\n", pos);
      curr_candidates = new FaceCandidateList();
      curr_optflows = new std::vector<SparseOptflow>();
      ret = this->track_frame(curr_index, prev_frame, curr_frame,
                              prev_candidates, *prev_optflows, &detector,
                              curr_candidates, curr_optflows);
      if (ret != 0) {
        break;
      }
      std::puts("done.");

      // Write tracking result of current grabbed frame to file(s).
      ret = this->write_result(curr_index, *tracked_positions,
                               sx, sy, curr_frame_orig,
                               *curr_candidates, *curr_optflows);
      if (ret != 0) {
        break;
      }

      // Get ready for next tracking.
      if (!curr_candidates->empty()) {
        last_index_detected = curr_index;
      }
      if (curr_index > 0) {
        delete prev_candidates;
      }
      prev_candidates = curr_candidates;
      prev_optflows = curr_optflows;
      curr_frame.copyTo(prev_frame);
      ++curr_index;
    }
    ++pos;
  }

  delete prev_candidates;

  for (const Face& f : this->labeled_faces_) {
    ret = this->write_tracklet(f, *tracked_positions);
    if (ret != 0) {
      break;
    }
  }

  return ret;
}

void FaceTracker::get_properties(cv::VideoCapture* video,
                                 VideoProperties* props) {
  double prop;
  prop = video->get(CV_CAP_PROP_FPS);
  props->fps = prop;
  prop = video->get(CV_CAP_PROP_FRAME_COUNT);
  props->frame_count = static_cast<unsigned long>(prop);
  prop = video->get(CV_CAP_PROP_FRAME_WIDTH);
  props->frame_width = static_cast<int>(prop);
  prop = video->get(CV_CAP_PROP_FRAME_HEIGHT);
  props->frame_height = static_cast<int>(prop);
}

int FaceTracker::track_frame(
    const temp_idx_t curr_index,
    const cv::Mat& prev_frame,
    const cv::Mat& curr_frame,
    const FaceCandidateList* prev_candidates,
    const std::vector<SparseOptflow>& prev_optflows,
    FaceDetector* detector,
    FaceCandidateList* curr_candidates,
    std::vector<SparseOptflow>* curr_optflows) {
  std::printf("Detecting faces... ");
  this->detect_faces(curr_index, curr_frame, detector, curr_candidates);
  std::printf("done. Found %lu faces.\n", curr_candidates->size());

  if (prev_candidates == NULL) {
    std::printf("no prev candidates");
    return 0;
  }

  std::printf("Computing Lucas-Kanade optical flow between previous frame and "
              "current frame... ");
  this->compute_optflow(prev_frame, curr_frame,
                        *prev_candidates, prev_optflows,
                        curr_optflows);
  std::printf("done.\n");
  std::printf("Associating detected faces between previous frame and current "
              "frame... ");
  KltFaceAssociator associator(this->labeled_faces_,
                               *prev_candidates, *curr_candidates,
                               curr_index, curr_frame, *curr_optflows,
                               this->cfg_->association.threshold);
  associator.associate();
  std::puts("done.");

  return 0;
}

int FaceTracker::write_result(
    const temp_idx_t curr_index,
    const std::vector<unsigned long>& tracked_positions,
    const double sx,
    const double sy,
    const cv::Mat& curr_frame,
    const FaceCandidateList& curr_candidates,
    const std::vector<SparseOptflow>& curr_optflows) {
  // Return value of this method.
  int ret = 0;
  // Color constants.
  static const cv::Scalar colors[] = {
    CV_RGB(255, 0, 0),
    CV_RGB(255, 255, 0),
    CV_RGB(0, 255, 0),
    CV_RGB(0, 255, 255),
    CV_RGB(0, 0, 255),
    CV_RGB(255, 0, 255),
  };
  static const cv::Scalar& color_green = colors[2];
  static const cv::Scalar& color_red = colors[0];
  // Position of current frame.
  const unsigned long curr_pos = tracked_positions[curr_index];
  // Image matrix for current frame.
  cv::Mat image = curr_frame.clone();

  // Draw face detections and their association results.
  for (FaceCandidateList::const_iterator it = curr_candidates.cbegin();
       it != curr_candidates.cend();
       ++it) {
    const cv::Rect& orig_rect = it->rect;
    const cv::Rect face(orig_rect.x * sx, orig_rect.y * sy,
                        orig_rect.width * sx, orig_rect.height * sy);
    const auto face_id = it->faceId;
    const auto fitted = it->fitted;
    const cv::Scalar& color =
        colors[face_id % (sizeof(colors) / sizeof(cv::Scalar))];
    cv::rectangle(image, face.tl(), face.br(), color);
    if(fitted == 0){ // candidate
      cv::putText(image, std::to_string(face_id), face.tl() + cv::Point(4, 4),
                cv::FONT_HERSHEY_PLAIN, 1.0, color);
    }else{
      cv::putText(image, "new", face.tl() + cv::Point(2, 2),
                cv::FONT_HERSHEY_PLAIN, 1.0, color);
    }
  }

  // Draw optical flows.
  for (std::vector<SparseOptflow>::const_iterator it = curr_optflows.cbegin();
       it != curr_optflows.cend();
       ++it) {
    const SparseOptflow& optflow = *it;
    const cv::Scalar color = optflow.found ? color_green : color_red;
    const cv::Point2f& orig_prev_point = optflow.prev_point;
    const cv::Point2f prev_point(orig_prev_point.x * sx,
                                 orig_prev_point.y * sy);
    cv::circle(image, prev_point, 3, color, -1);
    if (optflow.found) {
      const cv::Point2f& orig_next_point = optflow.next_point;
      const cv::Point2f next_point(orig_next_point.x * sx,
                                   orig_next_point.y * sy);
      cv::line(image, prev_point, next_point, color);
    }
  }

  // Write to file.
  char filename[1024];
  std::sprintf(filename, "%ld.png", curr_pos);
  ret = this->writer_->write_image(image, filename);
  ret |= this->writer_->write_video_frame(image, this->kVideoKey);

  return ret;
}

int FaceTracker::detect_faces(const temp_idx_t curr_index,
                              const cv::Mat& curr_frame,
                              FaceDetector* detector,
                              FaceCandidateList* curr_candidates) {
  std::vector<cv::Rect> rects;
  detector->detectFaces(curr_frame, rects);
  for (std::vector<cv::Rect>::const_iterator it = rects.cbegin();
       it != rects.cend();
       ++it) {
    cv::Mat candidate_img(curr_frame, *it);
    FaceCandidate candidate(curr_index, *it, candidate_img);
    curr_candidates->push_back(candidate);
  }
  return 0;
}

int FaceTracker::compute_optflow(
    const cv::Mat& prev_frame,
    const cv::Mat& curr_frame,
    const FaceCandidateList& prev_candidates,
    const std::vector<SparseOptflow>& prev_optflows,
    std::vector<SparseOptflow>* curr_optflows) {
  std::printf("compute_optflow\n");

  // Convert RGB images to grayscale images.
  cv::Mat prev_gray;
  cv::Mat curr_gray;
  cvtColor(prev_frame, prev_gray, CV_BGR2GRAY);
  cvtColor(curr_frame, curr_gray, CV_BGR2GRAY);
  // Points in the previous frame at which the optical flow will be computed.
  std::vector<cv::Point2f> prev_points =
      this->prepare_points_for_lk(prev_gray, prev_candidates, prev_optflows);

  if (prev_points.empty()) {
    curr_optflows->clear();
    return 0;
  }

  std::printf("Lucas Kanade\n");
  this->run_lk(prev_gray, curr_gray, prev_points, curr_optflows);
  return 0;
}

std::vector<cv::Point2f> FaceTracker::prepare_points_for_lk(
    const cv::Mat& gray_frame,
    const FaceCandidateList& candidates,
    const std::vector<SparseOptflow>& optflows) {
  // Points in the frame at which optical flows will be computed.
  std::vector<cv::Point2f> points;
  // Configuration section objects.
  const auto& cfg_gftt = this->cfg_->gftt;
  const auto& cfg_subpixel = this->cfg_->subpixel;

  // Compute mask for GFTT.
  cv::Mat mask;
  bool run_gftt = this->compute_roi_gftt(gray_frame.size(),
                                         candidates,
                                         optflows,
                                         &mask);
  // Run GFTT if needed.
  if (run_gftt) {
    // Run GFTT on prev_frame for specified ROI.
    cv::goodFeaturesToTrack(gray_frame, points,
                            cfg_gftt.max_n, cfg_gftt.quality_level,
                            cfg_gftt.min_distance, mask);
    if (!points.empty()) {
      // Refine corner locations.
      const auto& term_crit = cfg_subpixel.term_crit;
      const cv::Size subpixel_win_size(cfg_subpixel.window_size,
                                       cfg_subpixel.window_size);
      const cv::Size zero_zone_size(cfg_subpixel.zero_zone_size,
                                    cfg_subpixel.zero_zone_size);
      cv::cornerSubPix(gray_frame, points, subpixel_win_size,
                       zero_zone_size, term_crit);
    }
  }

  // Append destination points of optical flows computed for previous frame.
  for (const SparseOptflow& optflow : optflows) {
    if (optflow.found) {
      points.push_back(optflow.next_point);
    }
  }

  return points;
}

bool FaceTracker::compute_roi_gftt(
    const cv::Size& frame_size,
    const FaceCandidateList& candidates,
    const std::vector<SparseOptflow>& optflows,
    cv::Mat* roi) {
  roi->create(frame_size, CV_8UC1);
  roi->setTo(cv::Scalar(0));
  bool run_gftt = false;
  for (const FaceCandidate& candidate : candidates) {
    const cv::Rect& rect = candidate.rect;
    std::vector<SparseOptflow>::const_iterator it_b;
    bool set_mask = true;
    unsigned int n_optflows = 0;
    const unsigned int thres_n_optflows = rect.width / 10;

    for (const SparseOptflow& optflow : optflows) {
      if (rect.contains(optflow.next_point)) {
        ++n_optflows;
        if (n_optflows >= thres_n_optflows) {
          set_mask = false;
          break;
        }
      }
    }
    if (set_mask) {
      (*roi)(rect) = cv::Scalar(1);
      run_gftt = true;
    }
  }
  return run_gftt;
}

void FaceTracker::run_lk(const cv::Mat& prev_gray,
                         const cv::Mat& curr_gray,
                         const std::vector<cv::Point2f> points,
                         std::vector<SparseOptflow>* optflows) {
  // Configuration section object.
  const auto& cfg_lk = this->cfg_->lucas_kanade;
  // Compute Lucas-Kanade sparse optical flow.
  std::vector<cv::Point2f> curr_points;
  std::vector<uchar> status;
  std::vector<float> error;
  const cv::Size win_size(cfg_lk.window_size, cfg_lk.window_size);
  cv::calcOpticalFlowPyrLK(prev_gray, curr_gray, points, curr_points,
                           status, error, win_size, cfg_lk.max_level,
                           cfg_lk.term_crit);

  // Populate optflows with the result.
  const std::vector<cv::Point2f>::size_type num_curr_points =
      curr_points.size();
  optflows->clear();
  optflows->reserve(num_curr_points);
  for (std::vector<cv::Point2f>::size_type i = 0; i < num_curr_points; ++i) {
    SparseOptflow optflow;
    optflow.prev_point = points[i];
    if (status[i] == 1) {
      // Flow is found.
      optflow.next_point = curr_points[i];
      cv::Point2d diff = optflow.next_point - optflow.prev_point;
      const double optflow_distance_thres =
          prev_gray.cols * cfg_lk.coeff_thres_len;
      optflow.found = cv::norm(diff) <= optflow_distance_thres;
    } else {
      optflow.found = false;
    }
    optflows->push_back(optflow);
  }
}

int FaceTracker::write_tracklet(
    const Face& f,
    const std::vector<unsigned long>& tracked_positions) {
  // TODO: Move this constant to class declaration.
  static const int kSize = 64;
  static const int kNCols = 16;
  static const cv::Scalar kColorText = CV_RGB(255, 255, 255);
  static const cv::Scalar kColorTextbox = CV_RGB(0, 0, 0);
  static const int kMarginTextbox = 4;
  // Draw tracklet.
  const auto iterators = f.candidate_iterators();
  const int n_candidates = iterators.second - iterators.first;
  const int n_rows = n_candidates / kNCols + !!(n_candidates % kNCols);
  cv::Mat tracklet(n_rows * kSize, kNCols * kSize, CV_8UC3);
  tracklet = CV_RGB(255, 255, 255);
  int i = 0;
  for (auto it = iterators.first; it != iterators.second; ++it, ++i) {
    const FaceCandidate& fc = *it;
    const cv::Rect roi((i % kNCols) * kSize, (i / kNCols) * kSize,
                       kSize, kSize);
    cv::Mat tracklet_roi(tracklet, roi);
    // Draw the image of face candidate.
    fc.resized_image(kSize).copyTo(tracklet_roi);
    // Prepare the information of face candidate.
    char c_str_frame_pos[16];
    std::sprintf(c_str_frame_pos, "%lu", tracked_positions[fc.frameIndex]);
    std::string str_frame_pos(c_str_frame_pos);
    // Compute the position for the information text and box including it.
    int baseline;
    cv::Size text_size =
        cv::getTextSize(str_frame_pos, cv::FONT_HERSHEY_PLAIN, 1.0, 1,
                        &baseline);
    const cv::Rect box_rec(kMarginTextbox, kMarginTextbox,
                           2 * kMarginTextbox + text_size.width,
                           2 * kMarginTextbox + text_size.height);
    const cv::Point text_org(2 * kMarginTextbox,
                             2 * kMarginTextbox + text_size.height);
    // Draw the information text and box including it.
    // The box should be drawn first.
    cv::rectangle(tracklet_roi, box_rec, kColorTextbox, CV_FILLED);
    cv::putText(tracklet_roi, str_frame_pos, text_org, cv::FONT_HERSHEY_PLAIN,
                1.0, kColorText);
  }
  // Prepare the filename.
  char filename[256];
  std::sprintf(filename, "face_%d.png", f.id);
  // Write to file.
  return this->writer_->write_image(tracklet, filename);
}

}  // namespace ugproj
