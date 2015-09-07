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
  this->args_ = NULL;
}

int FaceTracker::set_input(FileInput* input) {
  this->input_ = input;
  return 0;
}

int FaceTracker::set_writer(FileWriter* writer) {
  this->writer_ = writer;
  return 0;
}

int FaceTracker::set_args(const Arguments* args) {
  this->args_ = args;
  return 0;
}

int FaceTracker::track(std::vector<unsigned long>* tracked_positions) {
  // Return value of this method.
  int ret = 0;

  if (!this->args_ || !this->input_ || !this->writer_) {
    return 1;
  }

  cv::CascadeClassifier& cascade = this->input_->cascade();
  cv::VideoCapture& video = this->input_->video();
  VideoProperties video_props;
  this->get_properties(&video, &video_props);
  FaceDetector detector(cascade);

  ret = this->writer_->open_video_file(
      this->kVideoKey,
      this->kVideoFilename,
      this->args_->target_fps,
      cv::Size(video_props.frame_width, video_props.frame_height));
  if (ret) {
    return ret;
  }

  // The position of current grabbed frame.
  unsigned long pos = 0;
  // The index number of current tracking.
  temp_idx_t curr_index = 0;
  // The last index number of tracking with at lease one detected face
  // before current one.
  temp_idx_t last_index_detected;
  // The matrix representing current grabbed frame.
  cv::Mat curr_frame;
  // The matrix representing previous grabbed frame.
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

    if(pos<230 || pos>1500){
      pos++;
      continue;
    }
    // If target fps is set to zero, every frame will be tracked.
    const double target_fps = this->args_ ? this->args_->target_fps : 0.0;
    const double mod = target_fps == 0.0 ?
        0.0 : std::fmod(pos, video_props.fps / target_fps);
    static const double epsilon = std::numeric_limits<double>::epsilon();
    if (mod - 1.0 <= -epsilon) {
      video.retrieve(curr_frame);
      if (curr_frame.empty()) {
        // Something went wrong on retrieving frame.
        ret = 1;
        break;
      }

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
                               prev_frame, curr_frame,
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
                               this->args_->assoc_threshold);
  associator.associate();
  std::puts("done.");

  return 0;
}

int FaceTracker::write_result(
    const temp_idx_t curr_index,
    const std::vector<unsigned long>& tracked_positions,
    const cv::Mat& prev_frame,
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
    const cv::Rect& face = it->rect;
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
    cv::circle(image, optflow.prev_point, 3, color, -1);
    if (optflow.found) {
      cv::line(image, optflow.prev_point, optflow.next_point, color);
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
  detector->detectFaces(curr_frame, rects, this->args_->detection_scale);
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
  // Compute mask for GFTT.
  cv::Mat mask(prev_frame.size(), CV_8UC1, cv::Scalar(0));
  bool run_gftt = false;
  for (FaceCandidateList::const_iterator it_a = prev_candidates.cbegin();
       it_a != prev_candidates.cend();
       ++it_a) {
    const FaceCandidate& candidate = *it_a;
    const cv::Rect& rect = candidate.rect;
    std::vector<SparseOptflow>::const_iterator it_b;
    bool set_mask = true;
    unsigned int n_optflows = 0;
    const unsigned int thres_n_optflows = rect.width / 10;

    for (it_b = prev_optflows.cbegin(); it_b != prev_optflows.cend(); ++it_b) {
      const SparseOptflow& optflow = *it_b;
      if (rect.contains(optflow.next_point)) {
        ++n_optflows;
        if (n_optflows >= thres_n_optflows) {
          set_mask = false;
          break;
        }
      }
    }
    if (set_mask) {
      mask(rect) = cv::Scalar(1);
      run_gftt = true;
    }
  }

  // Convert RGB images to grayscale images.
  cv::Mat prev_gray;
  cv::Mat curr_gray;
  cvtColor(prev_frame, prev_gray, CV_BGR2GRAY);
  cvtColor(curr_frame, curr_gray, CV_BGR2GRAY);

  // Points in the previous frame at which the optical flow will be computed.
  std::vector<cv::Point2f> prev_points;

  // The termination criteria of the iterative search algorithm.
  using cv::TermCriteria;
  static const TermCriteria term_crit(TermCriteria::COUNT + TermCriteria::EPS,
                                      20, 0.03);
  if (run_gftt) {
    // Run GFTT on prev_frame for specified ROI.
    cv::goodFeaturesToTrack(prev_gray, prev_points,
                            FaceTracker::kGfttMaxCorners, 0.01, 10, mask);
    if (!prev_points.empty()) {
      // Refine corner locations.
      static const cv::Size subpixel_win_size(10, 10);
      cv::cornerSubPix(prev_gray, prev_points, subpixel_win_size,
                       cv::Size(-1,-1), term_crit);
    }
  }

  // Append destination points of optical flows computed for previous frame.
  for (std::vector<SparseOptflow>::const_iterator it = prev_optflows.cbegin();
       it != prev_optflows.cend();
       ++it) {
    const SparseOptflow& optflow = *it;
    if (optflow.found) {
      prev_points.push_back(optflow.next_point);
    }
  }

  if (prev_points.empty()) {
    curr_optflows->clear();
    return 0;
  }

  std::printf("Lucas Kanade\n");
  // Compute Lucas-Kanade sparse optical flow.
  std::vector<cv::Point2f> curr_points;
  std::vector<uchar> status;
  std::vector<float> error;
  static const cv::Size win_size(31, 31);
  cv::calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_points, curr_points,
                           status, error, win_size, 3, term_crit);

  // Save the result.
  const std::vector<cv::Point2f>::size_type num_curr_points =
      curr_points.size();
  curr_optflows->clear();
  curr_optflows->reserve(num_curr_points);
  for (std::vector<cv::Point2f>::size_type i = 0; i < num_curr_points; ++i) {
    SparseOptflow optflow;
    optflow.prev_point = prev_points[i];
    if (status[i] == 1) {
      // Flow is found.
      optflow.next_point = curr_points[i];
      cv::Point2d diff = optflow.next_point - optflow.prev_point;
      double optflow_distance_thres = prev_frame.cols * 0.04;
      if (cv::norm(diff) > optflow_distance_thres) {
        optflow.found = false;
      } else {
        optflow.found = true;
      }
    } else {
      optflow.found = false;
    }
    curr_optflows->push_back(optflow);
  }

  return 0;
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
