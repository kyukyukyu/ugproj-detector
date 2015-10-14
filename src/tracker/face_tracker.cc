#include "face_tracker.h"

#include <cstdio>

#include "../visualizer.h"
#include "associator.h"

namespace ugproj {

const char* FaceTracker::kVideoKey = "result";
const char* FaceTracker::kVideoFilename = "result.avi";

FaceTracker::FaceTracker() {
  this->input_ = NULL;
  this->writer_ = NULL;
  this->cfg_ = NULL;
}

int FaceTracker::set_input(TrackerFileInput* input) {
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

int FaceTracker::track(std::vector<unsigned long>* tracked_positions,
                       std::vector<FaceTracklet>* tracklets) {
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
  // The list of faces detected in current grabbed frame.
  FaceList* curr_faces = NULL;
  // The list of faces detected in previous grabbed frame.
  FaceList* prev_faces = NULL;
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
      curr_faces = new FaceList();
      curr_optflows = new std::vector<SparseOptflow>();
      ret = this->track_frame(curr_index, prev_frame, curr_frame,
                              prev_faces, *prev_optflows, &detector,
                              curr_faces, curr_optflows, tracklets);
      if (ret != 0) {
        break;
      }
      std::puts("done.");

      // Write tracking result of current grabbed frame to file(s).
      ret = this->write_result(curr_index, *tracked_positions,
                               sx, sy, curr_frame_orig,
                               *curr_faces, *curr_optflows);
      if (ret != 0) {
        break;
      }

      // Get ready for next tracking.
      if (!curr_faces->empty()) {
        last_index_detected = curr_index;
      }
      if (curr_index > 0) {
        delete prev_faces;
      }
      prev_faces = curr_faces;
      prev_optflows = curr_optflows;
      curr_frame.copyTo(prev_frame);
      ++curr_index;
    }
    ++pos;
  }

  delete prev_faces;

  ret = this->write_mapping_file(*tracked_positions);

  ret = this->write_tracklet_metadata(*tracklets,*tracked_positions);
  if(ret != 0)
      return ret;

  for (const FaceTracklet& f : *tracklets) {
    ret = this->write_tracklet(f);
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
    const FaceList* prev_faces,
    const std::vector<SparseOptflow>& prev_optflows,
    FaceDetector* detector,
    FaceList* curr_faces,
    std::vector<SparseOptflow>* curr_optflows,
    std::vector<FaceTracklet>* tracklets) {
  std::printf("Detecting faces... ");
  this->detect_faces(curr_index, curr_frame, detector, curr_faces);
  std::printf("done. Found %lu faces.\n", curr_faces->size());

  if (prev_faces == NULL) {
    // This frame is the first scanned one. For each of faces detected in this
    // frame, face tracklet should be created and added to the list of
    // tracklets.
    std::vector<FaceTracklet>::size_type tracklet_id = 1;
    for (auto& face : *curr_faces) {
      FaceTracklet tracklet(tracklet_id);
      face.tracklet_id = tracklet_id;
      tracklet.add_face(face);
      tracklets->push_back(tracklet);
      ++tracklet_id;
    }
    std::printf("no prev faces");
    return 0;
  }

  std::printf("Computing Lucas-Kanade optical flow between previous frame and "
              "current frame... ");
  this->compute_optflow(prev_frame, curr_frame,
                        *prev_faces, prev_optflows,
                        curr_optflows);
  std::printf("done.\n");
  std::printf("Associating detected faces between previous frame and current "
              "frame... ");
  KltFaceAssociator associator(*tracklets,
                               *prev_faces, *curr_faces,
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
    const FaceList& curr_faces,
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
  for (FaceList::const_iterator it = curr_faces.cbegin();
       it != curr_faces.cend();
       ++it) {
    const cv::Rect& orig_rect = it->rect;
    const cv::Rect face(orig_rect.x * sx, orig_rect.y * sy,
                        orig_rect.width * sx, orig_rect.height * sy);
    const auto face_id = it->tracklet_id;
    const auto fitted = it->fitted;
    const cv::Scalar& color =
        colors[face_id % (sizeof(colors) / sizeof(cv::Scalar))];
    cv::rectangle(image, face.tl(), face.br(), color);
    if(fitted == 0){ // detected face
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
                              FaceList* curr_faces) {
  std::vector<cv::Rect> rects;
  detector->detectFaces(curr_frame, rects);
  for (std::vector<cv::Rect>::const_iterator it = rects.cbegin();
       it != rects.cend();
       ++it) {
    cv::Mat img_face(curr_frame, *it);
    Face f(curr_index, *it, img_face);
    curr_faces->push_back(f);
  }
  return 0;
}

int FaceTracker::compute_optflow(
    const cv::Mat& prev_frame,
    const cv::Mat& curr_frame,
    const FaceList& prev_faces,
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
      this->prepare_points_for_lk(prev_gray, prev_faces, prev_optflows);

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
    const FaceList& faces,
    const std::vector<SparseOptflow>& optflows) {
  // Points in the frame at which optical flows will be computed.
  std::vector<cv::Point2f> points;
  // Configuration section objects.
  const auto& cfg_gftt = this->cfg_->gftt;
  const auto& cfg_subpixel = this->cfg_->subpixel;

  // Compute mask for GFTT.
  cv::Mat mask;
  bool run_gftt = this->compute_roi_gftt(gray_frame.size(),
                                         faces,
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
    const FaceList& faces,
    const std::vector<SparseOptflow>& optflows,
    cv::Mat* roi) {
  roi->create(frame_size, CV_8UC1);
  roi->setTo(cv::Scalar(0));
  bool run_gftt = false;
  for (const Face& f : faces) {
    const cv::Rect& rect = f.rect;
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

int FaceTracker::write_tracklet(const FaceTracklet& tracklet) {
  std::vector< FaceRange<FaceList::const_iterator> > face_ranges;
  face_ranges.push_back(tracklet.face_iterators());
  const auto& cfg_output = this->cfg_->output;
  const auto& img_tracklet =
      visualize_faces(face_ranges.cbegin(), face_ranges.cend(),
                      cfg_output.face_size, cfg_output.n_cols_tracklet);
  // Prepare the filename.
  char filename[256];
  std::sprintf(filename, "tracklet_%d.png", tracklet.id);
  // Write to file.
  return this->writer_->write_image(img_tracklet, filename);
}

int FaceTracker::write_mapping_file(
    const std::vector<unsigned long>& tracked_positions) {

  std::string output_path = this->writer_->output_path();

  char filename[256];
  std::sprintf(filename, "%s/mapping.yaml",output_path.c_str());

  cv::FileStorage fs(filename,cv::FileStorage::WRITE);
  fs << "frame_positions" << "[:";
  int trackedCount = tracked_positions.size();
  for(int i = 0;i<trackedCount;i++){
      int temp = tracked_positions[i];
      fs << temp;
  }
  fs << "]";
  fs.release();

  return 0;
}
int FaceTracker::write_tracklet_metadata(
    const std::vector<FaceTracklet>& tracklets,
    const std::vector<unsigned long>& tracked_positions) {
  // TODO: Move this constant to class declaration.

  std::string output_path = this->writer_->output_path();

  char filename[256];
  std::sprintf(filename, "%s/tracklet.yaml",output_path.c_str());

  cv::FileStorage fs(filename,cv::FileStorage::WRITE);

  int trackletCount = tracklets.size();

  fs << "trackletCount" << trackletCount;
  fs << "tracklets" << "[";
  for (const FaceTracklet& ft : tracklets) {
    const auto iterators = ft.face_iterators();
    fs << "{:" << "frame_indices" << "[:";
    int i = 0;
    for (auto it = iterators.first; it != iterators.second; ++it, ++i) {
      const Face& f = *it;
      int frameIndex = f.frameIndex;
      fs << frameIndex;
    }
    fs << "]" << "}";
  }
  fs << "]";
  fs.release();

  return 0;
}
}  // namespace ugproj
