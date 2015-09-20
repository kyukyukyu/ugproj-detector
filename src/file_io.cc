#include "file_io.h"

namespace fs = boost::filesystem;

namespace ugproj {

int TrackerFileInput::open(const Configuration& cfg) {
  if (!this->video_.open(cfg.video_filepath.native())) {
    return 1;
  }
  if (!this->cascade_.load(cfg.detection.cascade_filepath.native())) {
    return 1;
  }
  return 0;
}

cv::VideoCapture& TrackerFileInput::video() {
  return this->video_;
}

cv::CascadeClassifier& TrackerFileInput::cascade() {
  return this->cascade_;
}

int ClustererFileInput::open(const Configuration& cfg) {
  // Load from input-dir
  return 0;
}

const std::vector<unsigned long>& ClustererFileInput::tracked_positions() const{
  return this->tracked_positions_;
}

const std::vector<ugproj::FaceTracklet>& ClustererFileInput::tracklets() const{
  return this->tracklets_;
}

int FileWriter::init(const Configuration& cfg) {
  this->output_path_ = cfg.output_dirpath;
  if (!fs::is_directory(this->output_path_) &&
      !fs::create_directory(this->output_path_)) {
    return 1;
  }
  return 0;
}

int FileWriter::write_image(const cv::Mat& image,
                            const std::string& filename) const {
  fs::path filepath = this->output_path_ / filename;
  if (!cv::imwrite(filepath.native(), image)) {
    return 1;
  }
  return 0;
}

const std::string FileWriter::output_path() const{
  return this->output_path_.string();
}

int FileWriter::open_video_file(
    const std::string& key,
    const std::string& filename,
    const double fps,
    const cv::Size& frame_size) {
  auto search = this->video_files_.find(key);
  if (search != this->video_files_.end()) {
    // A video file with same key string is already open.
    return 1;
  }
  fs::path filepath = this->output_path_ / filename;
  this->video_files_[key] = cv::VideoWriter(
      filepath.native(),
      CV_FOURCC('X', 'V', 'I', 'D'),
      fps,
      frame_size);
  return !this->video_files_[key].isOpened();
}

int FileWriter::write_video_frame(
    const cv::Mat& frame,
    const std::string& key) {
  auto search = this->video_files_.find(key);
  if (search == this->video_files_.end()) {
    // A video file with given key string does not exist.
    return 1;
  }
  this->video_files_[key].write(frame);
  return 0;
}

}  // namespace ugproj
