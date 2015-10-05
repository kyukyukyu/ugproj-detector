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

ClustererFileInput::~ClustererFileInput() {
  // Free memory space for Flandmark model.
  flandmark_free(this->flm_model_);
}

int ClustererFileInput::open(const Configuration& cfg) {
  const boost::filesystem::path& output_dirpath = cfg.output_dirpath;
  // Load mapping file and parse the data into the mapping between frame
  // indices and frame positions.
  boost::filesystem::path mapping_filepath = output_dirpath / "mapping.yaml";
  cv::FileStorage mapfs(mapping_filepath.native(), cv::FileStorage::READ);

  // Since FileNode from OpenCV does not support parsing numbers into unsigned
  // long type, temporary variable is required so that numbers can be casted
  // manually.
  std::vector<int> frame_positions;
  mapfs["frame_positions"] >> frame_positions;
  this->tracked_positions_.reserve(frame_positions.size());
  for (const auto& pos : frame_positions) {
    this->tracked_positions_.push_back(pos);
  }

  // Now done with mapping file.
  mapfs.release();

  // Load tracklet metadata file and tracklet image files, then parse the data
  // into face tracklet objects.
  boost::filesystem::path metadata_filepath = output_dirpath / "tracklet.yaml";
  cv::FileStorage metafs(metadata_filepath.native(), cv::FileStorage::READ);

  const cv::FileNode& tracklets_meta = metafs["tracklets"];
  std::vector<int> frame_indices;

  tracklet_id_t i = 1;
  boost::filesystem::path tracklet_img_filepath;
  char tracklet_img_filename[256];
  for (const auto& tracklet_meta : tracklets_meta) {
    // Create a new face tracklet object.
    FaceTracklet tracklet(i - 1);

    // Prepare tracklet image filename.
    std::sprintf(tracklet_img_filename, "tracklet_%u.png", i);
    tracklet_img_filepath = output_dirpath / tracklet_img_filename;

    // Load tracklet image file.
    cv::Mat tracklet_img = cv::imread(tracklet_img_filepath.native(),
                                      CV_LOAD_IMAGE_COLOR);
    if (!tracklet_img.data) {
      std::fputs("Could not open or find the image\n", stderr);
      return -1;
    }

    tracklet_meta["frame_indices"] >> frame_indices;

    static const int kSize = 64;
    static const int kNCols = 16;

    for (int j = 0; j < (int) frame_indices.size(); ++j) {
      const auto frame_index = frame_indices[j];
      const cv::Rect roi((j % kNCols) * kSize, (j / kNCols) * kSize,
                         kSize, kSize);
      cv::Mat cropped = tracklet_img(roi);
      ugproj::Face f(frame_index, cropped, i);
      tracklet.add_face(f);
    }

    // Append the tracklet to the tracklet list.
    this->tracklets_.push_back(tracklet);

    // Increment tracklet index.
    ++i;
  }

  // Now done with tracklet metadata file.
  metafs.release();

  // Load Flandmark model.
  const char* flm_model_filepath =
      cfg.clustering.flm_model_filepath.native().c_str();
  // Memory space pointed by this will be freed on the destruction of FileInput
  // object.
  this->flm_model_ = flandmark_init(flm_model_filepath);

  return 0;
}

const std::vector<unsigned long>& ClustererFileInput::tracked_positions() const{
  return this->tracked_positions_;
}

const std::vector<ugproj::FaceTracklet>& ClustererFileInput::tracklets() const{
  return this->tracklets_;
}

FLANDMARK_Model* ClustererFileInput::flm_model() {
  return this->flm_model_;
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
