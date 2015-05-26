#include "file_io.h"

namespace fs = boost::filesystem;

namespace ugproj {

int FileInput::open(const Arguments& args) {
  if (!this->video_.open(args.video_filename)) {
    return 1;
  }
  if (!this->cascade_.load(args.cascade_filename)) {
    return 1;
  }
  return 0;
}

cv::VideoCapture& FileInput::video() {
  return this->video_;
}

cv::CascadeClassifier& FileInput::cascade() {
  return this->cascade_;
}

int FileWriter::init(const Arguments& args) {
  this->output_path_ = args.output_dir;
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

}  // namespace ugproj
