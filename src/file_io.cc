#include "file_io.h"
#include <opencv2/highgui/highgui.hpp>

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
  // Parsing metadata file and mapping file to tracklet and tracked_positions
  
  char metadataFilename[256];
  std::sprintf(metadataFilename,"%s",cfg.metadata_filepath.string().c_str());

  cv::FileStorage fs(metadataFilename,cv::FileStorage::READ);
  
  int trackletCount = (int)fs["trackletCount"];

  cv::FileNode tracklets = fs["tracklets"];
  cv::FileNodeIterator it = tracklets.begin(), it_end = tracklets.end();
  std::vector<int> frame_indices;
   
  for(int i = 1;it != it_end;++it,i++){
      
      this->tracklets_.push_back(ugproj::FaceTracklet(i-1));
      
      std::string trackletFilename = cfg.input_dirpath.string();
      trackletFilename += "tracklet_";
      trackletFilename += std::to_string(i);
      trackletFilename += ".png";

      cv::Mat trackletMat = cv::imread(trackletFilename,CV_LOAD_IMAGE_COLOR);
      
      if(!trackletMat.data){
          std::printf("Could not open or find the image\n");
        return -1;
      }

      (*it)["frame_indices"] >> frame_indices;

      static const int kSize = 64;
      static const int kNCols = 16;

      for(int j=0;j<(int)frame_indices.size();j++){
        const cv::Rect roi((j % kNCols) * kSize, (j / kNCols) * kSize,
                          kSize, kSize);

        cv::Mat cropped = trackletMat(roi);
     
        ugproj::Face newFace(frame_indices[j-1],cropped,i);
        this->tracklets_[i-1].add_face(newFace);
      }

      const auto iterators = this->tracklets_[i-1].face_iterators();
      for(auto it = iterators.first; it != iterators.second; ++it){
        const ugproj::Face& f = *it;
      }
  }

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
