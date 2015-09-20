#ifndef UGPROJ_FILEIO_H_
#define UGPROJ_FILEIO_H_

#include "structure.h"

#include <string>
#include <unordered_map>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>

namespace ugproj {

// Virtual class which is responsible for loading required filepaths from
// configuration and initializing related objects.
class FileInput {
  public:
    // Loads required paths of files and/or directories from configuration
    // object and initializes objects related to them. Returns 0 if successful,
    // otherwise nonzero value.
    virtual int open(const Configuration& cfg) = 0;
};

// Class which is responsible for file inputs used in tracker. First, loads
// paths for these files from configuration:
//
// - Input video file
// - Cascade classifier file
//
// Then, initializes objects related to each file.
class TrackerFileInput : public FileInput {
  public:
    int open(const Configuration& cfg);
    cv::VideoCapture& video();
    cv::CascadeClassifier& cascade();

  private:
    cv::VideoCapture video_;
    cv::CascadeClassifier cascade_;
};

class FileWriter {
  public:
    int init(const Configuration& cfg);
    int write_image(const cv::Mat& image, const std::string& filename) const;
    // Opens a video file to write with given key string and properties.
    // Returns non-zero value if the file is not opened correctly, otherwise 0.
    int open_video_file(
        const std::string& key,
        const std::string& filename,
        const double fps,
        const cv::Size& frame_size);
    // Writes a new frame into video file with given key string. Returns
    // non-zero value if writing was not successful, otherwise 0.
    int write_video_frame(const cv::Mat& frame, const std::string& key);

  private:
    boost::filesystem::path output_path_;
    std::unordered_map<std::string, cv::VideoWriter> video_files_;
};

}  // namespace ugproj

#endif  // UGPROJ_FILEIO_H_
