#ifndef UGPROJ_FILEIO_H_
#define UGPROJ_FILEIO_H_

#include "structure.hpp"

#include <string>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>

namespace ugproj {

class FileInput {
  public:
    int open(const Arguments& args);
    cv::VideoCapture& video();
    cv::CascadeClassifier& cascade();

  private:
    cv::VideoCapture video_;
    cv::CascadeClassifier cascade_;
};

class FileWriter {
  public:
    int init(const Arguments& args);
    int write_image(const cv::Mat& image, const std::string& filename) const;

  private:
    boost::filesystem::path output_path_;
};

}  // namespace ugproj

#endif  // UGPROJ_FILEIO_H_
