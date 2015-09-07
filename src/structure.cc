#include "structure.h"

using namespace ugproj;

cv::Mat FaceCandidate::resized_image(int size) const {
  cv::Mat resized;
  const cv::Size& orig_size = this->rect.size();
  double f;
  if (orig_size.width >= orig_size.height) {
    f = (double) size / (double) orig_size.height;
  } else {
    f = (double) size / (double) orig_size.width;
  }
  cv::resize(this->image, resized,
             cv::Size(0, 0),    /* to make use of fx and fy */
             f, f);
  if (resized.rows != resized.cols) {
    // Resized image is not square-sized.
    cv::Rect roi;
    roi.x = (resized.cols - size) / 2;
    roi.y = (resized.rows - size) / 2;
    roi.width = roi.height = size;
    resized = cv::Mat(resized, roi);
  }
  return resized;
}

Face::Face(id_type id, const FaceCandidate& candidate): id(id) {
    addCandidate(candidate);
}

void Face::addCandidate(const FaceCandidate& candidate) {
    candidates.push_back(candidate);
}
