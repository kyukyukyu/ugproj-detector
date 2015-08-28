#ifndef UGPROJ_STRUCTURE_HEADER
#define UGPROJ_STRUCTURE_HEADER

#ifndef UGPROJ_SUPPRESS_CELIU
#include "celiu-optflow/optical_flow.h"
#endif

#include <string>
#include <utility>
#include <vector>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>

namespace ugproj {

class FaceCandidate;

typedef unsigned long temp_idx_t;
typedef unsigned int face_id_t;

#ifndef UGPROJ_SUPPRESS_CELIU
typedef opticalflow::MCImageDoubleX OptFlowArray;
#endif

typedef std::vector<FaceCandidate> FaceCandidateList;

// Represents single sparse optflow computed between two images.
struct SparseOptflow {
  // The position of point in the former image.
  cv::Point2f prev_point;
  // The position of point in the latter image.
  cv::Point2f next_point;
  // True if this sparse optical flow is valid.
  bool found;
};

enum AssociationMethod {
  ASSOC_INTERSECT = 0,
  ASSOC_OPTFLOW,
  ASSOC_SIFT,
};

struct Arguments {
  boost::filesystem::path video_filepath;
  boost::filesystem::path cascade_filepath;
  boost::filesystem::path output_dirpath;
  double target_fps;
  double detection_scale;
  double assoc_threshold;
  AssociationMethod assoc_method;
};

class FaceCandidate {
  public:
    temp_idx_t frameIndex;
    cv::Rect rect;
    cv::Mat image;
    face_id_t faceId;
    int fitted = 0;

    FaceCandidate(const temp_idx_t frameIndex, const cv::Rect& rect,
                  const cv::Mat& image) :
        frameIndex(frameIndex), rect(rect), image(image), faceId(0) {};
    FaceCandidate(const FaceCandidate& fc) {
      this->frameIndex = fc.frameIndex;
      this->rect = fc.rect;
      fc.image.copyTo(this->image);
      this->faceId = fc.faceId;
      this->fitted = fc.fitted;
    }
    cv::Mat resized_image(int size) const;
};

class Face {
  private:
    std::vector<FaceCandidate> candidates;

  public:
    typedef face_id_t id_type;
    const id_type id;
    Face(id_type id) : id(id) {};
    Face(id_type id, const FaceCandidate& candidate);
    void addCandidate(const FaceCandidate& candidate);
    std::pair<FaceCandidateList::const_iterator,
              FaceCandidateList::const_iterator> candidate_iterators() const {
      return {this->candidates.cbegin(), this->candidates.cend()};
    }
};

} // ugproj

#endif
