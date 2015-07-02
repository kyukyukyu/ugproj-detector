#ifndef UGPROJ_STRUCTURE_HEADER
#define UGPROJ_STRUCTURE_HEADER

#ifndef UGPROJ_SUPPRESS_CELIU
#include "celiu-optflow/optical_flow.h"
#endif

#include <string>
#include <vector>
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
  std::string video_filename;
  std::string cascade_filename;
  std::string output_dir;
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

    FaceCandidate(const temp_idx_t frameIndex, const cv::Rect& rect,
                  const cv::Mat& image) :
        frameIndex(frameIndex), rect(rect), image(image), faceId(0) {};
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
};

} // ugproj

#endif
