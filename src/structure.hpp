#ifndef UGPROJ_STRUCTURE_HEADER
#define UGPROJ_STRUCTURE_HEADER

#include "celiu-optflow/optical_flow.h"

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace ugproj {

typedef unsigned long temp_idx_t;
typedef unsigned int face_id_t;
typedef opticalflow::MCImageDoubleX OptFlowArray;

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
    const temp_idx_t frameIndex;
    const cv::Rect rect;
    const cv::Mat image;
    face_id_t faceId;

    FaceCandidate(const temp_idx_t frameIndex, const cv::Rect& rect,
                  cv::Mat& image) :
        frameIndex(frameIndex), rect(rect), image(image), faceId(0) {};
};

class Face {
  private:
    std::vector<FaceCandidate> candidates;

  public:
    typedef face_id_t id_type;
    const id_type id;
    Face(id_type id) : id(id) {};
    Face(id_type id, FaceCandidate& candidate);
    void addCandidate(FaceCandidate& candidate);
};

} // ugproj

#endif
