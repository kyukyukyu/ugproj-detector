#ifndef UGPROJ_STRUCTURE_HEADER
#define UGPROJ_STRUCTURE_HEADER

#include <string>
#include <utility>
#include <vector>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>

namespace ugproj {

class Face;

typedef unsigned long temp_idx_t;
typedef unsigned int tracklet_id_t;

typedef std::vector<Face> FaceList;

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

struct Configuration {
  struct ScanSection {
    // Frame size at which video will be scanned. Default value is (-1, -1),
    // which represents the original frame size.
    cv::Size frame_size;
    // Fps at which video will be scanned.
    double target_fps;
  };
  struct DetectionSection {
    // Lower bound for skin color range in YCrCb.
    cv::Scalar skin_lower;
    // Upper bound for skin color range in YCrCb.
    cv::Scalar skin_upper;
    // Path to cascade classifier file.
    boost::filesystem::path cascade_filepath;
    // Scale factor for cascade classifier used for detection.
    double scale;
  };
  struct GfttSection {
    // Maximum number of corners detected by GFTT algorithm.
    int max_n;
    // Quality level parameter used by GFTT algorithm.
    double quality_level;
    // Minimum possible Euclidean distance between corners detected by GFTT
    // algorithm.
    double min_distance;
  };
  struct SubpixelSection {
    // Half of the side length of the search window.
    int window_size;
    // Half of the size of the dead region in the middle of the search zone.
    // -1 indicates that there is no such size.
    int zero_zone_size;
    // Termination criteria for iterative search.
    cv::TermCriteria term_crit;

    SubpixelSection() {
      this->term_crit.type = cv::TermCriteria::COUNT + cv::TermCriteria::EPS;
    }
  };
  struct LucasKanadeSection {
    // Size of the search window at each pyramid level.
    int window_size;
    // Maximum number of pyramid levels.
    int max_level;
    // Termination criteria for iterative search.
    cv::TermCriteria term_crit;
    // Coefficient for threshold on optical flow length.
    double coeff_thres_len;

    LucasKanadeSection() {
      this->term_crit.type = cv::TermCriteria::COUNT + cv::TermCriteria::EPS;
    }
  };
  struct AssociationSection {
    // Threshold for probability used during association.
    double threshold;
    // Association method. should be one of 'intersect', and 'optflow'.
    AssociationMethod method;
    // Coefficient for threshold on the length of each optical flow.
    double coeff_thres_optflow;
    // Coefficient for threshold on the length of each computed match.
    double coeff_thres_match;
    // Coefficient for threshold on the size of each fit box.
    double coeff_thres_box;
    // Coefficient for threshold on the number of inliers in each fit box.
    double coeff_thres_inlier;
    // Coefficient for threshold on the aspect ratio of each fit box. Should
    // be greater than or equal to 1.0.
    double coeff_thres_aspect;
  };
  struct ClusteringSection {
    // The number of clusters to be found.
    int k;
    // Termination criteria for k-means.
    cv::TermCriteria term_crit;
    // The number of attempts with different initial labellings.
    int attempts;
    // Path to Flandmark model data file.
    boost::filesystem::path flm_model_filepath;
  };
  struct OutputSection {
    // The size of (square) face expressed in tracklet images. The value of
    // this variable is the length of one side. Defaults to 64.
    unsigned int face_size;
    // The number of columns in tracklet images. Defaults to 16.
    unsigned int n_cols_tracklet;
  };
  // Loads configuration from command line arguments and configuration file
  // whose path is provided as command line argument. Returns 0 on successful
  // loading, otherwise nonzero value.
  int load(int argc, const char** argv);
  boost::filesystem::path video_filepath;
  boost::filesystem::path output_dirpath;
  ScanSection scan;
  DetectionSection detection;
  GfttSection gftt;
  SubpixelSection subpixel;
  LucasKanadeSection lucas_kanade;
  AssociationSection association;
  ClusteringSection clustering;
  OutputSection output;
};

class Face {
  public:
    temp_idx_t frameIndex;
    cv::Rect rect;
    cv::Mat image;
    tracklet_id_t tracklet_id;
    int fitted = 0;

    Face(const temp_idx_t frameIndex, const cv::Rect& rect,
                  const cv::Mat& image) :
        frameIndex(frameIndex), rect(rect), image(image), tracklet_id(0) {};
    Face(const temp_idx_t frameIndex, const cv::Mat& image,
                  const tracklet_id_t tracklet_id) :
        frameIndex(frameIndex), image(image), tracklet_id(tracklet_id) {};
    Face(const Face& f) {
      this->frameIndex = f.frameIndex;
      this->rect = f.rect;
      f.image.copyTo(this->image);
      this->tracklet_id = f.tracklet_id;
      this->fitted = f.fitted;
    }
    cv::Mat resized_image(int size) const;
    cv::Mat get_image() const;
};

class FaceTracklet {
  private:
    std::vector<Face> faces;

  public:
    typedef tracklet_id_t id_type;
    const id_type id;
    FaceTracklet(id_type id) : id(id) {};
    FaceTracklet(id_type id, const Face& f);
    void add_face(const Face& f);
    std::pair<FaceList::const_iterator,
              FaceList::const_iterator> face_iterators() const {
      return {this->faces.cbegin(), this->faces.cend()};
    }
};

} // ugproj

#endif
