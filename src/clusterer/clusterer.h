#ifndef UGPROJ_FACECLUSTERER_H_
#define UGPROJ_FACECLUSTERER_H_

#include "../file_io.h"

namespace ugproj {

// Does clustering on vectors that represent face tracklets according to
// configuration, and puts labels on each vector so that which tracklets belong
// to which face group is known.
//
// Setting Configuration object is required to load configuration for
// clustering.
class FaceClusterer {
  public:
    // Constructor for an object of this class. Configuration object for this
    // object should be provided as argument.
    explicit FaceClusterer(const Configuration& cfg);
    // Sets Configuration object for this object.
    void set_cfg(const Configuration& cfg);
    // Does clustering on vectors that represent face tracklets according to
    // configuration for this object, and puts labels on each vector. Vectors
    // that inside same cluster are put same label.
    // If file writer is set for this object, visualization of clustering
    // result will be written in image files as described above.
    void do_clustering(const cv::Mat& repr_faces_reduced,
                       std::vector<face_id_t>* cluster_ids);

  private:
    // Configuration used for clustering.
    const Configuration* cfg_;
};

}   // namespace ugproj

#endif  // UGPROJ_FACECLUSTERER_H_
