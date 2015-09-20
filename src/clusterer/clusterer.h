#ifndef UGPROJ_FACECLUSTERER_H_
#define UGPROJ_FACECLUSTERER_H_

#include "../file_io.h"

namespace ugproj {

// Does clustering on vectors that represent faces according to configuration,
// and puts cluster label on each face tracklet based on voting result of faces
// in the tracklet.
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
    // Does clustering on vectors that represent faces according to
    // configuration for this object, and puts cluster labels on each face
    // tracklet based on voting result of faces in the tracklet. Face tracklets
    // that belongs to same cluster will have the same label.
    void do_clustering(const cv::Mat& faces_reduced,
                       const std::vector<FaceTracklet>& tracklets,
                       std::vector<int>* cluster_ids);

  private:
    // Iterates over face tracklets and runs voting for cluster label of the
    // tracklet.
    void vote_for_labels(const std::vector<int>& labels_face,
                         const std::vector<FaceTracklet>& tracklets,
                         std::vector<int>* cluster_ids);
    // Configuration used for clustering.
    const Configuration* cfg_;
};

}   // namespace ugproj

#endif  // UGPROJ_FACECLUSTERER_H_
