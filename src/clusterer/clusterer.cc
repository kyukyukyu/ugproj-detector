#include "clusterer.h"

#include <iterator>

#include "opencv2/opencv.hpp"

namespace ugproj {

FaceClusterer::FaceClusterer(const Configuration& cfg) {
  this->set_cfg(cfg);
}

void FaceClusterer::set_cfg(const Configuration& cfg) {
  this->cfg_ = &cfg;
}

void FaceClusterer::do_clustering(const cv::Mat& faces_reduced,
                                  const std::vector<FaceTracklet>& tracklets,
                                  std::vector<int>* cluster_ids) {
  // Compactness of k-means clustering result.
  double compactness;
  // Clustering section of configuration.
  const auto& cfg_cl = this->cfg_->clustering;
  // List of cluster labels for faces in tracklets. Will be populated by
  // k-means clustering.
  std::vector<int> labels_face;
  // Run k-means clustering provided by OpenCV.
  compactness = cv::kmeans(faces_reduced, cfg_cl.k, *cluster_ids,
                           cfg_cl.term_crit, cfg_cl.attempts,
                           cv::KMEANS_PP_CENTERS);
  // Put cluster labels on face tracklets.
  this->vote_for_labels(labels_face, tracklets, cluster_ids);
}

void FaceClusterer::vote_for_labels(const std::vector<int>& labels_face,
                                    const std::vector<FaceTracklet>& tracklets,
                                    std::vector<int>* cluster_ids) {
  auto it_label_face = labels_face.cbegin();
  int idx_tracklet = 0;
  for (const auto& tracklet : tracklets) {
    // Pair of iterators pointing the start and the end of faces in the
    // tracklet.
    const auto face_iterators = tracklet.face_iterators();
    // The number of faces in the tracklet.
    const auto n_faces =
        std::distance(face_iterators.first, face_iterators.second);
    // The number of clusters.
    const auto n_clusters = this->cfg_->clustering.k;
    // Counters for cluster labels.
    std::vector<int> counters(n_clusters, 0);

    // Vote for cluster label of the tracklet.
    for (int i = 0; i < n_faces; ++i) {
      const auto label = *it_label_face;
      ++counters[label];
      ++it_label_face;
    }

    // Get the voting result.
    // The maximum count.
    int max_count = 0;
    // Cluster label with the maximum count.
    int max_cluster_id = 0;
    for (int i = 0; i < n_clusters; ++i) {
      if (max_count < counters[i]) {
        max_count = counters[i];
        max_cluster_id = i;
      }
    }
    (*cluster_ids)[idx_tracklet] = max_cluster_id;

    ++idx_tracklet;
  }
}

}   // namespace ugproj
