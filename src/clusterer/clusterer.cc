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
  std::printf("before kmeans\n");
  compactness = cv::kmeans(faces_reduced, cfg_cl.k, labels_face,
                           cfg_cl.term_crit, cfg_cl.attempts,
                           cv::KMEANS_PP_CENTERS);

  std::printf("labels face size \n");

  // Put cluster labels on face tracklets.
  this->vote_for_labels(labels_face, tracklets, cluster_ids);
}

void FaceClusterer::vote_for_labels(const std::vector<int>& labels_face,
                                    const std::vector<FaceTracklet>& tracklets,
                                    std::vector<int>* cluster_ids) {
  auto it_label_face = labels_face.cbegin();
  std::printf("label face size %d\n",(int)labels_face.size());
  std::printf("tracklet size %d\n",(int)tracklets.size());
  // Clear the vector object pointed by cluster_ids and reserve space for new
  // elements.
  cluster_ids->clear();
  cluster_ids->reserve(tracklets.size());

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

  std::printf("before vote\n");
    // Vote for cluster label of the tracklet.
    for (int i = 0; i < n_faces; ++i) {
  std::printf("before label\n");
      const auto label = *it_label_face;
  std::printf("after label\n");

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
    cluster_ids->push_back(max_cluster_id);
  }
}

}   // namespace ugproj
