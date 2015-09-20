#include "clusterer.h"

#include "opencv2/opencv.hpp"

namespace ugproj {

FaceClusterer::FaceClusterer(const Configuration& cfg) {
  this->set_cfg(cfg);
}

void FaceClusterer::set_cfg(const Configuration& cfg) {
  this->cfg_ = &cfg;
}

void FaceClusterer::do_clustering(const cv::Mat& repr_faces_reduced,
                                  std::vector<int>* cluster_ids) {
  // Compactness of k-means clustering result.
  double compactness;
  // Clustering section of configuration.
  const auto& cfg_cl = this->cfg_->clustering;
  // List of cluster labels for faces in tracklets. Will be populated by
  // k-means clustering.
  std::vector<int> labels_face;
  // Run k-means clustering provided by OpenCV.
  compactness = cv::kmeans(repr_faces_reduced, cfg_cl.k, *cluster_ids,
                           cfg_cl.term_crit, cfg_cl.attempts,
                           cv::KMEANS_PP_CENTERS);
}

}   // namespace ugproj
