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
                                  std::vector<tracklet_id_t>* cluster_ids) {
  double compactness;
  const auto& cfg_cl = this->cfg_->clustering;
  compactness = cv::kmeans(repr_faces_reduced, cfg_cl.k, *cluster_ids,
                           cfg_cl.term_crit, cfg_cl.attempts,
                           cv::KMEANS_PP_CENTERS);
}

}   // namespace ugproj
