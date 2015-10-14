#include "visualizer.h"

#include "../visualizer.h"

namespace ugproj {

FaceClustersVisualizer::FaceClustersVisualizer(
    const Configuration& cfg,
    FileWriter* writer) : cfg_output_(cfg.output) {
  this->writer_ = writer;
}

int FaceClustersVisualizer::visualize(
    const std::vector<unsigned long>& tracked_positions,
    const std::vector<FaceTracklet>& labeled_faces,
    const int n_clusters,
    const std::vector<int>& cluster_ids) {
  int ret = 0;
  std::vector< std::vector<FaceTracklet> > clusters(n_clusters);
  for (unsigned long i = 0, n = labeled_faces.size(); i < n; ++i) {
    tracklet_id_t cluster_id = cluster_ids[i];
    const FaceTracklet& tracklet = labeled_faces[i];
    std::vector<FaceTracklet>& cluster = clusters[cluster_id];
    cluster.push_back(tracklet);
  }
  for (int i = 0; i < n_clusters; ++i) {
    if (clusters[i].empty()) {
      continue;
    }
    ret |= this->visualize_single(tracked_positions, clusters[i], i);
    if (ret) {
      break;
    }
  }
  return ret;
}

int FaceClustersVisualizer::visualize_single(
    const std::vector<unsigned long>& tracked_positions,
    const std::vector<FaceTracklet>& tracklets,
    int cluster_id) {
  std::vector< FaceRange<FaceList::const_iterator> > face_ranges;
  for (const auto& tracklet : tracklets) {
    face_ranges.push_back(tracklet.face_iterators());
  }
  const auto& cfg_output = this->cfg_output_;
  const auto& visualized =
      visualize_faces(face_ranges.cbegin(), face_ranges.cend(),
                      cfg_output.face_size, cfg_output.n_cols_tracklet,
                      tracked_positions);
  // Prepare the filename.
  char filename[256];
  std::sprintf(filename, "cluster_%d.png", cluster_id);
  // Write to file.
  return this->writer_->write_image(visualized, filename);
}

}   // namespace ugproj
