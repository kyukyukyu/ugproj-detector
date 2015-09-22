#include "visualizer.h"

namespace ugproj {

FaceClustersVisualizer::FaceClustersVisualizer(FileWriter* writer) {
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
    ret |= this->visualize_single(tracked_positions, clusters[i], i);
    if (ret) {
      break;
    }
  }
  return ret;
}

int FaceClustersVisualizer::visualize_single(
    const std::vector<unsigned long>& tracked_positions,
    const std::vector<FaceTracklet>& faces,
    int cluster_id) {
  // TODO: Get rid of duplicates on code.
  // TODO: Move this constant to class declaration.
  static const int kSize = 64;
  static const int kNCols = 16;
  static const cv::Scalar kColorText = CV_RGB(255, 255, 255);
  static const cv::Scalar kColorTextbox = CV_RGB(0, 0, 0);
  static const int kMarginTextbox = 4;
  // Count number of faces to be drawn.
  int n_faces = 0;
  for (const auto& tracklet : faces) {
    const auto iterators = tracklet.face_iterators();
    n_faces += iterators.second - iterators.first;
  }
  // Draw tracklet.
  const int n_rows = n_faces / kNCols + !!(n_faces % kNCols);
  cv::Mat visualized(n_rows * kSize, kNCols * kSize, CV_8UC3);
  visualized = CV_RGB(255, 255, 255);
  int i = 0;
  for (const auto& tracklet : faces) {
    const auto iterators = tracklet.face_iterators();
    for (auto it = iterators.first; it != iterators.second; ++it, ++i) {
      const Face& f = *it;
      const cv::Rect roi((i % kNCols) * kSize, (i / kNCols) * kSize,
                         kSize, kSize);
      cv::Mat visualized_roi(visualized, roi);
      // Draw the image of face.
      f.image.copyTo(visualized_roi);
      // Prepare the information of face.
      char c_str_frame_pos[16];
      std::sprintf(c_str_frame_pos, "%lu", tracked_positions[f.frameIndex]);
      std::string str_frame_pos(c_str_frame_pos);
      // Compute the position for the information text and box including it.
      int baseline;
      cv::Size text_size =
          cv::getTextSize(str_frame_pos, cv::FONT_HERSHEY_PLAIN, 1.0, 1,
                          &baseline);
      const cv::Rect box_rec(kMarginTextbox, kMarginTextbox,
                             2 * kMarginTextbox + text_size.width,
                             2 * kMarginTextbox + text_size.height);
      const cv::Point text_org(2 * kMarginTextbox,
                               2 * kMarginTextbox + text_size.height);
      // Draw the information text and box including it.
      // The box should be drawn first.
      cv::rectangle(visualized_roi, box_rec, kColorTextbox, CV_FILLED);
      cv::putText(visualized_roi, str_frame_pos, text_org,
                  cv::FONT_HERSHEY_PLAIN, 1.0, kColorText);
    }
  }
  // Prepare the filename.
  char filename[256];
  std::sprintf(filename, "cluster_%d.png", cluster_id);
  // Write to file.
  return this->writer_->write_image(visualized, filename);
}

}   // namespace ugproj
