#ifndef UGPROJ_FACECLUSTERSVISUALIZER_H_
#define UGPROJ_FACECLUSTERSVISUALIZER_H_

#include "../file_io.h"

namespace ugproj {

// Visualizes the result of face clustering by listing all the detected faces
// in face tracklets for each face cluster in color images and writing the
// images to files. A FileWriter object is required for file output.
class FaceClustersVisualizer {
  public:
    // Constructor. writer is a pointer to FileWriter object for the
    // constructed object. This MUST be not NULL.
    FaceClustersVisualizer(FileWriter* writer);
    // Visualizes the result of face clustering and writes it to multiple image
    // files. tracked_positions is the list of tracked frame positions.
    // labeled_faces is the list of face tracklets. n_clusters is the number of
    // clusters. cluster_ids is the result of face clustering. The name of
    // files will be like 'cluster_1.png'. Returns non-zero value if writing
    // files was not successful.
    int visualize(const std::vector<unsigned long>& tracked_positions,
                  const std::vector<FaceTracklet>& labeled_faces,
                  const int n_clusters,
                  const std::vector<int>& cluster_ids);

  private:
    // Generates visualization for single cluster and writes it to file.
    int visualize_single(const std::vector<unsigned long>& tracked_positions,
                         const std::vector<FaceTracklet>& faces,
                         int cluster_id);
    // File writer for this object.
    FileWriter* writer_;
};

}   // namespace ugproj

#endif  // UGPROJ_FACECLUSTERSVISUALIZER_H_
