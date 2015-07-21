#ifndef UGPROJ_FACETRACKER_H_
#define UGPROJ_FACETRACKER_H_
#define UGPROJ_SUPPRESS_CELIU

#include "detector/detector.hpp"
#include "file_io.h"
#include "structure.hpp"
#include <vector>

namespace ugproj {

class FaceTracker {
  public:
    FaceTracker();
    int set_input(FileInput* input);
    int set_writer(FileWriter* writer);
    int set_args(const Arguments* args);
    int track(std::vector<unsigned long>* tracked_positions);

  private:
    struct VideoProperties {
      double fps;
      unsigned long frame_count;
      int frame_width;
      int frame_height;
    };
    static const int kGfttMaxCorners = 100;
    void get_properties(cv::VideoCapture* video, VideoProperties* props);
    int track_frame(
        const temp_idx_t curr_index,
        const cv::Mat& prev_frame,
        const cv::Mat& curr_frame,
        const FaceCandidateList* prev_candidates,
        const std::vector<SparseOptflow>& prev_optflows,
        FaceDetector* detector,
      FaceCandidateList* curr_candidates,
      std::vector<SparseOptflow>* curr_optflows);
    int write_result(
        const temp_idx_t curr_index,
        const std::vector<unsigned long>& tracked_positions,
        const cv::Mat& prev_frame,
        const cv::Mat& curr_frame,
        const FaceCandidateList& curr_candidates,
        const std::vector<SparseOptflow>& curr_optflows);
    int detect_faces(const temp_idx_t curr_index,
                     const cv::Mat& curr_frame,
                     FaceDetector* detector,
                     FaceCandidateList* curr_candidates);
    int compute_optflow(const cv::Mat& prev_frame,
                        const cv::Mat& curr_frame,
                        const FaceCandidateList& prev_candidates,
                        const std::vector<SparseOptflow>& prev_optflows,
                        std::vector<SparseOptflow>* curr_optflows);
    // Draws tracklet for a labeled face and writes to a file. The name of file
    // will be formatted with format string `face_%3d.png` with interpolation
    // of face ID. An instance of labeled face and the list of tracked frame
    // positions should be given.
    int write_tracklet(const Face& f,
                       const std::vector<unsigned long>& tracked_positions);
    FileInput* input_;
    FileWriter* writer_;
    const Arguments* args_;
    // List of labeled faces.
    std::vector<Face> labeled_faces_;
};

}  // namespace ugproj

#endif  // UGPROJ_FACETRACKER_H_
