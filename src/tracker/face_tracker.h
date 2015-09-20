#ifndef UGPROJ_FACETRACKER_H_
#define UGPROJ_FACETRACKER_H_

#include "../file_io.h"
#include "../structure.h"
#include "detector.h"
#include <vector>

namespace ugproj {

class FaceTracker {
  public:
    FaceTracker();
    int set_input(TrackerFileInput* input);
    int set_writer(FileWriter* writer);
    int set_cfg(const Configuration* cfg);
    int track(std::vector<unsigned long>* tracked_positions,
              std::vector<FaceTracklet>* tracklets);

  private:
    struct VideoProperties {
      double fps;
      unsigned long frame_count;
      int frame_width;
      int frame_height;
    };
    static const char* kVideoKey;
    static const char* kVideoFilename;
    void get_properties(cv::VideoCapture* video, VideoProperties* props);
    int track_frame(
        const temp_idx_t curr_index,
        const cv::Mat& prev_frame,
        const cv::Mat& curr_frame,
        const FaceList* prev_faces,
        const std::vector<SparseOptflow>& prev_optflows,
        FaceDetector* detector,
        FaceList* curr_faces,
        std::vector<SparseOptflow>* curr_optflows,
        std::vector<FaceTracklet>* tracklets);
    int write_result(
        const temp_idx_t curr_index,
        const std::vector<unsigned long>& tracked_positions,
        const double sx,
        const double sy,
        const cv::Mat& curr_frame,
        const FaceList& curr_faces,
        const std::vector<SparseOptflow>& curr_optflows);
    int detect_faces(const temp_idx_t curr_index,
                     const cv::Mat& curr_frame,
                     FaceDetector* detector,
                     FaceList* curr_faces);
    int compute_optflow(const cv::Mat& prev_frame,
                        const cv::Mat& curr_frame,
                        const FaceList& prev_faces,
                        const std::vector<SparseOptflow>& prev_optflows,
                        std::vector<SparseOptflow>* curr_optflows);
    // Returns the list of points which will be used as input for Lucas-Kanade
    // algorithm.
    std::vector<cv::Point2f> prepare_points_for_lk(
        const cv::Mat& gray_frame,
        const FaceList& faces,
        const std::vector<SparseOptflow>& optflows);
    // Computes ROI used in GFTT for current frame with given set of faces
    // and optical flow from the previous frame. Returns true if
    // mask is set for any region in the frame. Otherwise, returns false.
    bool compute_roi_gftt(const cv::Size& frame_size,
                          const FaceList& faces,
                          const std::vector<SparseOptflow>& optflows,
                          cv::Mat* roi);
    // Runs Lucas-Kanade algorithm and populates the list of optical flows with
    // the result. Each optical flow is represented in SparseOptflow type.
    void run_lk(const cv::Mat& prev_gray,
                const cv::Mat& curr_gray,
                const std::vector<cv::Point2f> points,
                std::vector<SparseOptflow>* optflows);
    // Draws tracklet for a labeled face and writes to a file. The name of file
    // will be formatted with format string `face_%3d.png` with interpolation
    // of face ID. An instance of labeled face and the list of tracked frame
    // positions should be given.
    int write_tracklet(const FaceTracklet& tracklet,
                       const std::vector<unsigned long>& tracked_positions);
    TrackerFileInput* input_;
    FileWriter* writer_;
    const Configuration* cfg_;
};

}  // namespace ugproj

#endif  // UGPROJ_FACETRACKER_H_
