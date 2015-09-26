#ifndef UGPROJ_VISUALIZER_H_
#define UGPROJ_VISUALIZER_H_

#include <iterator>
#include <utility>
#include "structure.h"

namespace ugproj {

// Type alias for ranges on ugproj::Face objects. The first iterator points to
// the start of the face set, and the second iterator points to the end.
template <class FaceIt> using FaceRange = std::pair<FaceIt, FaceIt>;

// Visualize a set of sets of ugproj::Face objects in tile layout.
//
// The set of sets should be given in the form of two iterators: start and end.
// start and end are iterators which point to the start and end of the set of
// sets each. Each set of ugproj::Face objects are also given in the form of
// range on ugproj::Face objects.
//
// The side length of face images face_size and the number of columns in the
// image n_cols also should be given. The list of frame positions
// tracked_positions is optional.
template <class FaceRangeIt>
cv::Mat visualize_faces(
    const FaceRangeIt& start,
    const FaceRangeIt& end,
    const unsigned int face_size,
    const unsigned int n_cols,
    const std::vector<unsigned long>& tracked_positions={}) {
  // Constants for visualizing.
  static const cv::Scalar kColorText = CV_RGB(255, 255, 255);
  static const cv::Scalar kColorTextbox = CV_RGB(0, 0, 0);
  static const int kMarginTextbox = 4;

  // Count number of faces to be drawn.
  int n_faces = 0;
  for (auto it = start; it != end; ++it) {
    const auto& range = *it;
    n_faces += std::distance(range.first, range.second);
  }

  // Prepare matrix for visualization.
  const int n_rows = n_faces / n_cols + !!(n_faces % n_cols);
  cv::Mat visualized(n_rows * face_size, n_cols * face_size, CV_8UC3);
  visualized = CV_RGB(255, 255, 255);

  // Draw face images.
  const bool no_frame_position = tracked_positions.empty();
  int i = 0;
  for (auto it = start; it != end; ++it) {
    const auto& range = *it;
    for (auto it = range.first; it != range.second; ++it, ++i) {
      const Face& f = *it;
      const cv::Rect roi((i % n_cols) * face_size, (i / n_cols) * face_size,
                         face_size, face_size);
      cv::Mat visualized_roi(visualized, roi);
      // Draw the image of face.
      f.resized_image(face_size).copyTo(visualized_roi);

      // Skip drawing frame position if not given.
      if (no_frame_position) {
        continue;
      }

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
  return visualized;
}

}   // namespace ugproj

#endif  // UGPROJ_VISUALIZER_H_
