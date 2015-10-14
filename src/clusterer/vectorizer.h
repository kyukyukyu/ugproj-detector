#ifndef UGPROJ_CLUSTERER_VECTORIZER_H_
#define UGPROJ_CLUSTERER_VECTORIZER_H_

#include <vector>
#include <opencv2/opencv.hpp>
#include "flandmark_detector.h"

#include "../structure.h"

namespace ugproj {

// Abstract class that represents each of faces in vector.
class FaceVectorizer {
  public:
    // Represents each of faces in tracklets in vector.
    virtual cv::Mat vectorize(
        const std::vector<ugproj::FaceTracklet>& tracklets) = 0;
};

// Represents each of faces in vector based on the results of flandmark.
class FlandmarkVectorizer : public FaceVectorizer {
  public:
    // Constructor. Loads Flandmark model from a file at given path.
    FlandmarkVectorizer(const Configuration& cfg, FLANDMARK_Model* flm_model);
    // Represents each of faces in tracklets in vector.
    cv::Mat vectorize(const std::vector<ugproj::FaceTracklet>& tracklets);
  private:
    // Detects facial landmark from face using Flandmark. Returns pointers to
    // memory space for landmark positions. It is caller's responsibility to
    // free the memory space pointed by returned pointer.
    double* detect_landmarks(const Face& f);
    // Computes descriptor for face components whose region is determined based
    // on facial landmarks on face image.
    cv::Mat compute_desc(const double* landmarks, const Face& f);
    // Computes horizontally-wide ROI with given coordinate.
    cv::Rect get_roi_horiz(int x1, int y1, int x2, int y2,
                           double inv_aspect_ratio=3.0/5.0);
    // Computes vertically-wide ROI for nose.
    cv::Rect get_roi_nose(int x1, int y1, int x2, int y2, int nose_y);
    // Computes LBP histogram for an image. Radius is 1, and # of neighborhood
    // is 8. Therefore, the result vector should have 256 elmeents.
    void compute_lbp_hist(const cv::Mat& img, cv::Mat& hist);
    /* The side length of face images. */
    unsigned int face_size_;
    /* Boundary box for Flandmark. */
    int bbox_[4];
    /* Flandmark model. */
    FLANDMARK_Model* flm_model_;
};

// Represents each of faces in weight vectors of Eigenfaces.
class EigenfaceVectorizer : public FaceVectorizer {
  public:
    // Constructor. Loads configuration object.
    EigenfaceVectorizer(const Configuration& cfg);
    // Represents each of faces in tracklets in vector.
    cv::Mat vectorize(const std::vector<ugproj::FaceTracklet>& tracklets);
  private:
    /* The side length of face images. */
    unsigned int face_size_;
};

}   // namespace ugproj

#endif  // UGPROJ_CLUSTERER_VECTORIZER_H_
