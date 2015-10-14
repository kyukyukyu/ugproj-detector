#include "vectorizer.h"

#include "flandmark_detector.h"

namespace ugproj {

FlandmarkVectorizer::FlandmarkVectorizer(const Configuration& cfg,
                                       FLANDMARK_Model* flm_model)
: face_size_(cfg.output.face_size), flm_model_(flm_model) {
  this->bbox_[0] = this->bbox_[1] = 1;
  this->bbox_[2] = this->bbox_[3] = this->face_size_ - 1;
  this->flm_model_->data.options.bw_margin[0] = 0;
  this->flm_model_->data.options.bw_margin[1] = 0;
}

cv::Mat FlandmarkVectorizer::vectorize(
    const std::vector<ugproj::FaceTracklet>& tracklets) {
  // Matrix composed of vectors that represent faces in tracklets.
  cv::Mat faces_vectorized;
  // Loop variable.
  int i;
  // The number of faces in tracklets.
  int n_faces = 0;
  /* The number of landmarks. */
  const int n_landmarks = this->flm_model_->data.options.M;
  // The number of columns of faces_vectorized.
  const int n_cols = 4 * 256;
  // Count the number of faces in tracklets.
  for (const auto& tracklet : tracklets) {
    const auto& face_iterators = tracklet.face_iterators();
    n_faces += face_iterators.second - face_iterators.first;
  }
  // Set the size of faces_vectorized.
  faces_vectorized.create(n_faces, n_cols, CV_32F);
  // Fill faces_vectorized.
  i = 0;
  for (const auto& tracklet : tracklets) {
    const auto& face_iterators = tracklet.face_iterators();
    for (auto it = face_iterators.first; it != face_iterators.second; ++it) {
      // Face whose landmarks will be detected.
      const auto& face = *it;
      // Pointer to memory space for positions of facial landmarks.
      double* landmarks = this->detect_landmarks(face);
      // Compute BRISK descriptors at facial landmarks.
      cv::Mat desc = this->compute_desc(landmarks, face);
      desc.copyTo(faces_vectorized.row(i));
      // Free the memory space.
      std::free(landmarks);
      ++i;
    }
  }
  return faces_vectorized;
}

double* FlandmarkVectorizer::detect_landmarks(const Face& f) {
  /* Face image converted to grayscale image. */
  cv::Mat face_img_gray;
  /* face_img_gray in IplImage */
  IplImage face_iplimg_gray;
  /* Pointer to memory space for detected facial landmarks. */
  double* landmarks =
      (double*) std::malloc(2 * this->flm_model_->data.options.M *
                            sizeof(double));
  cv::cvtColor(f.image, face_img_gray, CV_BGR2GRAY);
  face_iplimg_gray = face_img_gray;
  flandmark_detect(&face_iplimg_gray, this->bbox_, this->flm_model_,
                   landmarks);
  // Face image whose landmarks are marked.
  cv::Mat face_marked;
  f.image.copyTo(face_marked);
  // Landmark points.
  std::vector<cv::Point2d> landmark_points;
  landmark_points.reserve(8);
  for (int i = 0; i < 16; i += 2) {
    landmark_points.push_back(cv::Point2d(landmarks[i], landmarks[i + 1]));
  }
  // Draw center point.
  cv::circle(face_marked, landmark_points[0], 3, CV_RGB(0, 0, 255), CV_FILLED);
  // Draw other points.
  for (int i = 1; i < 8; ++i) {
    cv::circle(face_marked, landmark_points[i], 3, CV_RGB(255, 0, 0),
               CV_FILLED);
  }
#ifdef UGPROJ_CLUSTERER_VERBOSE
  cv::imshow("Landmark Detection", face_marked);
#endif
  return landmarks;
}

cv::Mat FlandmarkVectorizer::compute_desc(const double* landmarks,
                                         const Face& f) {
  cv::Mat desc;
  desc.create(1, 4 * 256, CV_32F);
  cv::Mat desc_left_eye = desc.colRange(0, 256);
  cv::Mat desc_right_eye = desc.colRange(256, 512);
  cv::Mat desc_nose = desc.colRange(512, 768);
  cv::Mat desc_mouth = desc.colRange(768, 1024);

  const auto& face_img = f.image;
  cv::Mat left_eye;
  cv::Mat right_eye;
  cv::Mat nose;
  cv::Mat mouth;

  cv::Rect roi_left_eye =
      this->get_roi_horiz(landmarks[10], landmarks[11],
                          landmarks[2], landmarks[3], 2.0/3.0);
  cv::Rect roi_right_eye =
      this->get_roi_horiz(landmarks[4], landmarks[5],
                          landmarks[12], landmarks[13], 2.0/3.0);
  cv::Rect roi_nose =
      this->get_roi_nose(landmarks[2], landmarks[3],
                         landmarks[4], landmarks[5], landmarks[15]);
  cv::Rect roi_mouth =
      this->get_roi_horiz(landmarks[6], landmarks[7],
                          landmarks[8], landmarks[9], 2.0/3.0);

  left_eye = cv::Mat(face_img, roi_left_eye);
  right_eye = cv::Mat(face_img, roi_right_eye);
  nose = cv::Mat(face_img, roi_nose);
  mouth = cv::Mat(face_img, roi_mouth);
#ifdef UGPROJ_CLUSTERER_VERBOSE
  cv::imshow("Left Eye", left_eye);
  cv::imshow("Right Eye", right_eye);
  cv::imshow("Nose", nose);
  cv::imshow("Mouth", mouth);
  cv::waitKey(0);
#endif

  this->compute_lbp_hist(left_eye, desc_left_eye);
  this->compute_lbp_hist(right_eye, desc_right_eye);
  this->compute_lbp_hist(nose, desc_nose);
  this->compute_lbp_hist(mouth, desc_mouth);

  return desc;
}

cv::Rect FlandmarkVectorizer::get_roi_horiz(int x1, int y1, int x2, int y2,
                                           double inv_aspect_ratio) {
  int width = x2 - x1;
  int height = inv_aspect_ratio * width;
  int center_y = (y1 + y2) / 2;
  return cv::Rect(x1, center_y - height / 2, width, height);
}

cv::Rect FlandmarkVectorizer::get_roi_nose(int x1, int y1, int x2, int y2,
                                          int nose_y) {
  int top_y = y1 > y2 ? y1 : y2;
  int width = x2 - x1;
  int height = nose_y - top_y;
  return cv::Rect(x1, top_y, width, height);
}

void FlandmarkVectorizer::compute_lbp_hist(const cv::Mat& img, cv::Mat& hist) {
  cv::Mat img_gray;
  cv::cvtColor(img, img_gray, CV_BGR2GRAY);
  hist = 0.0;
  for (int i = 1; i < img.rows - 1; ++i) {
    for (int j = 1; j < img.cols - 1; ++j) {
      unsigned char center = img.at<unsigned char>(i, j);
      unsigned char code = 0;
      code |= (img.at<unsigned char>(i - 1, j - 1) >= center) << 7;
      code |= (img.at<unsigned char>(i - 1, j) >= center) << 6;
      code |= (img.at<unsigned char>(i - 1, j + 1) >= center) << 5;
      code |= (img.at<unsigned char>(i, j + 1) >= center) << 4;
      code |= (img.at<unsigned char>(i + 1, j + 1) >= center) << 3;
      code |= (img.at<unsigned char>(i + 1, j) >= center) << 2;
      code |= (img.at<unsigned char>(i + 1, j - 1) >= center) << 1;
      code |= (img.at<unsigned char>(i, j - 1) >= center) << 0;
      ++hist.at<float>(0, code);
    }
  }
  hist /= img.rows * img.cols;
}

}   // namespace ugproj
