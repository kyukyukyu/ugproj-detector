#ifndef UGPROJ_ASSOCIATOR_HEADER
#define UGPROJ_ASSOCIATOR_HEADER

#define UGPROJ_ASSOCIATOR_SIFT_TRIAL_COUNT 10
#define UGPROJ_ASSOCIATOR_SIFT_SCALE_THRESHOLD 1.25

#include "../structure.hpp"

#include <boost/optional.hpp>
#include <opencv2/opencv.hpp>
#include <set>
#include <utility>
#include <vector>

namespace ugproj {

class FaceAssociator {
  public:
    typedef std::vector<FaceCandidate> fc_v;

  protected:
    std::vector<Face>& faces;
    const fc_v& prevCandidates;
    fc_v& nextCandidates;
    double **prob;
    double threshold;

    void matchCandidates();

  public:
    FaceAssociator(
        std::vector<Face>& faces,
        const fc_v& prevCandidates,
        fc_v& nextCandidates,
        double threshold):
      faces(faces),
      prevCandidates(prevCandidates),
      nextCandidates(nextCandidates),
      threshold(threshold) {
        // probability array allocation
        fc_v::size_type rows, cols;
        rows = prevCandidates.size();
        cols = nextCandidates.size();

        prob = new double *[rows];
        while (rows--) {
          prob[rows] = new double[cols];
        }
      }
    virtual ~FaceAssociator() {
      fc_v::size_type row = prevCandidates.size();
      while (row--) {
        delete[] prob[row];
      }
      delete[] prob;
    }
    virtual void associate();
    virtual void calculateProb() = 0;
};

// Associates face candidates using KLT algorithm and RANSAC-based box fitting.
//
// KEEP IN MIND that calling `associate()` to instances of this class may put
// new face candidates into `nextCandidates` to approximate 'undetected' faces
// in image. For more about this issue, check GitHub issue #14.
class KltFaceAssociator : public FaceAssociator {
  public:
    // Represent single match between two feature points.
    typedef std::pair<cv::Point2d, cv::Point2d> Match;
    // Compares two Match objects. When ordered using this class, the object
    // whose outgoing point has lower x value will come first. If the x values
    // of two objects are same, the object whose outgoing point has lower y
    // value will come first. If two objects has the same outgoing point, the
    // same comparison will be made between the incoming points of them.
    struct MatchCompare {
      bool operator()(const Match& lhs, const Match& rhs) const {
        const cv::Point2d& lhs_prev = lhs.first;
        const cv::Point2d& rhs_prev = rhs.first;
        const cv::Point2d& lhs_next = lhs.second;
        const cv::Point2d& rhs_next = rhs.second;
        return this->cmp_point(lhs_prev, rhs_prev) ||
               (!this->cmp_point(rhs_prev, lhs_prev) &&
                this->cmp_point(lhs_next, rhs_next));
      }
      bool cmp_point(const cv::Point2d& lhs, const cv::Point2d& rhs) const {
        return lhs.x < rhs.x || (!(rhs.x < lhs.x) && lhs.y < rhs.y);
      }
    };
    typedef std::set<Match, MatchCompare> MatchSet;
    // Result of box-fitting with respect to single face candidate.
    struct Fit {
      // Fit box.
      cv::Rect box;
      double sx;
      double sy;
      int a;
      int b;

      cv::Point origin;
      // Set of outgoing matches from the face candidate.
      MatchSet matches;
      // The number of inliers inside the fit box.
      unsigned int num_inliers;
      // Returns if it is valid result.
      bool valid() const {
        return this->num_inliers > 0;
      }
    };
    KltFaceAssociator(std::vector<Face>& faces,
                      const FaceCandidateList& prev_candidates,
                      FaceCandidateList& next_candidates,
                      const temp_idx_t next_index,
                      const cv::Mat& next_frame,
                      const std::vector<SparseOptflow>& optflows,
                      double threshold);
    void associate();
    void calculateProb();

  private:
    // Enums for selecting which point of a match is checked. Used in
    // find_matches().
    enum MatchPointSelection {
      kOutgoing = 0,
      kIncoming
    };
    // Computes best fits for each face candidate in former (i.e. previous)
    // frame using RANSAC-based algorithm, and save it into `best_fits_`.
    void compute_best_fits();
    // Returns the set of matches whose outgoing/incoming points are inside a
    // rect. The rect and the selection of point for matches should be given.
    MatchSet find_matches(const cv::Rect& rect,
                          const MatchPointSelection point_selection) const;
    MatchSet find_matches_in_rect(const cv::Rect& rect,
                          const MatchSet& matches) const;
    int compute_inlier(const MatchSet& matches, const Fit& fit_box) const;
    // Returns the list of fit boxes computed based on RANSAC-based algorithm.
    // The set of matches, and the base box for box-fitting should be given.
    std::vector<Fit> compute_fit_boxes(const MatchSet& matches,
                                            const cv::Rect& base_rect) const;
    // Computes single fit box for given base rect and two matches. For
    // parameters, a const reference to base rect, two const reference to two
    // matches, and a pointer to a rect object where the result will be stored
    // should be given. If the fit box is computed correctly, this will return
    // true. Otherwise, this will return false.
    bool compute_fit_box(const cv::Rect& base_rect, const Match& match1,
                         const Match& match2, Fit* fit_box) const;
    // Returns the list of index pairs for picking two items in an item set.
    // The size of item set should be provided. For example, if the size of
    // item set is 3, {{0, 1}, {0, 2}, {1, 2}} is returned. When shuffle is
    // desired, the optional parameter should be set to true.
    static std::vector< std::pair<unsigned int, unsigned int> >
    list_index_pairs(unsigned int size, bool shuffle=false);
    // The index of next (latter) frame.
    const temp_idx_t next_index_;
    // The next (latter) frame.
    const cv::Mat& next_frame_;
    // The size of frames in the video.
    const cv::Size frame_size_;
    // The list of optical flows required for association.
    const std::vector<SparseOptflow>& optflows_;
    // The list of computed best fits. This is populated by calling
    // compute_best_fits().
    std::vector<Fit> best_fits_;
};

} // ugproj

#endif  // UGPROJ_ASSOCIATOR_HEADER
