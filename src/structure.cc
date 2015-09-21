#include "structure.h"
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

namespace ugproj {

cv::Mat Face::resized_image(int size) const {
  cv::Mat resized;
  const cv::Size& orig_size = this->rect.size();
  double f;
  if (orig_size.width >= orig_size.height) {
    f = (double) size / (double) orig_size.height;
  } else {
    f = (double) size / (double) orig_size.width;
  }
  cv::resize(this->image, resized,
             cv::Size(0, 0),    /* to make use of fx and fy */
             f, f);
  if (resized.rows != resized.cols) {
    // Resized image is not square-sized.
    cv::Rect roi;
    roi.x = (resized.cols - size) / 2;
    roi.y = (resized.rows - size) / 2;
    roi.width = roi.height = size;
    resized = cv::Mat(resized, roi);
  }
  return resized;
}

cv::Mat Face::get_image() const{
  return this->image;
}

FaceTracklet::FaceTracklet(id_type id, const Face& f) : id(id) {
  add_face(f);
}

void FaceTracklet::add_face(const Face& f) {
  faces.push_back(f);
}

int Configuration::load(int argc, const char** argv) {
  namespace po = boost::program_options;

  try {
    po::options_description tracker_options("Tracker options");
    po::options_description clusterer_options("Clusterer options");
    po::options_description config_options("Configuration");
    tracker_options.add_options()
      ("help", "produce help message.")
      ("config-file,c",
       po::value<std::string>()
         ->value_name("config_filepath")
         ->required(),
       "path to config file.")
      ("video-file,v",
       po::value<std::string>()
         ->value_name("video_filepath")
         ->required(),
       "path to input video file.")
      ("output-dir,o",
       po::value<std::string>()
         ->value_name("output_dir")
         ->default_value("output"),
       "path to output directory.")
    ;
    clusterer_options.add_options()
      ("help", "produce help message.")
      ("config-file,c",
       po::value<std::string>()
         ->value_name("config_filepath")
         ->required(),
       "path to config file.")
      ("metadata-file,m",
       po::value<std::string>()
         ->value_name("metadata_filepath")
         ->required(),
       "path to metadata file.")
      ("mapping-file,p",
       po::value<std::string>()
         ->value_name("mapping_filepath")
         ->required(),
       "path to mapping file.")
      ("input-dir,i",
       po::value<std::string>()
         ->value_name("input_dir")
         ->required(),
       "path to input directory.")
      ("output-dir,o",
       po::value<std::string>()
         ->value_name("output_dir")
         ->default_value("output"),
       "path to output directory.")
    ;
    config_options.add_options()
      ("scan.frame_size",
       po::value<cv::Size>(&this->scan.frame_size)
           ->default_value(cv::Size(-1, -1)),
       "frame size at which video will be scanned. "
       "default value is the original frame size.")
      ("scan.target_fps",
       po::value<double>(&this->scan.target_fps)->default_value(10.0),
       "fps at which video will be scanned.")

      ("detection.cascade_classifier_filepath",
       po::value<std::string>()->required(),
       "path to cascade classifier file.")
      ("detection.skin_lower",
       po::value<cv::Scalar>(&this->detection.skin_lower)->
           default_value(cv::Scalar(0, 133, 77)),
       "lower bound for skin color range in YCrCb.")
      ("detection.skin_upper",
       po::value<cv::Scalar>(&this->detection.skin_upper)->
           default_value(cv::Scalar(255, 173, 127)),
       "upper bound for skin color range in YCrCb.")
      ("detection.scale",
       po::value<double>(&this->detection.scale)->default_value(1.1),
       "scale factor for cascade classifier used for detection.")

      ("gftt.max_n",
       po::value<int>(&this->gftt.max_n)->default_value(100),
       "maximum number of corners detected by GFTT algorithm.")
      ("gftt.quality_level",
       po::value<double>(&this->gftt.quality_level)->default_value(0.01),
       "quality level parameter used by GFTT algorithm.")
      ("gftt.min_distance",
       po::value<double>(&this->gftt.min_distance)->default_value(10),
       "minimum possible Euclidean distance between corners detected by GFTT "
       "algorithm.")

      ("subpixel.window_size",
       po::value<int>(&this->subpixel.window_size)->default_value(10),
       "half of the side length of the search window.")
      ("subpixel.zero_zone_size",
       po::value<int>(&this->subpixel.zero_zone_size)->default_value(-1),
       "half of the size of the dead region in the middle of the search zone. "
       "-1 indicates that there is no such size.")
      ("subpixel.term_crit_n",
       po::value<int>(&this->subpixel.term_crit.maxCount)->default_value(20),
       "the maximum number of iterations.")
      ("subpixel.term_crit_eps",
       po::value<double>(&this->subpixel.term_crit.epsilon)
           ->default_value(0.03),
       "the desired accuracy.")

      ("lucas_kanade.window_size",
       po::value<int>(&this->lucas_kanade.window_size)->default_value(31),
       "size of the search window at each pyramid level.")
      ("lucas_kanade.max_level",
       po::value<int>(&this->lucas_kanade.max_level)->default_value(3),
       "maximum number of pyramid levels.")
      ("lucas_kanade.term_crit_n",
       po::value<int>(&this->lucas_kanade.term_crit.maxCount)
           ->default_value(20),
       "the maximum number of iterations.")
      ("lucas_kanade.term_crit_eps",
       po::value<double>(&this->lucas_kanade.term_crit.epsilon)
           ->default_value(0.03),
       "the desired accuracy.")
      ("lucas_kanade.coeff_thres_len",
       po::value<double>(&this->lucas_kanade.coeff_thres_len)
           ->default_value(0.04),
       "coefficient for threshold on optical flow length.")

      ("association.prob_threshold",
       po::value<double>(&this->association.threshold)->default_value(0.5),
       "threshold for probability used during association.")
      ("association.method",
       po::value<std::string>()->required(),
       "association method. should be one of 'intersect', and 'optflow'.")
      ("association.coeff_thres_optflow",
       po::value<double>(&this->association.coeff_thres_optflow)
           ->default_value(0.1),
       "coefficient for threshold on the length of each optical flow.")
      ("association.coeff_thres_match",
       po::value<double>(&this->association.coeff_thres_match)
           ->default_value(0.2),
       "coefficient for threshold on the length of each computed match.")
      ("association.coeff_thres_box",
       po::value<double>(&this->association.coeff_thres_box)
           ->default_value(0.025),
       "coefficient for threshold on the size of each fit box.")
      ("association.coeff_thres_inlier",
       po::value<double>(&this->association.coeff_thres_inlier)
           ->default_value(0.09),
       "coefficient for threshold on the number of inliers in each fit box.")
      ("association.coeff_thres_aspect",
       po::value<double>(&this->association.coeff_thres_aspect)
           ->default_value(1.5),
       "coefficient for threshold on the aspect ratio of each fit box. "
       "should be greater than or equal to 1.0.")

      ("clustering.k",
       po::value<int>(&this->clustering.k)->required(),
       "the number of clusters to be found.")
      ("clustering.term_crit_n",
       po::value<int>(&this->clustering.term_crit.maxCount),
       "the maximum number of iterations in k-means.")
      ("clusterer.term_crit_eps",
       po::value<double>(&this->clustering.term_crit.epsilon),
       "the desired accuracy for k-means.")
      ("clusterer.attempts",
       po::value<int>(&this->clustering.attempts)->default_value(8),
       "the number of attempts with different initial labellings.")
    ;

    po::options_description visible_options("Allowed options");
    if(!std::string("./tracker").compare(argv[0])){
      visible_options.add(tracker_options);
    }else if(!std::string("./clusterer").compare(argv[0])){
      visible_options.add(clusterer_options);
    }
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, visible_options), vm);
    if (vm.count("help")) {
      std::cout << visible_options << '\n';
      return -1;
    }
    po::notify(vm);

    // Resolve path options from command line into path objects.
    if(!std::string("./tracker").compare(argv[0])){
      this->video_filepath = vm["video-file"].as<std::string>();
    }else if(!std::string("./clusterer").compare(argv[0])){
      this->metadata_filepath = vm["metadata-file"].as<std::string>();
      this->mapping_filepath = vm["mapping-file"].as<std::string>();
      this->input_dirpath = vm["input-dir"].as<std::string>();
    }
    this->output_dirpath = vm["output-dir"].as<std::string>();
    boost::filesystem::path config_filepath =
        vm["config-file"].as<std::string>();
    auto config_dirpath = config_filepath.parent_path();

    // Load configuration from config file.
    po::store(
        po::parse_config_file<char>(config_filepath.c_str(), config_options),
        vm);
    po::notify(vm);

    // Resolve path options from config file into path objects.
    boost::filesystem::path cascade_filepath =
        vm["detection.cascade_classifier_filepath"].as<std::string>();
    this->detection.cascade_filepath =
        cascade_filepath.is_absolute() ?
        cascade_filepath :
        config_dirpath / cascade_filepath;

    ugproj::AssociationMethod& assoc_method = this->association.method;
    const auto& assoc_method_s = vm["association.method"].as<std::string>();
    if (assoc_method_s == "intersect") {
      assoc_method = ugproj::ASSOC_INTERSECT;
    } else if (assoc_method_s == "optflow") {
      assoc_method = ugproj::ASSOC_OPTFLOW;
    } else if (assoc_method_s == "sift") {
      assoc_method = ugproj::ASSOC_SIFT;
    } else {
      throw "invalid association method";
    }

    // Resolve termination criteria for k-means.
    {
      bool empty_n = vm["clustering.term_crit_n"].empty();
      bool empty_eps = vm["clustering.term_crit_eps"].empty();
      auto& type = this->clustering.term_crit.type;
      if (!empty_n) {
        type = cv::TermCriteria::MAX_ITER;
      }
      if (!empty_eps) {
        type = cv::TermCriteria::EPS;
      }
      if (empty_n && empty_eps) {
        throw "At least one of max iteration number and desired accuracy "
              "should be set for termination criteria for k-means.";
      }
      if (!empty_n && !empty_eps) {
        type = cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS;
      }
    }
  } catch (std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    return -1;
  } catch (const char*& e) {
    std::cerr << "Error: " << e << '\n';
    return -1;
  } catch (...) {
    std::cerr << "Unknown error!\n";
    return -1;
  }

  return 0;
}

}   // namespace ugproj

namespace boost {
namespace program_options {

// Validates if input for program options is in the form of '255,255,255' and
// converts it to cv::Scalar object.
template <> void validate(
    boost::any& v,
    const std::vector<std::string>& values,
    cv::Scalar*,
    long) {
  namespace po = boost::program_options;

  // No previous assignment allowed.
  po::validators::check_first_occurrence(v);
  // No more than one string allowed.
  const std::string& s = po::validators::get_single_string(values);

  // Scan three numbers from the string.
  int n1, n2, n3;
  if (std::sscanf(s.c_str(), "%3u , %3u , %3u", &n1, &n2, &n3) == 3) {
    v = boost::any(cv::Scalar(n1, n2, n3));
  } else {
    throw po::validation_error::invalid_option_value;
  }
}

// Validates if input for program options is in the form of '1280x720' and
// converts it to cv::Size object.
template <> void validate(
    boost::any& v,
    const std::vector<std::string>& values,
    cv::Size*,
    long) {
  namespace po = boost::program_options;

  // No previous assignment allowed.
  po::validators::check_first_occurrence(v);
  // No more than one string allowed.
  const std::string& s = po::validators::get_single_string(values);

  // Scan two numbers from the string and create cv::Size object.
  int n1, n2;
  if (std::sscanf(s.c_str(), "%u x %u", &n1, &n2) == 2) {
    v = boost::any(cv::Size(n1, n2));
  } else {
    throw po::validation_error::invalid_option_value;
  }
}

}   // namespace program_options
}   // namespace boost
