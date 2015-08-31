#define UGPROJ_SUPPRESS_CELIU

#include "face_tracker.h"
#include "file_io.h"
#include "structure.hpp"

#include <string>
#include <vector>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>


bool parse_args(int argc, const char** argv, ugproj::Configuration* cfg);

int main(int argc, const char** argv) {
  int ret = 0;

  ugproj::Configuration cfg;
  if (!parse_args(argc, argv, &cfg)) {
    return 1;
  }

  ugproj::FileInput input;
  ret = input.open(cfg);
  if (ret != 0) {
    return ret;
  }

  ugproj::FileWriter writer;
  ret = writer.init(cfg);
  if (ret != 0) {
    return ret;
  }

  // The list of tracked frame positions.
  std::vector<unsigned long> tracked_positions;
  ugproj::FaceTracker tracker;
  tracker.set_input(&input);
  tracker.set_writer(&writer);
  tracker.set_cfg(&cfg);
  ret = tracker.track(&tracked_positions);
  if (ret != 0) {
    return ret;
  }

  return 0;
}

bool parse_args(int argc, const char** argv, ugproj::Configuration* cfg) {
  namespace po = boost::program_options;

  try {
    po::options_description generic_options("Generic options");
    po::options_description config_options("Configuration");
    generic_options.add_options()
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
    config_options.add_options()
      ("scan.target_fps",
       po::value<double>(&cfg->scan.target_fps)->default_value(10.0),
       "fps at which video will be scanned.")
      ("scan.detection_scale",
       po::value<double>(&cfg->scan.detection_scale)->default_value(1.0),
       "scale at which image will be transformed during detection.")

      ("detection.cascade_classifier_filepath",
       po::value<std::string>()->required(),
       "path to cascade classifier file.")
      ("detection.scale",
       po::value<double>(&cfg->detection.scale)->default_value(1.1),
       "scale factor for cascade classifier used for detection.")

      ("gftt.max_n",
       po::value<int>(&cfg->gftt.max_n)->default_value(100),
       "maximum number of corners detected by GFTT algorithm.")
      ("gftt.quality_level",
       po::value<double>(&cfg->gftt.quality_level)->default_value(0.01),
       "quality level parameter used by GFTT algorithm.")
      ("gftt.min_distance",
       po::value<double>(&cfg->gftt.min_distance)->default_value(10),
       "minimum possible Euclidean distance between corners detected by GFTT "
       "algorithm.")

      ("subpixel.window_size",
       po::value<int>(&cfg->subpixel.window_size)->default_value(10),
       "half of the side length of the search window.")
      ("subpixel.zero_zone_size",
       po::value<int>(&cfg->subpixel.zero_zone_size)->default_value(-1),
       "half of the size of the dead region in the middle of the search zone. "
       "-1 indicates that there is no such size.")
      ("subpixel.term_crit_n",
       po::value<int>(&cfg->subpixel.term_crit.maxCount)->default_value(20),
       "the maximum number of iterations.")
      ("subpixel.term_crit_eps",
       po::value<double>(&cfg->subpixel.term_crit.epsilon)
           ->default_value(0.03),
       "the desired accuracy.")

      ("lucas_kanade.window_size",
       po::value<int>(&cfg->lucas_kanade.window_size)->default_value(31),
       "size of the search window at each pyramid level.")
      ("lucas_kanade.max_level",
       po::value<int>(&cfg->lucas_kanade.max_level)->default_value(3),
       "maximum number of pyramid levels.")
      ("lucas_kanade.term_crit_n",
       po::value<int>(&cfg->lucas_kanade.term_crit.maxCount)
           ->default_value(20),
       "the maximum number of iterations.")
      ("lucas_kanade.term_crit_eps",
       po::value<double>(&cfg->lucas_kanade.term_crit.epsilon)
           ->default_value(0.03),
       "the desired accuracy.")
      ("lucas_kanade.coeff_thres_len",
       po::value<double>(&cfg->lucas_kanade.coeff_thres_len)
           ->default_value(0.04),
       "coefficient for threshold on optical flow length.")

      ("association.prob_threshold",
       po::value<double>(&cfg->association.threshold)->default_value(0.5),
       "threshold for probability used during association.")
      ("association.method",
       po::value<std::string>()->required(),
       "association method. should be one of 'intersect', and 'optflow'.")
      ("association.coeff_thres_optflow",
       po::value<double>(&cfg->association.coeff_thres_optflow)
           ->default_value(0.1),
       "coefficient for threshold on the length of each optical flow.")
      ("association.coeff_thres_match",
       po::value<double>(&cfg->association.coeff_thres_match)
           ->default_value(0.2),
       "coefficient for threshold on the length of each computed match.")
      ("association.coeff_thres_box",
       po::value<double>(&cfg->association.coeff_thres_box)
           ->default_value(0.025),
       "coefficient for threshold on the size of each fit box.")
      ("association.coeff_thres_inlier",
       po::value<double>(&cfg->association.coeff_thres_inlier)
           ->default_value(0.09),
       "coefficient for threshold on the number of inliers in each fit box.")
      ("association.coeff_thres_aspect",
       po::value<double>(&cfg->association.coeff_thres_aspect)
           ->default_value(1.5),
       "coefficient for threshold on the aspect ratio of each fit box. "
       "should be greater than or equal to 1.0.")
    ;

    po::options_description visible_options("Allowed options");
    visible_options.add(generic_options);

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, visible_options), vm);
    if (vm.count("help")) {
      std::cout << visible_options << '\n';
      return false;
    }
    po::notify(vm);

    // Resolve path options from command line into path objects.
    cfg->video_filepath = vm["video-file"].as<std::string>();
    cfg->output_dirpath = vm["output-dir"].as<std::string>();
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
    cfg->detection.cascade_filepath =
        cascade_filepath.is_absolute() ?
        cascade_filepath :
        config_dirpath / cascade_filepath;

    ugproj::AssociationMethod& assoc_method = cfg->association.method;
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
  } catch (std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    return false;
  } catch (...) {
    std::cerr << "Unknown error!\n";
    return false;
  }

  return true;
}
