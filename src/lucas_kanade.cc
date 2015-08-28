#define UGPROJ_SUPPRESS_CELIU

#include "face_tracker.h"
#include "file_io.h"
#include "structure.hpp"

#include <string>
#include <vector>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>


bool parse_args(int argc, const char** argv, ugproj::Arguments* args);

int main(int argc, const char** argv) {
  int ret = 0;

  ugproj::Arguments args;
  if (!parse_args(argc, argv, &args)) {
    return 1;
  }

  ugproj::FileInput input;
  ret = input.open(args);
  if (ret != 0) {
    return ret;
  }

  ugproj::FileWriter writer;
  ret = writer.init(args);
  if (ret != 0) {
    return ret;
  }

  // The list of tracked frame positions.
  std::vector<unsigned long> tracked_positions;
  ugproj::FaceTracker tracker;
  tracker.set_input(&input);
  tracker.set_writer(&writer);
  tracker.set_args(&args);
  ret = tracker.track(&tracked_positions);
  if (ret != 0) {
    return ret;
  }

  return 0;
}

bool parse_args(int argc, const char** argv, ugproj::Arguments* args) {
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
       po::value<double>(&args->target_fps)->default_value(10.0),
       "fps at which video will be scanned.")
      ("scan.detection_scale",
       po::value<double>(&args->detection_scale)->default_value(1.0),
       "scale at which image will be transformed during detection.")
      ("detection.cascade_classifier_filepath",
       po::value<std::string>()->required(),
       "path to cascade classifier file.")
      ("association.prob_threshold",
       po::value<double>(&args->assoc_threshold)->default_value(0.5),
       "threshold for probability used during association.")
      ("association.method",
       po::value<std::string>()->required(),
       "association method. should be one of 'intersect', and 'optflow'.")
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
    args->video_filepath = vm["video-file"].as<std::string>();
    args->output_dirpath = vm["output-dir"].as<std::string>();
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
    args->cascade_filepath =
        cascade_filepath.is_absolute() ?
        cascade_filepath :
        config_dirpath / cascade_filepath;

    ugproj::AssociationMethod& assoc_method = args->assoc_method;
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
