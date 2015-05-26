#define UGPROJ_SUPPRESS_CELIU

#include "face_tracker.h"
#include "file_io.h"
#include "structure.hpp"

#include <string>
#include <vector>
#include <boost/program_options.hpp>


bool parse_args(int argc, const char** argv, ugproj::Arguments* args);

int main(int argc, const char** argv) {
  int ret = 0;

  ugproj::Arguments args;
  ret = parse_args(argc, argv, &args);
  if (ret != 0) {
    return ret;
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
    std::string assoc_method_s;
    po::options_description desc("Allowed options");
    desc.add_options()
      ("help", "produce help message.")
      ("video-file,v",
       po::value<string>(&args->video_filename)->required(),
       "path to input video file.")
      ("cascade-classifier,c",
       po::value<string>(&args->cascade_filename)->required(),
       "path to cascade classifier file.")
      ("output-dir,o",
       po::value<string>(&args->output_dir)->default_value("output"),
       "path to output directory.")
      ("target-fps,f",
       po::value<double>(&args->target_fps)->default_value(10.0),
       "fps at which video will be scanned.")
      ("detection-scale,s",
       po::value<double>(&args->detection_scale)->default_value(1.0),
       "scale at which image will be transformed during detection.")
      ("association-threshold,a",
       po::value<double>(&args->assoc_threshold)->default_value(0.5),
       "threshold for probability used during association.")
      ("association-method,m",
       po::value<string>(&assoc_method_s)->required(),
       "association method. should be one of 'intersect', and 'optflow'.")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      cout << desc << '\n';
      return false;
    }
    po::notify(vm);

    ugproj::AssociationMethod& assoc_method = args->assoc_method;
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
