#include "../file_io.h"
#include "../structure.h"
#include "face_tracker.h"

#include <cstdio>
#include <string>
#include <vector>


bool parse_args(int argc, const char** argv, ugproj::Configuration* cfg);

int main(int argc, const char** argv) {
  int ret = 0;

  ugproj::Configuration cfg;
  if (cfg.load(argc, argv)) {
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
  // The list of labeled faces using face tracker.
  std::vector<ugproj::Face> labeled_faces;
  ugproj::FaceTracker tracker;
  tracker.set_input(&input);
  tracker.set_writer(&writer);
  tracker.set_cfg(&cfg);
  ret = tracker.track(&tracked_positions, &labeled_faces);
  if (ret != 0) {
    return ret;
  }

  return 0;
}
