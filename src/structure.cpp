#include "structure.hpp"

using namespace ugproj;

Face::Face(id_type id, FaceCandidate& candidate): id(id) {
    addCandidate(candidate);
}

void Face::addCandidate(FaceCandidate& candidate) {
    candidates.push_back(candidate);
}
