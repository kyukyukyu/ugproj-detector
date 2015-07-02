#include "structure.hpp"

using namespace ugproj;

Face::Face(id_type id, const FaceCandidate& candidate): id(id) {
    addCandidate(candidate);
}

void Face::addCandidate(const FaceCandidate& candidate) {
    candidates.push_back(candidate);
}
