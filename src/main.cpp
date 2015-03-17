#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <cstdio>
#include <cmath>
#include <string>
#include <limits>
#include <utility>
#include <vector>

#include "structure.hpp"
#include "detector/detector.hpp"
#include "associator/associator.hpp"
#include "celiu-optflow/optical_flow.h"
#include "optflow/flow_to_color.hpp"

using namespace std;
using namespace cv;
using namespace ugproj;
namespace fs = boost::filesystem;
namespace po = boost::program_options;

typedef unsigned long temp_pos_t;
typedef enum {
    intersect, optflow, sift
} asc_meth_t;

static vector<Face> faces;
static vector<temp_pos_t> frameNumbers;
static const double DOUBLE_EPSILON = numeric_limits<double>::epsilon();

bool parseOptions(int argc, const char** argv,
        string& videoFilename, string& cascadeFilename, string& outputDir,
        double& targetFps, double& detectionScale, double& associationThreshold,
        asc_meth_t& associationMethod);
void calculateOptFlow(Mat& frame1, Mat& frame2, OptFlowArray& vx, OptFlowArray& vy);
void drawRect(Mat& frame, Face::id_type id, const Rect& facePosition);

int main(int argc, const char** argv) {
    string videoFilename, cascadeFilename, outputDir;
    double targetFps, detectionScale, associationThreshold;
    asc_meth_t associationMethod;
    bool parsed =
        parseOptions(
                argc,
                argv,
                videoFilename,
                cascadeFilename,
                outputDir,
                targetFps,
                detectionScale,
                associationThreshold,
                associationMethod);
    if (!parsed)
        return 1;

    VideoCapture cap(videoFilename);
    if (!cap.isOpened())
        return -1;
    CascadeClassifier cascade;
    if (!cascade.load(cascadeFilename))
        return -1;
    fs::path outputPath(outputDir);
    if (!fs::is_directory(outputPath) && !fs::create_directory(outputPath))
        return -1;

    double sourceFps = cap.get(CV_CAP_PROP_FPS);

    char filename[100];
    fs::path filepath;
    temp_pos_t pos = 0;
    temp_idx_t index = 0;
    temp_idx_t prevIndex;
    unsigned long frameCount = cap.get(CV_CAP_PROP_FRAME_COUNT);
    double frameWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    double frameHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    Mat frame, prevFrame;

    FaceDetector detector(cascade);
    OpticalFlowManager flowManager(frameWidth, frameHeight);

    FaceAssociator::fc_v *prevCandidates = NULL, *currCandidates = NULL;

    while (pos < frameCount) {
        cap.grab();     // grab next frame
        if (fmod(pos, sourceFps / targetFps) - 1.0 > -DOUBLE_EPSILON) {
            ++pos;
            continue;
        }
        cap.retrieve(frame);
        if (frame.empty())
            break;

        // save frame index - frame number mapping
        frameNumbers.push_back(pos);

        // calculate optical flow if this is not the first frame
        if (associationMethod == optflow && index > 0) {
            printf("Calculating optical flow between frame #%lu and frame #%lu... ", frameNumbers[index - 1], frameNumbers[index]);

            OptFlowArray* vx = new OptFlowArray;
            OptFlowArray* vy = new OptFlowArray;
            OpticalFlowManager::flow_t* flow =
                new OpticalFlowManager::flow_t(vx, vy);
            calculateOptFlow(prevFrame, frame, *vx, *vy);
            flowManager.append(flow);

            printf("done.\n");
        }
		
        printf("Detecting faces in frame #%lu... ", pos);
        // detect position of faces here
        vector<Rect> rects;
        detector.detectFaces(frame, rects, detectionScale);

        if (index > 0)
            delete prevCandidates;
        prevCandidates = currCandidates;
        currCandidates = new FaceAssociator::fc_v();
        for (vector<Rect>::const_iterator it = rects.begin();
             it != rects.end();
             ++it) {
            Mat cddImage(frame, *it);
            // dynamically allocate cdd to handle multiple candidates
            FaceCandidate* cdd = new FaceCandidate(index, *it, cddImage);
            currCandidates->push_back(cdd);
        }
        printf("Found %lu faces.\n", currCandidates->size());

        // perform association here
        if (index == 0) {
            // skip if the first detection was performed at this time
            printf("Skip association at the first scanned frame...\n");
            goto add_all;
        } else if (prevCandidates->size() == 0) {
            // if there is no candidate for previous frame, add all the
            // candidates as new faces
            printf("No candidate for previous frame: add all candidates as "
                   "new faces.\n");
add_all:
            for (FaceAssociator::fc_v::iterator it = currCandidates->begin();
                 it != currCandidates->end();
                 ++it) {
                Face::id_type faceId = faces.size() + 1;
                (*it)->faceId = faceId;
                faces.push_back(Face(faceId, **it));
            }
        } else {
            printf("Performing association for faces... ");
            FaceAssociator* associator;

            if (associationMethod == intersect) {
                associator =
                    new IntersectionFaceAssociator(faces, *prevCandidates,
                                                   *currCandidates, associationThreshold);
            } else if (associationMethod == optflow) {
                associator =
                    new OpticalFlowFaceAssociator(faces, *prevCandidates,
                                                  *currCandidates, flowManager,
                                                  prevIndex, index,
                                                  associationThreshold);

            } else {        // sift
                associator =
                    new SiftFaceAssociator(faces, *prevCandidates,
                                           *currCandidates, prevFrame,
                                           frame, associationThreshold);

            }
            associator->associate();
            delete associator;
            printf("done.\n");
        }

        // set current frame as previous frame for next iteration
        frame.copyTo(prevFrame);

        printf("Drawing rectangles on detected faces... ");
        // draw rectangles here
        for (FaceAssociator::fc_v::const_iterator it = currCandidates->begin();
             it != currCandidates->end();
             ++it) {
            FaceCandidate* candidate = *it;
            drawRect(frame, candidate->faceId, candidate->rect);
        }
        printf("done.\n");

        printf("Writing frame #%lu... ", pos);

        sprintf(filename, "%.3lu.jpg", pos);
        filepath = outputPath / fs::path(filename);
        imwrite(filepath.string(), frame);

        printf("done.\n");
        if (currCandidates->size() > 0) {
            // set prevIndex to index if any face is detected on this frame
            prevIndex = index;
        }
        ++index;
        ++pos;
    }

    delete prevCandidates;
    delete currCandidates;

    return 0;
}

bool parseOptions(int argc, const char** argv,
        string& videoFilename, string& cascadeFilename, string& outputDir,
        double& targetFps, double& detectionScale, double& associationThreshold,
        asc_meth_t& associationMethod) {
    try {
        string _associationMethod;
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message.")
            ("video-file,v",
             po::value<string>(&videoFilename)->required(),
             "path to input video file.")
            ("cascade-classifier,c",
             po::value<string>(&cascadeFilename)->required(),
             "path to cascade classifier file.")
            ("output-dir,o",
             po::value<string>(&outputDir)->default_value("output"),
             "path to output directory.")
            ("target-fps,f",
             po::value<double>(&targetFps)->default_value(10.0),
             "fps at which video will be scanned.")
            ("detection-scale,s",
             po::value<double>(&detectionScale)->default_value(1.0),
             "scale at which image will be transformed during detection.")
            ("association-threshold,a",
             po::value<double>(&associationThreshold)->default_value(0.5),
             "threshold for probability used during association.")
            ("association-method,m",
             po::value<string>(&_associationMethod)->required(),
             "association method. should be one of 'intersect', and 'optflow'.")
        ;

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            cout << desc << '\n';
            return false;
        }
        po::notify(vm);

        if (_associationMethod == "intersect") {
            associationMethod = intersect;
        } else if (_associationMethod == "optflow") {
            associationMethod = optflow;
        } else if (_associationMethod == "sift") {
            associationMethod = sift;
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

void calculateOptFlow(Mat& frame1, Mat& frame2, OptFlowArray& vx, OptFlowArray& vy) {
    // convert images
    opticalflow::MCImageDoubleX prevImg(frame1.cols, frame1.rows, frame1.channels());
    opticalflow::MCImageDoubleX nextImg(frame2.cols, frame2.rows, frame2.channels());
    for (int y = 0; y < frame1.rows; ++y) {
        for (int x = 0; x < frame1.cols; ++x) {
            cv::Vec3b prevPixel(frame1.at<cv::Vec3b>(y, x)),
                      nextPixel(frame2.at<cv::Vec3b>(y, x));
            for (int d = 0; d < frame1.channels(); ++d) {
                prevImg(x, y, d) = (double)prevPixel[d] / 255;
                nextImg(x, y, d) = (double)nextPixel[d] / 255;
            }
        }
    }

    // calculate optical flow
    static double alpha = .012, ratio = .75;
    static int minWidth = 40, nOutIter = 7, nInIter = 1, nSORIter = 30;

    opticalflow::MCImageDoubleX warpI2;
    opticalflow::OpticalFlow::Coarse2FineFlow(
            vx, vy, warpI2, prevImg, nextImg,
            alpha, ratio, minWidth, nOutIter, nInIter, nSORIter);
}

Scalar colorPreset[] = {
    CV_RGB(0, 255, 0),
    CV_RGB(255, 0, 0),
    CV_RGB(0, 0, 255),
    CV_RGB(255, 255, 0),
    CV_RGB(255, 0, 255),
    CV_RGB(0, 255, 255)
};

void drawRect(Mat& frame, Face::id_type id, const Rect& facePosition) {
    Scalar color = colorPreset[id % (sizeof(colorPreset) / sizeof(Scalar))];
    rectangle(frame,
              cvPoint(facePosition.x, facePosition.y),
              cvPoint(
                facePosition.x + facePosition.width - 1,
                facePosition.y + facePosition.height - 1),
              color);
    putText(frame,
            to_string(id),
            cvPoint(facePosition.x + 4, facePosition.y + 4),
            FONT_HERSHEY_PLAIN,
            1.0,
            color);
}
