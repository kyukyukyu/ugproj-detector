#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <cstdio>
#include <cmath>
#include <string>
#include <limits>
#include <utility>
#include <vector>

#include "structure.hpp"

using namespace std;
using namespace cv;
namespace fs = boost::filesystem;
namespace po = boost::program_options;

typedef pair<Face::id_type, FaceCandidate*> fc_pair;
class fc_pair_v : public vector<fc_pair> {
    public:
        ~fc_pair_v();
};
fc_pair_v::~fc_pair_v() {
    for (fc_pair_v::iterator it = this->begin();
         it != this->end();
         ++it)
        delete it->second;
}

static CascadeClassifier cascade;
static vector<Face> faces;
static const double DOUBLE_EPSILON = numeric_limits<double>::epsilon();

bool parseOptions(int argc, const char** argv,
        string& videoFilename, string& cascadeFilename, string& outputDir,
        double& targetFps, double& detectionScale, double& associationThreshold);
void detectFaces(Mat& frame, vector<Rect>& rects, const float scale);
void associate(fc_pair_v& prev, fc_pair_v& next,
        double threshold);
void drawRect(Mat& frame, Face::id_type id, const Rect& facePosition);

int main(int argc, const char** argv) {
    string videoFilename, cascadeFilename, outputDir;
    double targetFps, detectionScale, associationThreshold;
    bool parsed =
        parseOptions(
                argc,
                argv,
                videoFilename,
                cascadeFilename,
                outputDir,
                targetFps,
                detectionScale,
                associationThreshold);
    if (!parsed)
        return 1;

    VideoCapture cap(videoFilename);
    if (!cap.isOpened())
        return -1;
    if (!cascade.load(cascadeFilename))
        return -1;
    fs::path outputPath(outputDir);
    if (!fs::is_directory(outputPath) && !fs::create_directory(outputPath))
        return -1;

    double sourceFps = cap.get(CV_CAP_PROP_FPS);

    char filename[100];
    fs::path filepath;
    unsigned long pos = 0;
    unsigned long frameCount = cap.get(CV_CAP_PROP_FRAME_COUNT);
    Mat frame;

    fc_pair_v *prevCandidates = NULL, *currCandidates = NULL;

    while (pos < frameCount) {
        cap.grab();     // grab next frame
        if (fmod(pos, sourceFps / targetFps) - 1.0 > -DOUBLE_EPSILON) {
            ++pos;
            continue;
        }
        cap.retrieve(frame);
        if (frame.empty())
            break;

        printf("Detecting faces in frame #%lu... ", pos);
        // detect position of faces here
        vector<Rect> rects;
        detectFaces(frame, rects, detectionScale);

        if (prevCandidates != NULL)
            delete prevCandidates;
        prevCandidates = currCandidates;
        currCandidates = new fc_pair_v();
        for (vector<Rect>::const_iterator it = rects.begin();
             it != rects.end();
             ++it) {
            Mat cddImage(frame, *it);
            // dynamically allocate cdd to handle multiple candidates
            FaceCandidate* cdd = new FaceCandidate(pos, *it, cddImage);
            currCandidates->push_back(fc_pair(0, cdd));
        }
        printf("Found %lu faces.\n", currCandidates->size());

        // perform association here
        if (prevCandidates == NULL) {
            // skip if the first detection was performed at this time
            printf("Skip association at the first scanned frame...\n");
            goto add_all;
        } else if (prevCandidates->size() == 0) {
            // if there is no candidate for previous frame, add all the
            // candidates as new faces
            printf("No candidate for previous frame: add all candidates as "
                   "new faces.\n");
add_all:
            for (fc_pair_v::iterator it = currCandidates->begin();
                 it != currCandidates->end();
                 ++it) {
                Face::id_type faceId = faces.size();
                it->first = faceId;
                faces.push_back(Face(faceId, *(it->second)));
            }
        } else {
            printf("Performing association for faces... ");
            associate(*prevCandidates, *currCandidates, associationThreshold);
            printf("done.\n");
        }

        printf("Drawing rectangles on detected faces... ");
        // draw rectangles here
        for (fc_pair_v::const_iterator it = currCandidates->begin();
             it != currCandidates->end();
             ++it)
            drawRect(frame, it->first, it->second->rect);
        printf("done.\n");

        printf("Writing frame #%lu... ", pos);

        sprintf(filename, "%.3lu.jpg", pos);
        filepath = outputPath / fs::path(filename);
        imwrite(filepath.native(), frame);

        printf("done.\n");
        ++pos;
    }

    delete prevCandidates;
    delete currCandidates;

    return 0;
}

bool parseOptions(int argc, const char** argv,
        string& videoFilename, string& cascadeFilename, string& outputDir,
        double& targetFps, double& detectionScale, double& associationThreshold) {
    try {
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
        ;

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            cout << desc << '\n';
            return false;
        }
        po::notify(vm);
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return false;
    } catch (...) {
        std::cerr << "Unknown error!\n";
        return false;
    }

    return true;
}

void detectFaces(Mat& frame, vector<Rect>& rects, const float scale) {
    const static Scalar lowerBound(0, 133, 77);
    const static Scalar upperBound(255, 173, 127);
    Mat ycrcb;
    Mat mask;
    Mat gray;
    Mat smallImg(cvRound(frame.rows * scale),
                 cvRound(frame.cols * scale),
                 CV_8UC1);

    cvtColor(frame, ycrcb, CV_BGR2YCrCb);
    inRange(ycrcb, lowerBound, upperBound, mask);

    cvtColor(frame, gray, COLOR_BGR2GRAY);
    equalizeHist(gray, gray);
    gray &= mask;
    resize(gray, smallImg, smallImg.size());

    vector<Rect> facesInGray;
    cascade.detectMultiScale(
            gray,
            facesInGray,
            1.1,
            2,
            0 | CASCADE_SCALE_IMAGE,
            Size(30, 30));

    for (vector<Rect>::const_iterator r = facesInGray.begin();
         r != facesInGray.end();
         ++r) {
        int sourceX = r->x / scale;
        int sourceY = r->y / scale;
        int sourceWidth = r->width / scale;
        int sourceHeight = r->height / scale;
        Mat croppedMask(mask,
                Range(sourceY, sourceY + sourceHeight),
                Range(sourceX, sourceX + sourceWidth));
        double m = norm( mean(croppedMask) );
        if (m/256 < 0.8)
            continue;
        Rect new_r(sourceX, sourceY, sourceWidth, sourceHeight);
        rects.push_back(new_r);
    }
}

void associate(fc_pair_v& prev, fc_pair_v& next,
        double threshold) {
    typedef fc_pair_v::size_type size_type;

    const size_type
        prevSize = prev.size(), nextSize = next.size();

    double **prob;
    // array allocation
    prob = new double *[prevSize];
    for (size_type i = 0; i < prevSize; ++i)
        prob[i] = new double[nextSize];

    // calculate probability
    for (size_type i = 0; i < prevSize; ++i) {
        const Rect& rectI = prev[i].second->rect;
        for (size_type j = 0; j < nextSize; ++j) {
            const Rect& rectJ = next[j].second->rect;
            Rect intersect = rectI & rectJ;
            int intersectArea = intersect.area();
            int unionArea =
                rectI.area() + rectJ.area() - intersectArea;
            prob[i][j] = (double)intersectArea / unionArea;
        }
    }

    // match candidates by probability
    for (size_type j = 0; j < nextSize; ++j) {
        double max = -1;
        size_type maxRow;

        for (size_type i = 0; i < prevSize; ++i) {
            if (prob[i][j] > max && prob[i][j] > threshold) {
                max = prob[i][j];
                maxRow = i;
            }
        }

        vector<Face>::size_type faceId;
        if (max > 0) {
            faceId = prev[maxRow].first;
        } else {
            faceId = faces.size();
            faces.push_back(Face(faceId));
        }
        next[j].first = faceId;
        faces[faceId].addCandidate(*(next[j].second));
    }

    // array deallocation
    for (size_type i = 0; i < prevSize; ++i)
        delete[] prob[i];
    delete[] prob;
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
