#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

#include <cstdio>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
namespace fs = boost::filesystem;

static CascadeClassifier cascade;
void detectFaces(Mat& frame, vector<Rect>& faces, const float scale=1.0);
void drawRect(Mat& frame, int id, Rect& facePosition);

int main(int argc, const char** argv) {
    if (argc != 4) {
        printf("Usage: FaceDetector <video_filename> "
                "<cascade_filename> <output_dir> \n");
        return -1;
    }

    const char* videoFilename = argv[1];
    const char* cascadeFilename = argv[2];
    const char* outputDir = argv[3];

    VideoCapture cap(videoFilename);
    if (!cap.isOpened())
        return -1;
    if (!cascade.load(cascadeFilename))
        return -1;
    fs::path outputPath(outputDir);
    if (!fs::is_directory(outputPath) && !fs::create_directory(outputPath))
        return -1;

    char filename[100];
    fs::path filepath;
    int pos = 0;
    Mat frame;

    while (1) {
        cap >> frame;
        if (frame.empty())
            break;

        printf("Detecting faces in frame #%d... ", pos);
        vector<Rect> faces;
        // detect position of faces here
        detectFaces(frame, faces);
        printf("Found %d faces.\n", faces.size());

        printf("Drawing rectangles on detected faces... ");
        // draw rectangles here
        for (int i = 0, size = faces.size();
             i < size;
             ++i)
            drawRect(frame, i, faces[i]);
        printf("done.\n");

        printf("Writing frame #%d... ", pos);

        sprintf(filename, "%.3d.jpg", pos);
        filepath = outputPath / fs::path(filename);
        imwrite(filepath.native(), frame);

        printf("done.\n");
        ++pos;
    }

    return 0;
}

void detectFaces(Mat& frame, vector<Rect>& faces, const float scale) {
    Mat gray;
    Mat smallImg(cvRound(frame.rows / scale),
                 cvRound(frame.cols / scale),
                 CV_8UC1);

    cvtColor(frame, gray, COLOR_BGR2GRAY);
    equalizeHist(gray, gray);
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
        Rect new_r(r->x * scale, r->y * scale,
                   r->width * scale, r->height * scale);
        faces.push_back(new_r);
    }
}

void drawRect(Mat& frame, int id, Rect& facePosition) {
    rectangle(frame,
              cvPoint(facePosition.x, facePosition.y),
              cvPoint(
                facePosition.x + facePosition.width - 1,
                facePosition.y + facePosition.height - 1),
              CV_RGB(0, 255, 0));
}
