#include <iostream>
#include "image.h"
#include "optical_flow.h"
#include "image_file.h"

//using namespace rvslam;
using namespace opticalflow;

void convertImg(MCImageDoubleX& src, MCImageRGB8& dst) {
  for (int r=0; r<dst.height(); r++)
    for (int c=0; c<dst.width(); c++)
      for (int ch=0; ch<dst.num_channels(); ch++)
        dst(c, r, ch) = static_cast<uint8_t>(src(c, r, ch)*255);
}

int main(int argc, char** argv) {
  MCImageRGB8 Im1, Im2;
  std::string im1_name = "car0.png";
  std::string im2_name = "car1.png";

  if (argc == 3) {
    im1_name = argv[2];
    im2_name = argv[3];
  }

  ReadImage(im1_name, &Im1);
  ReadImage(im2_name, &Im2);

  MCImageDoubleX Im1d(Im1.width(), Im1.height(), Im1.num_channels()), 
                 Im2d(Im1.width(), Im1.height(), Im1.num_channels()); 
  for (int r=0; r<Im1.height(); r++) {
    for (int c=0; c<Im1.width(); c++) {
      for (int ch=0; ch<Im1.num_channels(); ch++) {
        Im1d(c,r,ch) = (double)Im1(c,r,ch)/255;
        Im2d(c,r,ch) = (double)Im2(c,r,ch)/255;
      }
    }
  }

  //Convert rgb images into double
  MCImageDoubleX vx, vy;
  
  double alpha = .3, ratio = .75;
  int minWidth = 20, nOutIter = 7, nInIter = 1, nSORIter = 30;

  MCImageDoubleX warpI2;
  OpticalFlow::Coarse2FineFlow(vx, vy, warpI2, Im1d, Im2d,
      alpha, ratio, minWidth, nOutIter, nInIter, nSORIter);

  FILE *fp_u, *fp_v;
  fp_u = fopen("u.txt", "wt");
  fp_v = fopen("v.txt", "wt");
  for (int r=0; r<vx.height(); r++) {
    for (int c=0; c<vx.width(); c++) {
      fprintf(fp_u, "%lf ", vx(c, r, 0));
      fprintf(fp_v, "%lf ", vy(c, r, 0));
    }
    fprintf(fp_u, "\n");
    fprintf(fp_v, "\n");
  }
  fclose(fp_u);
  fclose(fp_v);

  //Convert double images into rgb
  MCImageRGB8 wImg(warpI2.width(), warpI2.height(),3);
  MCImageRGB8 vxImg(vx.width(), vx.height(), 3);
  MCImageRGB8 vyImg(vy.width(), vy.height(), 3);
  convertImg(warpI2, wImg);
  convertImg(vx, vxImg);
  convertImg(vy, vyImg);

  WriteImage(wImg, "warpI2.png");
  WriteImage(vxImg, "u.png");
  WriteImage(vyImg, "v.png");
  //WriteImage(Im2, "warpI2.png");
  return 0;
}
