#include <math.h>

#include "gaussian_pyramid.h"
#include "optical_flow.h"

namespace opticalflow {
GaussianPyramid::GaussianPyramid(void) {
  img_pyramid = NULL;
}

GaussianPyramid::~GaussianPyramid(void) {
  if (img_pyramid != NULL)
    delete []img_pyramid;
}

void GaussianPyramid::ConstructPyramid(const MCImageDoubleX& image, double ratio, int minWidth) {
  if (ratio > 0.98 || ratio < 0.4)
    ratio = 0.75;
  num_levels = log((double)minWidth/image.width())/log(ratio);
  if (img_pyramid != NULL)
    delete []img_pyramid;

  img_pyramid = new MCImageDoubleX[num_levels];
  img_pyramid[0].Copy(image);
  double baseSigma = (1/ratio-1);
  int n = log(0.25)/log(ratio);
  double nSigma = baseSigma * n;
  for (int i=1; i<num_levels; i++) {
    MCImageDoubleX foo1, foo2;
    double rate = 1.f;
    if (i <= n) {
      double sigma = baseSigma * i;
      //GaussianSmoothingImage<MCImageDoubleX, MCImageDoubleX>(image, sigma, &foo, 3*sigma);
      GaussianSmoothing(foo1, image, sigma, 3*sigma);
      rate = pow(ratio, i);
    }
    else {
      //GaussianSmoothingImage<MCImageDoubleX, MCImageDoubleX>(img_pyramid[i-n], nSigma, &foo, 3*nSigma);
      GaussianSmoothing(foo1, img_pyramid[i-n], nSigma, 3*nSigma);
      rate = (double)pow(ratio,i)*image.width()/foo1.width();
    }
    OpticalFlow::ResizeImage(foo1, foo2, rate);
    img_pyramid[i].Copy(foo2);
  }
}


void GaussianPyramid::GaussianSmoothing(MCImageDoubleX& image, const MCImageDoubleX& target, double sigma, int fsize) {
  if (!OpticalFlow::matchDimension(image, target)) {
    image.Resize(target.width(), target.height(), target.num_channels());
    image.fill(0);
  }
  Eigen::ArrayXXd gFilter;
  gFilter.resize(1, fsize*2+1);
  double sum = 0;
  sigma = sigma*sigma*2;
  for (int i = -fsize; i <= fsize; i++) {
    gFilter(0, i+fsize) = exp(-(double)(i*i)/sigma);
    sum += gFilter(0, i+fsize);
  }
  for (int i = 0; i < 2*fsize+1; i++)
    gFilter(0, i) /= sum;

  OpticalFlow::imfilter_hv(target, image, gFilter, fsize, gFilter, fsize);
}


void GaussianPyramid::SavePyramid() {
  for (int i = 0; i < nlevels(); i++) {
    //Image(i)
  }
}

}

