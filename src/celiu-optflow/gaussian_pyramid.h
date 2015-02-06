#ifndef _GAUSSIANPYRAMID_H_
#define _GAUSSIANPYRAMID_H_

#include "image.h"

namespace opticalflow {

typedef MCImage<double, Eigen::Dynamic> MCImageDoubleX;

class GaussianPyramid
{
private:
  MCImageDoubleX *img_pyramid;
  int num_levels;
public:
  GaussianPyramid(void);
  ~GaussianPyramid(void);
  void ConstructPyramid(const MCImageDoubleX& image, double ratio=0.8, int minWidth=30);
  void GaussianSmoothing(MCImageDoubleX& image, const MCImageDoubleX& target, double sigma, int fsize);
  void SavePyramid();
  
  inline int nlevels() const { return num_levels; };
  inline MCImageDoubleX& Image(int index) { return img_pyramid[index]; };
};
}

#endif
