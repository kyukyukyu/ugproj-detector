#ifndef _OPTICAL_FLOW_H_
#define _OPTICAL_FLOW_H_

#include <vector>
#include "image.h"

using namespace std;
//using namespace rvslam;

namespace opticalflow {

  enum ColorType { RGB, BGR };

  typedef MCImage<double, 1> MCImageDouble;
  typedef MCImage<double, 3> MCImageDouble3;
  typedef MCImage<double, Eigen::Dynamic> MCImageDoubleX;

class OpticalFlow
{
public:
  OpticalFlow() {}
  ~OpticalFlow() {}

  static void SmoothFlowSOR(const MCImageDoubleX &Im1, const MCImageDoubleX &Im2, MCImageDoubleX& warpIm2, MCImageDoubleX& u, MCImageDoubleX& v,
      double alpha, int nOuterFPIterations, int nInnerFPIterations, int nSORIterations);
  static void Coarse2FineFlow(MCImageDoubleX& vx, MCImageDoubleX& vy, MCImageDoubleX& warpI2, const MCImageDoubleX &Im1, const MCImageDoubleX &Im2, 
      double alpha, double ratio, int minWidth, int nOuterFPIterations, int nInnerFPIterations, int nCGIterations);
  static const int EnforceRange(const int& x, const int& MaxValue) { return min(max(x, 0), MaxValue - 1); };

  static const MCImageDoubleX Mult(const MCImageDoubleX& im1, const MCImageDoubleX& im2, const MCImageDoubleX& im3);
  static void multiplyWith(MCImageDoubleX& image, double ratio);
  static void add(MCImageDoubleX& img1, MCImageDoubleX& img2, MCImageDoubleX& result);
  static void add(double val, MCImageDoubleX& img, MCImageDoubleX& result);
  static void subtract(MCImageDoubleX& img1, MCImageDoubleX& img2, MCImageDoubleX& result);
  
  static void hfiltering(const MCImageDoubleX& pSrcImage, MCImageDoubleX& pDstImage, int width, int height, int nChannels, const Eigen::ArrayXXd& pfilter1D, int fsize);
  static void vfiltering(const MCImageDoubleX& pSrcImage, MCImageDoubleX& pDstImage, int width, int height, int nChannels, const Eigen::ArrayXXd& pfilter1D, int fsize);
  static void imfilter_hv(const MCImageDoubleX& input, MCImageDoubleX& result, const Eigen::ArrayXXd& hfilter, int hfsize, const Eigen::ArrayXXd& vfilter, int vfsize);
  
  static void dx(const MCImageDoubleX& input, MCImageDoubleX &result, bool IsAdvancedFilter=false);
  static void dy(const MCImageDoubleX& input, MCImageDoubleX& result, bool IsAdvancedFilter=false);
  static void dxx(const MCImageDoubleX& input, MCImageDoubleX& result);
  static void dyy(const MCImageDoubleX& input, MCImageDoubleX& result);
  static void collapse(const MCImageDoubleX& in, MCImageDoubleX& out);

  static void getDxs(MCImageDoubleX& imdx, MCImageDoubleX& imdy, MCImageDoubleX& imdt, const MCImageDoubleX& im1, const MCImageDoubleX& im2);
  
  static bool matchDimension(const MCImageDoubleX& img1, const MCImageDoubleX& img2);
  static bool matchDimension(const MCImageDoubleX& img, int width, int height, int nchannels);
  static void genInImageMask(MCImageDoubleX& mask, const MCImageDoubleX& vx, const MCImageDoubleX& vy, int interval = 0);
  static void genInImageMask(MCImageDoubleX &mask, const MCImageDoubleX &flow, int interval = 0);
 
  static void Laplacian(MCImageDoubleX& output, const MCImageDoubleX& input, const MCImageDoubleX& weight);
  static void estLaplacianNoise(const MCImageDoubleX& Im1, const MCImageDoubleX& Im2, MCImageDoubleX& para);

  static void BilinearInterpolate(const MCImageDoubleX& pImage,int width,int height,int nChannels,double x,double y,MCImageDoubleX& result, int r, int c);
  static MCImageDoubleX BilinearInterpolate(const MCImageDoubleX& pImage,int width,int height, int nChannels,double x,double y);
  static void warpFL(MCImageDoubleX& warpIm2, const MCImageDoubleX& Im1, const MCImageDoubleX& Im2, const MCImageDoubleX& vx, const MCImageDoubleX& vy);
  static void ResizeImage(const MCImageDoubleX& pSrcImage,MCImageDoubleX& pDstImage,double Ratio);
	static void ResizeImage(const MCImageDoubleX& pSrcImage,MCImageDoubleX& pDstImage, int dstWidth, int dstHeight);
	
  static void im2feature(MCImageDoubleX &imfeature, const MCImageDoubleX &im);
  static void desaturate(const MCImageDoubleX & image, MCImageDoubleX& out, ColorType color_type=BGR);

  static MCImageDoubleX LapPara;
};

}

#endif
