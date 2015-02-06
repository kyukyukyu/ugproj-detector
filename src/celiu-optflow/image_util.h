// image_util.h
//
// Author: Jongwoo Lim (jongwoo.lim@gmail.com)
//         Hyon Lim (limhyon@gmail.com)

#ifndef _RVSLAM_IMAGE_UTIL_H_
#define _RVSLAM_IMAGE_UTIL_H_

#include <iostream>
#include <cstdio>
#include <cstdarg>
#include <cmath>
#include <iomanip>  // setfill(' ')
#include <glog/logging.h>
#include <Eigen/Dense>
#include "image.h"

//namespace rvslam {
namespace opticalflow {
typedef MCImage<double, Eigen::Dynamic> MCImageDoubleX;

extern uint8_t _g_fontimgdata[];

//-----------------------------------------------------------------------------
// Eigen::Array and Image convolution functions
//  - two functions use slightly different logic.

template <typename Input, typename Kernel, typename Output> inline
void Convolve(const Eigen::DenseBase<Input>& image,
              const Eigen::DenseBase<Kernel>& kernel, Output* out) {
  const int rows = image.rows();
  const int cols = image.cols();
  const int k_rows = kernel.rows();
  const int k_cols = kernel.cols();
  if (rows <= 0 || cols <= 0 || k_rows <= 0 || k_cols <= 0) return;
  const int k_half_rows = k_rows / 2;
  const int k_half_cols = k_cols / 2;
  const int k_rows_left = k_rows - k_half_rows;
  const int k_cols_left = k_cols - k_half_cols;
  out->Resize(rows, cols);
  out->fill(0);
//  LOG(ERROR) << rows << "x" << cols << " - " << out->rows() << "x"
//      << out->cols() << ", " << k_half_rows << "," << k_half_cols << ", "
//      << k_rows_left << "," << k_cols_left;
  for (int r = 0; r < k_rows; ++r) {
    for (int c = 0; c < k_cols; ++c) {
      int r0 = 0, c0 = 0, r1 = 0, c1 = 0, nr = rows, nc = cols;
      if (r < k_half_rows) nr -= (r0 = k_half_rows - r);
      if (c < k_half_cols) nc -= (c0 = k_half_cols - c);
      if (r > k_half_rows) nr -= (r1 = r - k_half_rows);
      if (c > k_half_cols) nc -= (c1 = c - k_half_cols);
      out->block(r1, c1, nr, nc) +=
          kernel(r, c) * image.block(r0, c0, nr, nc);
    }
  }
}

template <typename Input, typename Kernel, typename Output> inline
void ConvolveImage(const Input& image, const Kernel& kernel, Output* out) {
  const int rows = image.width();
  const int cols = image.height();
  const int k_rows = kernel.rows();
  const int k_cols = kernel.cols();
  if (rows <= 0 || cols <= 0 || k_rows <= 0 || k_cols <= 0) return;
  const int k_half_rows = k_rows / 2;
  const int k_half_cols = k_cols / 2;
  const int k_rows_left = k_rows - k_half_rows;
  const int k_cols_left = k_cols - k_half_cols;
  out->Resize(rows, cols, image.depth());
  out->fill(0);
//  LOG(ERROR) << rows << "x" << cols << " - " << out->rows() << "x"
//      << out->cols() << ", " << k_half_rows << "," << k_half_cols << ", "
//      << k_rows_left << "," << k_cols_left;
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      const int r0 = r < k_half_rows ? k_half_rows - r : 0;
      const int r1 = r > rows - k_rows_left ? k_half_rows + rows - r: k_rows;
      const int c0 = c < k_half_cols ? k_half_cols - c : 0;
      const int c1 = c > cols - k_cols_left ? k_half_cols + cols - c: k_cols;

//      typename Output::PixelType weighted_sum = 0;
      for (int i = r0; i < r1; ++i) {
        for (int j = c0; j < c1; ++j) {
          (*out)(r, c) +=
          image(r - k_half_rows + i, c - k_half_cols + j) *
              kernel(k_rows - i - 1, k_cols - j - 1);
        }
      }
//      (*out)(r, c) = static_cast<typename Output::Scalar>(weighted_sum);
    }
  }
}

//-----------------------------------------------------------------------------
// Eigen::Array and Image Gaussian smoothing functions

template <typename Kernel> inline
void BuildGaussianKernel(double sigma, Kernel* kernel) {
  const int k_rows = kernel->rows();
  const int k_cols = kernel->cols();
  const double k_half_rows = (k_rows - 1) / 2.0;
  const double k_half_cols = (k_cols - 1) / 2.0;
  const double coef = -1 / (2 * sigma * sigma);
  for (int r = 0; r < k_rows; ++r) {
    for (int c = 0; c < k_cols; ++c) {
      const double dr = r - k_half_rows, dc = c - k_half_cols;
      (*kernel)(r, c) = static_cast<typename Kernel::Scalar>(
          exp(coef * (dr * dr + dc * dc)));
    }
  }
  *kernel *= 1.0 / kernel->sum();
}

template <typename Array2D>
inline void BuildGaussianKernel1D(double sigma, Array2D* kernel,
                                  int kernel_size = 0) {
  if (kernel_size <= 0) kernel_size = ceil(3 * sigma);
  kernel->resize(2 * kernel_size + 1, 1);
  const double coef = -1 / (2 * sigma * sigma);
  for (int i = -kernel_size; i <= kernel_size; ++i) {
    (*kernel)(kernel_size + i, 0) = exp(coef * i * i);
  }
  (*kernel) *= 1.0 / kernel->sum();
}

template <typename Array2D> inline
void GaussianSmoothing(const Eigen::DenseBase<Array2D>& image, double sigma,
                       Array2D* out, int kernel_size = 0) {
  Array2D kernel, tmp;
  BuildGaussianKernel1D(sigma, &kernel, kernel_size);
  Convolve(image, kernel, &tmp);
  Convolve(tmp, kernel.transpose(), out);
}

template <typename Input, typename Output> inline
void GaussianSmoothingImage(const Input& image, double sigma,
                            Output* out, int kernel_size = 0) {
  Eigen::ArrayXXd kernel;
  BuildGaussianKernel1D(sigma, &kernel, kernel_size);
//LOG(ERROR) << std::setfill(' ') << sigma << std::endl << kernel;
  Input tmp;
  ConvolveImage(image, kernel, &tmp);
  ConvolveImage(tmp, kernel.transpose(), out);
}

//-----------------------------------------------------------------------------
// Eigen::Array and Image bilinear interpolation functions

template <typename Array2D, typename T> inline
void Interp2Patch(const Eigen::DenseBase<Array2D>& image, T x, T y,
                  Eigen::DenseBase<Array2D>* patch) {
  const int nr = patch->rows(), nc = patch->cols();
  const int x0 = static_cast<int>(x - (nr / 2)), x1 = x0 + 1;
  const int y0 = static_cast<int>(y - (nc / 2)), y1 = y0 + 1;
  const T rx = x - x0, ry = y - y0;
  Array2D tmp(nr ,nc);
  tmp = ((1 - rx) * (1 - ry)) * image.block(x0, y0, nr, nc);
  tmp += (rx * (1 - ry)) * image.block(x1, y0, nr, nc);
  tmp += ((1 - rx) * ry) * image.block(x0, y1, nr, nc);
  tmp += (rx * ry) * image.block(x1, y1, nr, nc);
  *patch = tmp;
}

template <typename Image, typename Pixel, bool SAFE> inline
Pixel Interp2_(const Image& image, int w, int h, double x, double y) {
  if (SAFE) {
    x = std::max(0.0, std::min(x, w - 1.0));
    y = std::max(0.0, std::min(y, h - 1.0));
  }
  const int x0 = static_cast<int>(x), x1 = x0 + 1;
  const int y0 = static_cast<int>(y), y1 = y0 + 1;
  const double rx1 = x - x0, ry1 = y - y0, rx0 = 1 - rx1, ry0 = 1 - ry1;
  if (SAFE) {
    // This is to handle image(a,b) returns nan.
//LOG(ERROR) << x << ", " << y << " / " << x0 <<"," << y0 << ", " <<x1<<","<<y1<<"/ " << w << "x" << h << " / " << rx << ", " << ry;
    if (ry0 <= 0) {
      if (rx0 <= 0) return image(x1, y1);
      if (rx1 <= 0) return image(x0, y1);
      return static_cast<Pixel>(image(x0, y1) * rx0 + image(x1, y1) * rx1);
    }
    if (ry1 <= 0) {
      if (rx0 <= 0) return image(x1, y0);
      if (rx1 <= 0) return image(x0, y0);
      return static_cast<Pixel>(image(x0, y0) * rx0 + image(x1, y0) * rx1);
    }
    if (rx0 <= 0) {
      return static_cast<Pixel>(image(x1, y0) * ry0 + image(x1, y1) * ry1);
    }
    if (rx1 <= 0) {
      return static_cast<Pixel>(image(x0, y0) * ry0 + image(x0, y1) * ry1);
    }
  }
  return static_cast<Pixel>((image(x0, y0) * rx0 + image(x1, y0) * rx1) * ry0 +
                            (image(x0, y1) * rx0 + image(x1, y1) * rx1) * ry1);
}

template <typename Array2D> inline
typename Array2D::Scalar Interp2(const Eigen::DenseBase<Array2D>& image,
                                 double x, double y) {
  return Interp2_<Eigen::DenseBase<Array2D>, typename Array2D::Scalar, false>(
      image, image.rows(), image.cols(), x, y);
}

template <typename Array2D> inline
typename Array2D::Scalar SafeInterp2(const Eigen::DenseBase<Array2D>& image,
                                     double x, double y) {
  return Interp2_<Eigen::DenseBase<Array2D>, typename Array2D::Scalar, true>(
      image, image.rows(), image.cols(), x, y);
}

template <typename Image> inline
typename Image::PixelType Interp2Image(const Image& image, double x, double y) {
  return Interp2_<Image, typename Image::PixelType, false>(
      image, image.width(), image.height(), x, y);
}

template <typename Image> inline
typename Image::PixelType SafeInterp2Image(const Image& image,
                                           double x, double y) {
  return Interp2_<Image, typename Image::PixelType, true>(
      image, image.width(), image.height(), x, y);
}

//-----------------------------------------------------------------------------
// Eigen::Array and Image resize functions

inline int GetResizedSize_(int w, double ratio) { return ceil(w * ratio); }

template <typename Input, typename Output, typename Pixel> inline
void Resize_(const Input& image, int w0, int h0, int w, int h,
             double x_ratio, double y_ratio, Output* out) {
  // Matlab's imresize with method = bilinear and antialising = false.
  const double x_inc = 1 / x_ratio, y_inc = 1 / y_ratio;
  double y0 = (y_inc - 1) / 2;
  for (int y = 0; y < h - 1; ++y, y0 += y_inc) {
    double x0 = (x_inc - 1) / 2;
    for (int x = 0; x < w - 1; ++x, x0 += x_inc) {
      (*out)(x, y) = Interp2_<Input, Pixel, false>(image, w0, h0, x0, y0);
    }
    (*out)(w - 1, y) = Interp2_<Input, Pixel, true>(image, w0, h0, x0, y0);
  }
  double x0 = (x_inc - 1) / 2;
  for (int x = 0; x < w; ++x, x0 += x_inc) {
    (*out)(x, h - 1) = Interp2_<Input, Pixel, true>(image, w0, h0, x0, y0);
  }
}

template <typename Array2D> inline
void Resize(const Eigen::DenseBase<Array2D>& image,
            double x_ratio, double y_ratio, Array2D* out) {
  const int w0 = image.rows(), h0 = image.cols();
  const int w = GetResizedSize_(w0, x_ratio), h = GetResizedSize_(h0, y_ratio);
  out->resize(w, h);
  Resize_<Eigen::DenseBase<Array2D>, Array2D, typename Array2D::Scalar>(
      image, w0, h0, w, h, x_ratio, y_ratio, out);
}

template <typename Array2D> inline
void Resize(const Eigen::DenseBase<Array2D>& image, double ratio,
            Array2D* out) {
  Resize(image, ratio, ratio, out);
}

template <typename Array2D> inline
void Resize(const Eigen::DenseBase<Array2D>& image, Array2D* out) {
  const int w0 = image.rows(), h0 = image.cols();
  const int w = out->rows(), h = out->cols();
  Resize_<Eigen::DenseBase<Array2D>, Array2D, typename Array2D::Scalar>(
      image, w0, h0, w, h, w / (double) w0, h / (double) h0, out);
}

template <typename Image> inline
void ResizeImage(const Image& image, double x_ratio, double y_ratio,
                 Image* out) {
  const int w0 = image.width(), h0 = image.height();
  const int w = GetResizedSize_(w0, x_ratio), h = GetResizedSize_(h0, y_ratio);
  out->Resize(w, h, image.depth());
  Resize_<Image, Image, typename Image::PixelType>(
      image, w0, h0, w, h, x_ratio, y_ratio, out);
}

template <typename Image> inline
void ResizeImage(const Image& image, double ratio, Image* out) {
  ResizeImage(image, ratio, ratio, out);
}

template <typename Image> inline
void ResizeImage(const Image& image, Image* out) {
  const int w0 = image.width(), h0 = image.height();
  const int w = out->width(), h = out->height();
  Resize_<Image, Image, typename Image::PixelType>(
      image, w0, h0, w, h, w / (double) w0, h / (double) h0, out);
}

/*
template <typename Input, typename Scalar> inline
void Resize(const Eigen::DenseBase<Input>& image,
            Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic>* out) {
  const double x_ratio = out->width() / (double) image.width();
  const double y_ratio = out->height() / (double) image.height();
  for (int y = 0; y < out->height(); ++y) {
    for (int x = 0; x < out->width(); ++x) {
      (*out)(r, c) = static_cast<Scalar>(
          (image(r0, c0) + image(r0, c0 + 1) +
           image(r0 + 1, c0) + image(r0 + 1, c0 + 1)) / 4);
    }
  }
}
*/
template <typename Input, typename Scalar> inline
void ReduceSize(const Eigen::DenseBase<Input>& image,
                Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic>* out) {
  int rows = image.rows() / 2, cols = image.cols() / 2;
  out->resize(rows, cols);
  for (int r = 0, r0 = 0; r < rows; ++r, r0 += 2) {
    for (int c = 0, c0 = 0; c < cols; ++c, c0 += 2) {
      (*out)(r, c) = static_cast<Scalar>(
          (image(r0, c0) + image(r0, c0 + 1) +
           image(r0 + 1, c0) + image(r0 + 1, c0 + 1)) / 4);
    }
  }
}

template <typename Output, typename Input, int DEPTH> inline
void ReduceSize(const MCImage<Input, DEPTH>& image,
                MCImage<Output, DEPTH>* out) {
  int width = image.width() / 2, height = image.height() / 2;
  out->Resize(width, height);
  for (int y = 0, y0 = 0; y < height; ++y, y0 += 2) {
    for (int x = 0, x0 = 0; x < width; ++x, x0 += 2) {
      out->At(x, y) = (image(x0, y0) + image(x0, y0 + 1) +
                       image(x0 + 1, y0) + image(x0 + 1, y0 + 1)) / 4;
    }
  }
}

//-----------------------------------------------------------------------------
// Basic drawing functions.

template <typename T, int DEPTH, typename P, typename S> inline
void DrawDot(MCImage<T, DEPTH>& img, P x, P y, const S& c, int rad = 1) {
  const int w = img.width(), h = img.height();
  if (0 <= x + rad && x - rad < w && 0 <= y + rad && y - rad < h) {
    const int x0 = std::max(0, static_cast<int>(x - rad));
    const int x1 = std::min(w - 1, static_cast<int>(x + rad));
    const int y0 = std::max(0, static_cast<int>(y - rad));
    const int y1 = std::min(h - 1, static_cast<int>(y + rad));
    for (int y = y0; y <= y1; ++y) {
      for (int x = x0; x <= x1; ++x) {
        img(x, y) = c;
      }
    }
  }
}

template <typename T, int DEPTH, typename P, typename S> inline
void DrawCircle(MCImage<T, DEPTH>& img, P x, P y, const S& c, int rad = 5) {
  const int w = img.width(), h = img.height();
  if (0 <= x + rad && x - rad < w && 0 <= y + rad && y - rad < h) {
    int l = (int) rad * cos (M_PI/4);
    for (int a = 0; a <= l ; a++) {
      int b = (int) sqrt ((double)(rad*rad) - (a*a));
      if(a+x < 0 || b+y < 0 || -a+x < 0 || -b+y < 0) continue;
      if(a+x > w || -a+x > w || b+y > h || -b+y > h) continue;
      img(a+x, b+y) = c;
      img(a+x, -b+y) = c;
      img(-a+x, b+y) = c;
      img(-a+x, -b+y) = c;
      img(b+x, a+y) = c;
      img(b+x, -a+y) = c;
      img(-b+x, a+y) = c;
      img(-b+x, -a+y) = c;
    }
  }
}

#define _STRFMT(str,fmt) \
  va_list arg; \
  char str[1024]; \
  va_start(arg, fmt); \
  vsprintf(str, fmt, arg); \
  va_end(arg);

template <typename T, int DEPTH, typename P, typename S> inline
void DrawText(MCImage<T, DEPTH>& img, P x, P y, const S& c, const char *str) {
  // Prepare character map
  Image<unsigned char> img_char;
  img_char.FromBuffer(_g_fontimgdata, 480, 8);

  const int w = img.width(), h = img.height();
  const int len = (int)strlen(str), fw = 5, fh = 8;

  if(x > fw/2 && x < w - fh/2)
    x -= fw/2;

  if(y > fh/2 && y < h - fh/2)
    y -= fh/2;
  
  // for length of character.
  for (register int i = 0, dx=fw; i < len; i++, x+=dx) {
    if (str[i] > ' ' && str[i] <= '~') {
      int sx = (str[i] - '!') * fw; // 0, 5, 10 ...
      for (register int j = sx; j < sx + fw; ++j) { // horz
        for (register int k = 0; k < 8; ++k) { // vert
          const int py = (int)y + k, px = (int)x + j - sx;
          if(img_char(j,k) > 0 && px > 0 && py > 0 && px < w && py < h)
            img(static_cast<int>(px), static_cast<int>(py)) = c;
        }
      }
    }
  }
}

template <typename T, int DEPTH, typename P, typename S> inline
void DrawTextFormat(MCImage<T, DEPTH>& img,
                    P x, P y, const S& c, 
                    const char *fmt, ...) {
  _STRFMT(str,fmt);
  DrawText(img, x, y, c, str);
}

template <typename T, int DEPTH, typename P, typename S> inline
void DrawLine(MCImage<T, DEPTH>& img, P x0, P y0, P x1, P y1,
              const S& c, int dd = 1) {
  const int w = img.width(), h = img.height();
  register P xf = x0, yf = y0, xt = x1, yt = y1;
  register P dx = (x0 < x1) ? x1 - x0 : x0 - x1;
  register P dy = (y0 < y1) ? y1 - y0 : y0 - y1;
  register P x, y, to;
  if (dy != 0 && dx < dy) {
    if (y0 < y1) yf = y0, yt = y1, xf = x0, xt = x1, dx = (x1 - x0);
    else         yf = y1, yt = y0, xf = x1, xt = x0, dx = (x0 - x1);
    for (y = (yf < 0 ? 0 : yf), to = (yt >= h ? h - 1 : yt); y <= to; y += dd) {
      x = xf + (y - yf) * dx / dy;
      if (0 <= x && x < w) img(static_cast<int>(x), static_cast<int>(y)) = c;
    }
  } else if (dx != 0) {
    if (x0 < x1) xf = x0, xt = x1, yf = y0, yt = y1, dy = (y1 - y0);
    else         xf = x1, xt = x0, yf = y1, yt = y0, dy = (y0 - y1);
    for (x = (xf < 0 ? 0 : xf), to = (xt >= w ? w - 1 : xt); x <= to; x += dd) {
      y = yf + (x - xf) * dy / dx;
      if (0 <= y && y < h) img(static_cast<int>(x), static_cast<int>(y)) = c;
    }
  }
}

}  // namespace rvslam
#endif  // _RVSLAM_IMAGE_UTIL_H_

