// image_file.h
//
// Author: Jongwoo Lim (jongwoo.lim@gmail.com)

#ifndef _RVSLAM_IMAGE_FILE_H_
#define _RVSLAM_IMAGE_FILE_H_

#include <stdarg.h>
#include <stdio.h>
#include <cstdlib>
#include <string>
#include "image.h"

//namespace rvslam {
namespace opticalflow {

inline std::string StringPrintf(const std::string fmt, ...) {
  char buf[1024];
  va_list a;
  va_start(a, fmt);
  vsprintf(buf, fmt.c_str(), a);
  va_end(a);
  return std::string(buf);
}

enum {
  PIXEL_NO_ADJUST, PIXEL_TYPE_ADJUST,
//  PIXEL_ABS_MAX_ADJUST, PIXEL_MIN_MAX_ADJUST
};

template <class T>
bool ReadImage(const std::string& filepath, Image<T>* image,
               int pixel_adjust = PIXEL_NO_ADJUST);

template <class T, int D>
bool ReadImage(const std::string& filepath, MCImage<T, D>* image,
               int pixel_adjust = PIXEL_NO_ADJUST);

template <class T>
bool WriteImage(const Image<T>& image, const std::string& filepath,
                int pixel_adjust = PIXEL_NO_ADJUST);

template <class T, int D>
bool WriteImage(const MCImage<T, D>& image, const std::string& filepath,
                int pixel_adjust = PIXEL_NO_ADJUST);

template <typename T, typename S>
Image<T> ConvertImageTo(const S& src, int pixel_adjust = PIXEL_TYPE_ADJUST);

template <typename T, int D, typename S>
MCImage<T, D> ConvertImageToMC(const S& src,
                               int pixel_adjust = PIXEL_TYPE_ADJUST);

}  // namespace rvslam
#endif  // _RVSLAM_IMAGE_FILE_H_
