// image_file.cc
//
// Author: Jongwoo Lim (jongwoo.lim@gmail.com)

#include "image_file.h"

#include <stdio.h>
#include <algorithm>
#include <cctype>
#include <vector>

#include <png.h>
#include <glog/logging.h>
#include <typeinfo>

using namespace std;
using Eigen::Dynamic;

//namespace rvslam {
namespace opticalflow {

namespace {

enum { GRAY8, GRAY16, RGB24, FLOAT, FLOAT3 };

struct RawImageBuffer {
  string data;
  int pixfmt, width, height, pitch;

  RawImageBuffer() : data(), pixfmt(GRAY8), width(0), height(0), pitch(0) {}
};

string GetFileExtension(const string& filepath) {
  size_t dot_pos = filepath.rfind(".");
  if (dot_pos == string::npos) return string();
  string ext = filepath.substr(dot_pos);
  if (!ext.empty()) ext = ext.substr(1);
  transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  return ext;
}

bool ReadRawImageBufferFromPNG(const string& filepath, RawImageBuffer* imgbuf) {
  // Open the image file and check if it's a valid png file.
  FILE* fp = fopen(filepath.c_str(), "rb");
  if (!fp) {
    LOG(ERROR) << "Failed to open file " << filepath;
    return false;
  }
  png_byte png_header[8];  // 8 is the maximum size that can be checked.
  fread(png_header, 1, 8, fp);
  if (png_sig_cmp(png_header, 0, 8)) {
    LOG(ERROR) << "Incorrect PNG header " << filepath;
    fclose(fp);
    return false;
  }
  // Initialize the png struct to load the content.
  png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING,
                                           NULL, NULL, NULL);
  if (!png) {
    LOG(ERROR) << "png_create_read_struct failed " << filepath;
    fclose(fp);
    return false;
  }
  png_infop info = png_create_info_struct(png);
  if (!info) {
    LOG(ERROR) << "png_create_info_struct failed " << filepath;
    png_destroy_read_struct(&png, NULL, NULL);
    fclose(fp);
    return false;
  }
  if (setjmp(png_jmpbuf(png))) {
    LOG(ERROR) << "Error during setting up png read " << filepath;
    png_destroy_read_struct(&png, &info, NULL);
    fclose(fp);
    return false;
  }
  png_init_io(png, fp);
  png_set_sig_bytes(png, 8);
  png_read_info(png, info);

  const png_byte ctype = png_get_color_type(png, info);
  const png_byte bit_depth = png_get_bit_depth(png, info);
  if (ctype == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
  if ((ctype & PNG_COLOR_MASK_ALPHA)) png_set_strip_alpha(png);
  if (bit_depth < 8)  png_set_packing(png);

  imgbuf->pixfmt = GRAY8;
  if (ctype == PNG_COLOR_TYPE_RGB || ctype == PNG_COLOR_TYPE_RGB_ALPHA) {
    imgbuf->pixfmt = RGB24;
    if (bit_depth > 8)  png_set_strip_16(png);
  } else if (bit_depth > 8) {
    imgbuf->pixfmt = GRAY16;
    png_set_swap(png);  // LSB -> MSB
  } else if (ctype == PNG_COLOR_TYPE_PALETTE) {
    LOG(ERROR) << "palette ctype not supported.";
  }
  // png_set_interlace_handling(png);
  // png_read_update_info(png, info);

  imgbuf->width = png_get_image_width(png, info);
  imgbuf->height = png_get_image_height(png, info);
  imgbuf->pitch = imgbuf->width * (imgbuf->pixfmt == RGB24 ? 3 :
                                   imgbuf->pixfmt == GRAY16 ? 2 : 1);
  VLOG(2) << imgbuf->width << "x" << imgbuf->height << ", " << imgbuf->pitch;
  VLOG(2) << "ctype: " << (int) ctype << ", bit_depth: " << (int) bit_depth;
  VLOG(2) << "ctype: " << (int) PNG_COLOR_TYPE_RGB;
  VLOG(2) << "ctype: " << (int) PNG_COLOR_TYPE_RGB_ALPHA;
  imgbuf->data.resize(imgbuf->pitch * imgbuf->height);
  vector<png_bytep> rows(imgbuf->height);
  for (int i = 0; i < imgbuf->height; ++i) {
    rows[i] = (png_bytep) &imgbuf->data[i * imgbuf->pitch];
  }
  // Read image from the file.
  if (setjmp(png_jmpbuf(png))) {
    LOG(ERROR) << "Error during reading a png image " << filepath;
    png_destroy_read_struct(&png, &info, NULL);
    fclose(fp);
    return false;
  }
  png_read_image(png, &rows[0]);

  png_destroy_read_struct(&png, &info, NULL);
  fclose(fp);
  return true;
}

//----------------------------------------------------------------------------

bool WriteRawImageBufferToPNG(const void* data, int pixfmt,
                              int width, int height, int pitch,
                              const string& filepath) {
  if (width <= 0 || height <= 0) {
    LOG(ERROR) << "Empty image " << filepath;
    return false;
  }
  FILE *fp = fopen(filepath.c_str(), "wb");
  if (!fp) {
    LOG(ERROR) << "Failed to open " << filepath;
    return false;
  }
  // Initialize png structs.
  png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING,
                                            NULL, NULL, NULL);
  if (!png) {
    LOG(ERROR) << "png_create_write_struct failed " << filepath;
    fclose(fp);
    return false;
  }
  png_infop info = png_create_info_struct(png);
  if (!info) {
    LOG(ERROR) << "png_create_info_struct failed " << filepath;
    png_destroy_write_struct(&png, NULL);
    fclose(fp);
    return false;
  }
  if (setjmp(png_jmpbuf(png))) {
    LOG(ERROR) << "Error during setting up png write " << filepath;
    png_destroy_write_struct(&png, &info);
    fclose(fp);
    return false;
  }
  png_init_io(png, fp);

  // Write png header.
  if (setjmp(png_jmpbuf(png))) {
    LOG(ERROR) << "Error during writting png header " << filepath;
    png_destroy_write_struct(&png, &info);
    fclose(fp);
    return false;
  }
  png_set_compression_level(png, 3);
  png_set_compression_mem_level(png, 9);
  png_set_IHDR(
      png, info, width, height,
      pixfmt == GRAY16 ? 16 : 8,
      pixfmt == RGB24 ? PNG_COLOR_TYPE_RGB : PNG_COLOR_TYPE_GRAY,
      PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
      PNG_FILTER_TYPE_DEFAULT);
  png_write_info(png, info);
  if (pixfmt == GRAY16) png_set_swap(png);

  // Write image content.
  vector<png_bytep> rows(height);
  for (int i = 0; i < height; ++i) {
    rows[i] = (png_bytep) (((uint8_t*) data) + i * pitch);
  }
  if (setjmp(png_jmpbuf(png))) {
    LOG(ERROR) << "Error during writting png image " << filepath;
    png_destroy_write_struct(&png, &info);
    fclose(fp);
    return false;
  }
  png_write_image(png, &rows[0]);
  png_write_end(png, NULL);

  fclose(fp);
  png_destroy_write_struct(&png, &info);
  return true;
}

//-----------------------------------------------------------------------------
// Utility functions.

inline bool ReadRawImageBufferFromFile(const string& filepath,
                                       RawImageBuffer* imgbuf) {
  string ext = GetFileExtension(filepath);
  bool ret = false;
  if (ext == "png") {
    ret = ReadRawImageBufferFromPNG(filepath, imgbuf);
    VLOG(1) << "ReadImageGray8 " << imgbuf->pixfmt << ","
        << imgbuf->width << "x" << imgbuf->height << ", " << imgbuf->pitch;;
  }
  return ret;
}

inline bool WriteRawImageBufferToFile(const void* data, int pixfmt,
                                      int width, int height, int pitch,
                                      const string& filepath) {
  string ext = GetFileExtension(filepath);
  bool ret = false;
  if (ext == "png") {
    ret = WriteRawImageBufferToPNG(data, pixfmt, width, height, pitch,
                                   filepath);
    VLOG(1) << "WriteImageGray8 '" << filepath << "', "
        << width << "x" << height;
  } else {
    LOG(ERROR) << "unsupported image type " << ext;
  }
  return ret;
}

template <typename S, typename T> inline
void CopyBuf(const RawImageBuffer& src, int num, RawImageBuffer* dst) {
  const S *srcbuf = reinterpret_cast<const S*>(&src.data[0]);
  T *dstbuf = reinterpret_cast<T*>(&dst->data[0]);
  for (int i = 0; i < num; ++i) dstbuf[i] = srcbuf[i];
}

template <typename S, typename T> inline
void CopyBufUp3(const RawImageBuffer& src, int num, RawImageBuffer* dst) {
  const S *srcbuf = reinterpret_cast<const S*>(&src.data[0]);
  T *dstbuf = reinterpret_cast<T*>(&dst->data[0]);
  for (int i = 0, k = 0; i < num; ++i, k += 3) {
    dstbuf[k] = dstbuf[k + 1] = dstbuf[k + 2] = srcbuf[i];
  }
}

template <typename S, typename T> inline
void CopyBufDown3(const RawImageBuffer& src, int num, RawImageBuffer* dst) {
  const S *srcbuf = reinterpret_cast<const S*>(&src.data[0]);
  T *dstbuf = reinterpret_cast<T*>(&dst->data[0]);
  for (int i = 0, k = 0; i < num; ++i, k += 3) {
    dstbuf[i] = ((int(srcbuf[k]) + srcbuf[k + 1]) + srcbuf[k + 2]) / T(3);
  }
}

template <typename S, typename T> inline
void CopyBufAdj(const RawImageBuffer& src, int num, RawImageBuffer* dst) {
  const S *srcbuf = reinterpret_cast<const S*>(&src.data[0]);
  T *dstbuf = reinterpret_cast<T*>(&dst->data[0]);
  if (typeid(T) == typeid(float)) {
    if (typeid(S) == typeid(uint8_t)) {
      for (int i = 0; i < num; ++i) dstbuf[i] = srcbuf[i] / 255.f;
    } else if (typeid(S) == typeid(uint16_t)) {
      for (int i = 0; i < num; ++i) dstbuf[i] = srcbuf[i] / 65535.f;
    }
  } else if (typeid(S) == typeid(float)) {
    if (typeid(T) == typeid(uint8_t)) {
      for (int i = 0; i < num; ++i) dstbuf[i] = srcbuf[i] * 255.f;
    } else if (typeid(T) == typeid(uint16_t)) {
      for (int i = 0; i < num; ++i) dstbuf[i] = srcbuf[i] * 65535.f;
    }
  } else if (typeid(S) == typeid(uint8_t) && typeid(T) == typeid(uint16_t)) {
    for (int i = 0; i < num; ++i) dstbuf[i] = srcbuf[i] * 256;
  } else if (typeid(S) == typeid(uint16_t) && typeid(T) == typeid(uint8_t)) {
    for (int i = 0; i < num; ++i) dstbuf[i] = srcbuf[i] / 256;
  }
}

// Converts one image buffer to another pixel format.
// If src.pixfmt == dst_pixfmt, it returns the pointer to src.
// If no conversion is possible, it returns NULL.
//
const RawImageBuffer* ConvertImageBuffer(const RawImageBuffer& src,
                                         int dst_pixfmt, bool pixel_type_adjust,
                                         RawImageBuffer* dst) {
  if (src.pixfmt == dst_pixfmt) return &src;

  dst->pixfmt = dst_pixfmt;
  dst->width = src.width;
  dst->height = src.height;
  if (dst_pixfmt == GRAY8) dst->pitch = dst->width;
  else if (dst_pixfmt == GRAY16) dst->pitch = 2 * dst->width;
  else if (dst_pixfmt == RGB24) dst->pitch = 3 * dst->width;
  else if (dst_pixfmt == FLOAT) dst->pitch = sizeof(float) * dst->width;
  else if (dst_pixfmt == FLOAT3) dst->pitch = 3 * sizeof(float) * dst->width;
  else return NULL;

  RawImageBuffer tmp;
  dst->data.resize(dst->pitch * dst->height);
  const int n = src.width * src.height;
  if (src.pixfmt == GRAY8) {
    if (dst_pixfmt == GRAY16) {
      if (pixel_type_adjust) CopyBufAdj<uint8_t, uint16_t>(src, n, dst);
      else CopyBuf<uint8_t, uint16_t>(src, n, dst);
    } else if (dst_pixfmt == RGB24) {
      CopyBufUp3<uint8_t, uint8_t>(src, n, dst);
    } else if (dst_pixfmt == FLOAT) {
      if (pixel_type_adjust) CopyBufAdj<uint8_t, float>(src, n, dst);
      else CopyBuf<uint8_t, float>(src, n, dst);
    } else if (dst_pixfmt == FLOAT3) {
      if (pixel_type_adjust) {
        tmp.data.resize(n * sizeof(float));
        CopyBufAdj<uint8_t, float>(src, n, &tmp);
        CopyBufUp3<float, float>(tmp, n, dst);
      } else {
        CopyBufUp3<uint8_t, float>(src, n, dst);
      }
    } else {
      return NULL;
    }
  } else if (src.pixfmt == GRAY16) {
    if (dst_pixfmt == GRAY8) {
      if (pixel_type_adjust) CopyBufAdj<uint16_t, uint8_t>(src, n, dst);
      else CopyBuf<uint16_t, uint8_t>(src, n, dst);
    } else if (dst_pixfmt == RGB24) {
      tmp.data.resize(n);
      if (pixel_type_adjust) CopyBufAdj<uint16_t, uint8_t>(src, n, &tmp);
      else CopyBuf<uint16_t, uint8_t>(src, n, &tmp);
      CopyBufUp3<uint8_t, uint8_t>(tmp, n, dst);
    } else if (dst_pixfmt == FLOAT) {
      if (pixel_type_adjust) CopyBufAdj<uint16_t, float>(src, n, dst);
      else CopyBuf<uint16_t, float>(src, n, dst);
    } else if (dst_pixfmt == FLOAT3) {
      if (pixel_type_adjust) {
        tmp.data.resize(n * sizeof(float));
        CopyBufAdj<uint16_t, float>(src, n, &tmp);
        CopyBufUp3<float, float>(tmp, n, dst);
      } else {
        CopyBufUp3<uint16_t, float>(src, n, dst);
      }
    } else {
      return NULL;
    }
  } else if (src.pixfmt == RGB24) {
    if (dst_pixfmt == GRAY8) {
      CopyBufDown3<uint8_t, uint8_t>(src, n, dst);
    } else if (dst_pixfmt == GRAY16) {
      tmp.data.resize(n);
      CopyBufDown3<uint8_t, uint8_t>(src, n, &tmp);  // Suboptimal.
      if (pixel_type_adjust) CopyBufAdj<uint8_t, uint16_t>(tmp, n, dst);
      else CopyBuf<uint8_t, uint16_t>(tmp, n, dst);
    } else if (dst_pixfmt == FLOAT) {
      if (pixel_type_adjust) {
        tmp.data.resize(n * sizeof(float));
        CopyBufDown3<uint8_t, float>(src, n, &tmp);
        CopyBufAdj<float, float>(tmp, n, dst);
      } else {
        CopyBufDown3<uint8_t, float>(src, n, dst);
      }
    } else if (dst_pixfmt == FLOAT3) {
      if (pixel_type_adjust) CopyBufAdj<uint8_t, float>(src, n * 3, dst);
      else CopyBuf<uint8_t, float>(src, n * 3, dst);
    } else {
      return NULL;
    }
  }
/*
  switch (src.pixfmt) {
    case GRAY8: {
      const uint8_t *srcbuf = reinterpret_cast<const uint8_t*>(&src.data[0]);
      if (dst_pixfmt == GRAY16) {
        uint16_t *dstbuf = reinterpret_cast<uint16_t*>(&dst->data[0]);
        if (pixel_type_adjust) {
          for (int i = 0; i < num_pixel; ++i) dstbuf[i] = srcbuf[i] * 256;
        } else {
          for (int i = 0; i < num_pixel; ++i) dstbuf[i] = srcbuf[i];
        }
      } else if (dst_pixfmt == RGB24) {
        uint8_t *dstbuf = reinterpret_cast<uint8_t*>(&dst->data[0]);
        for (int i = 0, k = 0; i < num_pixel; ++i, k += 3) {
          dstbuf[k] = dstbuf[k + 1] = dstbuf[k + 2] = srcbuf[i];
        }
      } else if (dst_pixfmt == FLOAT) {
        float *dstbuf = reinterpret_cast<float*>(&dst->data[0]);
        if (pixel_type_adjust) {
          for (int i = 0; i < num_pixel; ++i) dstbuf[i] = srcbuf[i] / 255.f;
        } else {
          for (int i = 0; i < num_pixel; ++i) dstbuf[i] = srcbuf[i];
        }
      } else if (dst_pixfmt == FLOAT3) {
        float *dstbuf = reinterpret_cast<float*>(&dst->data[0]);
        if (pixel_type_adjust) {
          for (int i = 0, k = 0; i < num_pixel; ++i, k += 3) {
            dstbuf[k] = dstbuf[k + 1] = dstbuf[k + 2] = srcbuf[i] / 255.f;
          }
        } else {
          for (int i = 0, k = 0; i < num_pixel; ++i, k += 3) {
            dstbuf[k] = dstbuf[k + 1] = dstbuf[k + 2] = srcbuf[i];
          }
        }
      }
      break;
    }
    case GRAY16: {
      const uint16_t *srcbuf = reinterpret_cast<const uint16_t*>(&src.data[0]);
      uint8_t *dstbuf = reinterpret_cast<uint8_t*>(&dst->data[0]);
      if (dst_pixfmt == GRAY8) {
        if (pixel_type_adjust) {
          for (int i = 0; i < num_pixel; ++i) dstbuf[i] = srcbuf[i] / 256;
        } else {
          for (int i = 0; i < num_pixel; ++i) dstbuf[i] = srcbuf[i];
        }
      } else if (dst_pixfmt == RGB24) {
        if (pixel_type_adjust) {
          for (int i = 0, k = 0; i < num_pixel; ++i, k += 3) {
            dstbuf[k] = dstbuf[k + 1] = dstbuf[k + 2] = srcbuf[i] / 256;
          }
        } else {
          for (int i = 0, k = 0; i < num_pixel; ++i, k += 3) {
            dstbuf[k] = dstbuf[k + 1] = dstbuf[k + 2] = srcbuf[i];
          }
        }
      } else if (dst_pixfmt == FLOAT) {
        float *dstbuf = reinterpret_cast<float*>(&dst->data[0]);
        if (pixel_type_adjust) {
          for (int i = 0; i < num_pixel; ++i) dstbuf[i] = srcbuf[i] / 65535.f;
        } else {
          for (int i = 0; i < num_pixel; ++i) dstbuf[i] = srcbuf[i];
        }
      } else if (dst_pixfmt == FLOAT3) {
        float *dstbuf = reinterpret_cast<float*>(&dst->data[0]);
        if (pixel_type_adjust) {
          for (int i = 0, k = 0; i < num_pixel; ++i, k += 3) {
            dstbuf[k] = dstbuf[k + 1] = dstbuf[k + 2] = srcbuf[i] / 65535.f;
          }
        } else {
          for (int i = 0, k = 0; i < num_pixel; ++i, k += 3) {
            dstbuf[k] = dstbuf[k + 1] = dstbuf[k + 2] = srcbuf[i];
          }
        }
      }
      break;
    }
    case RGB24: {
      const uint8_t *srcbuf = reinterpret_cast<const uint8_t*>(&src.data[0]);
      if (dst_pixfmt == GRAY8) {
        uint8_t *dstbuf = reinterpret_cast<uint8_t*>(&dst->data[0]);
        for (int i = 0, j = 0; i < num_pixel; ++i, j += 3) {
          int sum = int(srcbuf[j]) + int(srcbuf[j + 1]) + int(srcbuf[j + 2]);
          dstbuf[i] = sum / 3;
        }
      } else if (dst_pixfmt == GRAY16) {
        uint16_t *dstbuf = reinterpret_cast<uint16_t*>(&dst->data[0]);
        if (pixel_type_adjust) {
          for (int i = 0, j = 0; i < num_pixel; ++i, j += 3) {
            int sum = int(srcbuf[j]) + int(srcbuf[j + 1]) + int(srcbuf[j + 2]);
            dstbuf[i] = sum / (256 * 3);
          }
        } else {
          for (int i = 0, j = 0; i < num_pixel; ++i, j += 3) {
            int sum = int(srcbuf[j]) + int(srcbuf[j + 1]) + int(srcbuf[j + 2]);
            dstbuf[i] = sum / 3;
          }
        }
      } else if (dst_pixfmt == FLOAT) {
        float *dstbuf = reinterpret_cast<float*>(&dst->data[0]);
        if (pixel_type_adjust) {
          for (int i = 0, j = 0; i < num_pixel; ++i, j += 3) {
            int sum = int(srcbuf[j]) + int(srcbuf[j + 1]) + int(srcbuf[j + 2]);
            dstbuf[i] = sum / 3;
          }
          for (int i = 0; i < num_pixel; ++i) dstbuf[i] = srcbuf[i] / 255.f;
          dstbuf[i] = kj
          int sum = int(srcbuf[j]) + int(srcbuf[j + 1]) + int(srcbuf[j + 2]);
        } else {
          for (int i = 0; i < num_pixel; ++i) dstbuf[i] = srcbuf[i];
        }
      } else if (dst_pixfmt == FLOAT3) {
        float *dstbuf = reinterpret_cast<float*>(&dst->data[0]);
        if (pixel_type_adjust) {
          for (int i = 0, k = 0; i < num_pixel; ++i, k += 3) {
            dstbuf[k] = dstbuf[k + 1] = dstbuf[k + 2] = srcbuf[i] / 255.f;
          }
        } else {
          for (int i = 0, k = 0; i < num_pixel; ++i, k += 3) {
            dstbuf[k] = dstbuf[k + 1] = dstbuf[k + 2] = srcbuf[i];
          }
        }
      }
      break;
    }
    default: return NULL;
  }
*/
  return dst;
}

}  // namespace

//-----------------------------------------------------------------------------
// ReadImage functions for various image types.

template <class T, int PIXFMT> inline
bool ReadImageTmpl(const string& filepath, bool typeadj, Image<T>* image) {
  RawImageBuffer imgbuf, tmp;
  if (!ReadRawImageBufferFromFile(filepath, &imgbuf)) return false;
  const RawImageBuffer* ptr = ConvertImageBuffer(imgbuf, PIXFMT, typeadj, &tmp);
  if (ptr == NULL) return false;
  image->FromBuffer(&ptr->data[0], ptr->width, ptr->height);
  return true;
}

template <class T, int D, int PIXFMT, int DEPTH> inline
bool ReadMCImageTmpl(const string& filepath, bool typeadj,
                     MCImage<T, D>* image) {
  RawImageBuffer imgbuf, tmp;
  if (!ReadRawImageBufferFromFile(filepath, &imgbuf)) return false;
  const RawImageBuffer* ptr = ConvertImageBuffer(imgbuf, PIXFMT, typeadj, &tmp);
  if (ptr == NULL) return false;
  image->FromBuffer(&ptr->data[0], ptr->width, ptr->height, DEPTH);
  return true;
}

template <class T, int D> inline
bool ReadMCImageFloatTmpl(const string& filepath, bool typeadj,
                          MCImage<T, D>* image) {
  RawImageBuffer rib;
  if (!ReadRawImageBufferFromFile(filepath, &rib)) return false;
  const int depth = rib.pixfmt == RGB24 ? 3 : 1;
  image->Resize(rib.width, rib.height, depth);
  const int num_pixel = rib.width * rib.height * depth;
  T* dstbuf = image->data();
  if (rib.pixfmt == GRAY8 || rib.pixfmt == RGB24) {
    const uint8_t* srcbuf = reinterpret_cast<const uint8_t*>(&rib.data[0]);
    if (typeadj) {
      for (int i = 0; i < num_pixel; ++i) dstbuf[i] = srcbuf[i] / 255.f;
    } else {
      for (int i = 0; i < num_pixel; ++i) dstbuf[i] = srcbuf[i];
    }
  } else if (rib.pixfmt == GRAY16) {
    const uint16_t* srcbuf = reinterpret_cast<const uint16_t*>(&rib.data[0]);
    if (typeadj) {
      for (int i = 0; i < num_pixel; ++i) dstbuf[i] = srcbuf[i] / 65535.f;
    } else {
      for (int i = 0; i < num_pixel; ++i) dstbuf[i] = srcbuf[i];
    }
  } else {
    return false;
  }
  return true;
}

template <>
bool ReadImage<uint8_t>(const string& filepath, Image<uint8_t>* image,
                        int pixel_adjust) {
  bool type_adj = (pixel_adjust == PIXEL_TYPE_ADJUST);
  return ReadImageTmpl<uint8_t, GRAY8>(filepath, type_adj, image);
}

template <>
bool ReadImage<uint16_t>(const string& filepath, Image<uint16_t>* image,
                         int pixel_adjust) {
  bool type_adj = (pixel_adjust == PIXEL_TYPE_ADJUST);
  return ReadImageTmpl<uint16_t, GRAY16>(filepath, type_adj, image);
}

template <>
bool ReadImage<uint8_t, 1>(const string& filepath, MCImage<uint8_t, 1>* image,
                           int pixel_adjust) {
  bool type_adj = (pixel_adjust == PIXEL_TYPE_ADJUST);
  return ReadMCImageTmpl<uint8_t, 1, GRAY8, 1>(filepath, type_adj, image);
}

template <>
bool ReadImage<uint16_t, 1>(const string& filepath, MCImage<uint16_t, 1>* image,
                            int pixel_adjust) {
  bool type_adj = (pixel_adjust == PIXEL_TYPE_ADJUST);
  return ReadMCImageTmpl<uint16_t, 1, GRAY16, 1>(filepath, type_adj, image);
}

template <>
bool ReadImage<uint8_t, 3>(const string& filepath, MCImage<uint8_t, 3>* image,
                           int pixel_adjust) {
  bool type_adj = (pixel_adjust == PIXEL_TYPE_ADJUST);
  return ReadMCImageTmpl<uint8_t, 3, RGB24, 3>(filepath, type_adj, image);
}

template <>
bool ReadImage<uint8_t, Dynamic>(const string& path,
                                 MCImage<uint8_t, Dynamic>* image,
                                 int pixel_adjust) {
  bool type_adj = (pixel_adjust == PIXEL_TYPE_ADJUST);
  if (image->depth() == 1) {
    return ReadMCImageTmpl<uint8_t, Dynamic, GRAY8, 1>(path, type_adj, image);
  }
  if (image->depth() == 3) {
    return ReadMCImageTmpl<uint8_t, Dynamic, RGB24, 3>(path, type_adj, image);
  }
  LOG(ERROR) << "invalid depth(" << image->depth() << ").";
  return false;
}

template <>
bool ReadImage<uint16_t, Dynamic>(const string& path,
                                  MCImage<uint16_t, Dynamic>* image,
                                  int pixel_adjust) {
  bool type_adj = (pixel_adjust == PIXEL_TYPE_ADJUST);
  if (image->depth() == 1) {
    return ReadMCImageTmpl<uint16_t, Dynamic, GRAY16, 1>(path, type_adj, image);
  }
  LOG(ERROR) << "invalid depth(" << image->depth() << ").";
  return false;
}

template <>
bool ReadImage<float>(const string& path, Image<float>* img, int pixel_adj) {
  bool type_adj = (pixel_adj == PIXEL_TYPE_ADJUST);
  return ReadImageTmpl<float, FLOAT>(path, type_adj, img);
}

template <>
bool ReadImage<float, 1>(const string& path, MCImage<float, 1>* img, int padj) {
  bool type_adj = (padj == PIXEL_TYPE_ADJUST);
  return ReadMCImageFloatTmpl(path, type_adj, img);
}

template <>
bool ReadImage<float, 3>(const string& path, MCImage<float, 3>* img, int padj) {
  bool type_adj = (padj == PIXEL_TYPE_ADJUST);
  return ReadMCImageFloatTmpl(path, type_adj, img);
}

template <>
bool ReadImage<float, Dynamic>(const string& path, MCImage<float, Dynamic>* img,
                               int pixel_adjust) {
  bool type_adj = (pixel_adjust == PIXEL_TYPE_ADJUST);
  return ReadMCImageFloatTmpl(path, type_adj, img);
}

//-----------------------------------------------------------------------------
// WriteImage functions for various image types.

template <class T, int PIXFMT, int PIXEL_SIZE> inline
bool WriteImageTmpl(const Image<T>& image, const string& filepath) {
  return WriteRawImageBufferToFile(
      image.data(), PIXFMT, image.width(), image.height(),
      PIXEL_SIZE * image.width(), filepath);
}

template <class T, int D, int PIXFMT, int PIXEL_SIZE> inline
bool WriteMCImageTmpl(const MCImage<T, D>& image, const string& filepath) {
  return WriteRawImageBufferToFile(
      image.data(), PIXFMT, image.width(), image.height(),
      PIXEL_SIZE * image.width(), filepath);
}

template <>
bool WriteImage<uint8_t>(const Image<uint8_t>& image, const string& filepath,
                         int pixel_adjust) {
  return WriteImageTmpl<uint8_t, GRAY8, 1>(image, filepath);
}

template <>
bool WriteImage<uint16_t>(const Image<uint16_t>& image, const string& filepath,
                          int pixel_adjust) {
  return WriteImageTmpl<uint16_t, GRAY16, 2>(image, filepath);
}

template <>
bool WriteImage<uint8_t, 1>(const MCImage<uint8_t, 1>& image,
                            const string& filepath, int pixel_adjust) {
  return WriteMCImageTmpl<uint8_t, 1, GRAY8, 1>(image, filepath);
}

template <>
bool WriteImage<uint16_t, 1>(const MCImage<uint16_t, 1>& image,
                            const string& filepath, int pixel_adjust) {
  return WriteMCImageTmpl<uint16_t, 1, GRAY16, 2>(image, filepath);
}

template <>
bool WriteImage<uint8_t, 3>(const MCImage<uint8_t, 3>& image,
                            const string& filepath, int pixel_adjust) {
  return WriteMCImageTmpl<uint8_t, 3, RGB24, 3>(image, filepath);
}

template <>
bool WriteImage<uint8_t, Dynamic>(
    const MCImage<uint8_t, Dynamic>& image,
    const string& filepath,
    int pixel_adjust) {
  if (image.depth() == 1) {
    return WriteMCImageTmpl<uint8_t, Dynamic, GRAY8, 1>(image, filepath);
  }
  if (image.depth() == 3) {
    return WriteMCImageTmpl<uint8_t, Dynamic, RGB24, 3>(image, filepath);
  }
  LOG(ERROR) << "invalid depth(" << image.depth() << ").";
  return false;
}

template <>
bool WriteImage<uint16_t, Dynamic>(
    const MCImage<uint16_t, Dynamic>& image,
    const string& filepath,
    int pixel_adjust) {
  if (image.depth() == 1) {
    return WriteMCImageTmpl<uint16_t, Dynamic, GRAY16, 2>(image, filepath);
  }
  LOG(ERROR) << "invalid depth(" << image.depth() << ").";
  return false;
}

//-----------------------------------------------------------------------------
// WriteImage functions for various image types.

template <typename T> inline
void ConvertFloatBuffer(const float* srcbuf, int num, bool adjust, T* dstbuf) {
  if (adjust) {  // Assumes the values are 0~1.
    for (int i = 0; i < num; ++i) {
      dstbuf[i] = static_cast<T>(srcbuf[i] * numeric_limits<T>::max());
    }
  } else {
    for (int i = 0; i < num; ++i) dstbuf[i] = static_cast<T>(srcbuf[i]);
  }
}

template <typename T> inline
Image<T> ConvertFloatImageTo(const float* srcbuf, int w, int h, bool adjust) {
  Image<T> ret(w, h);
  ConvertFloatBuffer<T>(srcbuf, w * h, adjust, ret.data());
  return ret;
}

template <typename T, int D> inline
MCImage<T, D> ConvertFloatImageToMC(const float* srcbuf, int w, int h, int d,
                                    bool adjust) {
  MCImage<T, D> ret(w, h, d);
  ConvertFloatBuffer<T>(srcbuf, w * h * d, adjust, ret.data());
  return ret;
}

template <>
Image<uint8_t> ConvertImageTo(const Image<float>& src, int pixel_adjust) {
  return ConvertFloatImageTo<uint8_t>(src.data(), src.rows(), src.cols(),
                                      pixel_adjust == PIXEL_TYPE_ADJUST);
}

template <>
Image<uint16_t> ConvertImageTo(const Image<float>& src, int pixel_adjust) {
  return ConvertFloatImageTo<uint16_t>(src.data(), src.rows(), src.cols(),
                                       pixel_adjust == PIXEL_TYPE_ADJUST);
}

template <>
MCImage<uint8_t, Dynamic> ConvertImageToMC(const MCImage<float, Dynamic>& src,
                                           int pixel_adjust) {
  return ConvertFloatImageToMC<uint8_t, Dynamic>(
      src.data(), src.width(), src.height(), src.depth(),
      pixel_adjust == PIXEL_TYPE_ADJUST);
}

}  // namespace rvslam

