// image_util_test.cc
//

#include <iostream>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "image_file.h"
//#include "image_type.h"
#include "image_util.h"

using namespace std;
using namespace rvslam;

//DEFINE_string(image, "test_gray8.png", "Test image path.");

#define EXPECT_ARRAY_NEAR(arr1, arr2, thr) \
  do { \
    EXPECT_EQ(arr1.cols(), arr2.cols()); \
    EXPECT_EQ(arr1.rows(), arr2.rows()); \
    for (int y = 0; y < arr1.cols(); ++y) { \
      for (int x = 0; x < arr1.rows(); ++x) { \
        EXPECT_NEAR(arr1.coeffRef(x, y), arr2.coeffRef(x, y), thr); \
      } \
    } \
  } while(0)

namespace {

typedef Eigen::ArrayXXf ArrayF;
typedef Eigen::ArrayXXd ArrayD;

void TestGaussianKernel(const ArrayF& kernel, const double sigma) {
  int rad = kernel.cols() / 2;
  for (int i = 1; i < rad; ++i) {
    const double coef = -1 / (2 * sigma * sigma);
    const double ratio = exp(coef * i * i);
    EXPECT_NEAR(ratio, kernel(0, rad - i) / kernel(0, rad), 1e-6);
    EXPECT_NEAR(ratio, kernel(0, rad + i) / kernel(0, rad), 1e-6);
  }
}

TEST(ImageUtilTest, FontMapDump) {
  Image<unsigned char> img_char;
  img_char.FromBuffer(_g_fontimgdata, 480, 8);

  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 480; ++j)
      img_char(j,i) = img_char(j,i) * 255;

  WriteImage(img_char, "font.png");
}

TEST(ImageUtilTest, FontHelloWorld) {
  MCImage<uint8_t, 3> img_rgb;
  img_rgb.Resize(100,100);
  img_rgb.fill(255);

  DrawText(img_rgb, 10, 20, MCImageRGB8::MakePixel(255, 0, 0), "Hello World!\0");
  DrawTextFormat(img_rgb, 10, 50, MCImageRGB8::MakePixel(0, 255, 0), "Frame %04d", 2);
  WriteImage(img_rgb, "font-test.png");
}

TEST(ImageUtilTest, TestBuildGaussianKernel) {
  ArrayF kernel(1, 9);
  double sigma = 1.0;
  BuildGaussianKernel(sigma, &kernel);
  TestGaussianKernel(kernel, sigma);
  sigma = 0.5;
  BuildGaussianKernel(sigma, &kernel);
  TestGaussianKernel(kernel, sigma);
  sigma = 1.5;
  BuildGaussianKernel(sigma, &kernel);
  TestGaussianKernel(kernel, sigma);
  kernel.resize(1, 13);
  BuildGaussianKernel(sigma, &kernel);
  TestGaussianKernel(kernel, sigma);
  sigma = 2.5;
  BuildGaussianKernel(sigma, &kernel);
  TestGaussianKernel(kernel, sigma);
}

TEST(ImageUtilTest, TestConvolve) {
  ArrayF kernel(5, 1);
  BuildGaussianKernel(0.5, &kernel);

  ArrayF imgf(10, 10), out(10, 10);
  imgf.fill(10.0);
  for (int y = 0; y < 5; ++y) {
    for (int x = 0; x < 5; ++x) {
      imgf(x, y + 5) = imgf(x + 5, y) = 0.0;
    }
  }
  Convolve(imgf, kernel, &out);
  ArrayF ref(10, 1);
  ref << 8.9329, 9.9974, 10.000, 9.9974, 8.9329, 1.0672, 0.0027, 0.0, 0.0, 0.0;
  for (int y = 0; y < imgf.cols() / 2; ++y) {
    for (int x = 0; x < imgf.rows(); ++x) {
      EXPECT_NEAR(ref(x), out(x, y), 1e-4);
      EXPECT_NEAR(ref(imgf.rows() - 1 - x), out(x, y + imgf.cols() / 2), 1e-4);
    }
  }
  ArrayF ref2d(10, 10), out2d(10, 10);
  Convolve(out, kernel.transpose(), &out2d);
  ref2d <<
 7.9796, 8.9305, 8.9329, 8.9305, 7.9796, 0.9533, 0.0024, 0.0000, 0.0000, 0.0000,
 8.9305, 9.9947, 9.9974, 9.9947, 8.9305, 1.0669, 0.0026, 0.0000, 0.0000, 0.0000,
 8.9329, 9.9974, 10.000, 9.9974, 8.9329, 1.0672, 0.0026, 0.0000, 0.0000, 0.0000,
 8.9305, 9.9947, 9.9974, 9.9947, 8.9308, 1.0692, 0.0053, 0.0026, 0.0026, 0.0024,
 7.9796, 8.9305, 8.9329, 8.9308, 8.0935, 1.9065, 1.0692, 1.0672, 1.0669, 0.9533,
 0.9533, 1.0669, 1.0672, 1.0692, 1.9065, 8.0935, 8.9308, 8.9329, 8.9305, 7.9796,
 0.0024, 0.0026, 0.0026, 0.0053, 1.0692, 8.9308, 9.9947, 9.9974, 9.9947, 8.9305,
 0.0000, 0.0000, 0.0000, 0.0026, 1.0672, 8.9329, 9.9974, 10.000, 9.9974, 8.9329,
 0.0000, 0.0000, 0.0000, 0.0026, 1.0669, 8.9305, 9.9947, 9.9974, 9.9947, 8.9305,
 0.0000, 0.0000, 0.0000, 0.0024, 0.9533, 7.9796, 8.9305, 8.9329, 8.9305, 7.9796;
  EXPECT_ARRAY_NEAR(ref2d, out2d, 1e-4);

  ArrayF derivative(3, 1), imgx(10, 10), imgy(10, 10);
  derivative << -1.0, 0, 1.0;
  Convolve(out2d, derivative, &imgx);
  Convolve(out2d, derivative.transpose(), &imgy);
  for (int y = 1; y < imgf.cols() - 1; ++y) {
    for (int x = 1; x < imgf.rows() - 1; ++x) {
      EXPECT_NEAR(out2d(x - 1, y) - out2d(x + 1, y), imgx(x, y), 1e-4);
      EXPECT_NEAR(out2d(x, y - 1) - out2d(x, y + 1), imgy(x, y), 1e-4);
    }
  }

  MCImage<float, 2> img_mcf(10, 10, 2), tmp_mcf, out_mcf, ref_mcf(10, 10, 2);
  img_mcf.SetAllPlanes(imgf);
  ref_mcf.SetAllPlanes(ref2d);
//  LOG(ERROR) << setfill(' ') << endl << img_mcf;
  ConvolveImage(img_mcf, kernel, &tmp_mcf);
  ConvolveImage(tmp_mcf, kernel.transpose(), &out_mcf);
//  LOG(ERROR) << setfill(' ') << endl << out_mcf;
//  LOG(ERROR) << setfill(' ') << endl << ref_mcf;
  EXPECT_ARRAY_NEAR(ref_mcf, out_mcf, 1e-4);
}

TEST(ImageUtilTest, TestResize) {
  ArrayD img(16, 12);
  img << 49,  62,  90,   9,  68,  49,  81,  52,  17,  37,  67,  16,
         44,  59,  98,  26,  40,  78,  58,  10,  39,  20,  54,  86,
         45,  21,  44,  80,  37,  72,  18,  82,  83,  49,  70,  64,
         31,  30,  11,   3,  99,  90,  24,  82,  80,  34,  67,  38,
         51,  47,  26,  93,   4,  89,  89,  72,   6,  95,  18,  19,
         51,  23,  41,  73,  89,  33,   3,  15,  40,  92,  13,  43,
         82,  84,  59,  49,  91,  70,  49,  66,  53,   5, 100,  48,
         79,  19,  26,  58,  80,  20,  17,  52,  42,  74,  17,  12,
         64,  23,  60,  24,  10,   3,  98,  97,  66,  27,   3,  59,
         38,  17,  71,  46,  26,  74,  71,  65,  63,  42,  56,  23,
         81,  23,  22,  96,  34,  50,  50,  80,  29,  55,  88,  38,
         53,  44,  12,  55,  68,  48,  47,  45,  43,  94,  67,  58,
         35,  31,  30,  52,  14,  90,   6,  43,   2,  42,  19,  25,
         94,  92,  32,  23,  72,  61,  68,  83,  98,  98,  37,  29,
         88,  43,  42,  49,  11,  62,   4,   8,  17,  30,  46,  62,
         55,  18,  51,  62,  65,  86,   7,  13,  11,  70,  98,  27;
  ArrayD img_80(13, 10), img_75(12, 9), img_70(12, 9), img_40(7, 5);
  // Matlab's imresize with method = bilinear and antialising = false.
  img_80 <<
 50.031, 72.641, 41.078, 57.828, 55.812, 66.359, 29.875, 32.984, 60.297, 24.750,
 44.422, 57.125, 58.062, 39.797, 71.656, 40.750, 48.562, 33.953, 62.219, 77.750,
 35.047, 25.406, 28.688, 70.266, 75.562, 44.344, 81.453, 44.812, 65.578, 47.750,
 48.047, 37.094, 60.141, 24.109, 88.094, 78.016, 37.000, 78.359, 23.781, 21.375,
 51.844, 35.359, 59.969, 86.844, 34.016, 13.484, 34.031, 76.188, 26.344, 43.625,
 78.219, 54.750, 50.219, 82.562, 49.469, 45.906, 53.328, 33.125, 64.578, 34.500,
 63.609, 31.156, 40.688, 36.312, 16.656, 72.312, 65.672, 46.172, 12.391, 41.375,
 38.312, 37.203, 53.141, 26.406, 66.281, 72.359, 65.484, 43.031, 46.641, 27.500,
 71.016, 23.797, 64.578, 44.828, 49.734, 59.375, 47.578, 56.234, 79.766, 40.500,
 45.359, 31.484, 40.703, 48.516, 59.734, 36.359, 33.859, 68.641, 48.578, 45.625,
 71.531, 54.922, 32.891, 48.203, 68.484, 53.469, 64.250, 75.125, 29.906, 27.500,
 83.797, 45.984, 43.875, 22.016, 55.641, 14.016, 23.469, 37.078, 46.500, 57.875,
 50.375, 30.375, 57.875, 64.625, 76.125, 9.250, 11.750, 62.625, 89.125, 27.000;
  img_75 <<
      50.389, 76.417, 25.083, 61.750, 65.500, 50.361, 22.917, 49.500, 33.861,
      43.750, 55.500, 56.000, 44.583, 56.500, 44.667, 56.583, 48.250, 72.833,
      32.528, 22.500, 15.944, 88.389, 55.000, 72.167, 73.167, 52.000, 46.528,
      49.667, 35.750, 79.472, 28.417, 77.167, 64.528, 25.472, 55.833, 22.028,
      64.333, 51.750, 59.167, 83.583, 38.750, 38.083, 46.833, 52.500, 47.333,
      71.222, 30.667, 52.333, 72.917, 25.333, 49.000, 46.944, 46.667, 20.139,
      53.389, 41.917, 33.361, 13.028, 54.167, 91.972, 59.500, 20.667, 46.139,
      52.917, 33.250, 66.917, 35.333, 61.250, 70.500, 46.417, 60.250, 37.417,
      54.806, 27.083, 53.806, 60.000, 47.917, 50.278, 48.472, 79.000, 57.306,
      44.222, 35.750, 44.361, 33.917, 50.750, 44.111, 23.556, 36.667, 25.056,
      87.083, 52.250, 36.167, 44.833, 48.750, 43.917, 58.583, 52.750, 44.833,
      54.111, 35.833, 58.111, 60.333, 44.250, 11.222, 20.556, 76.333, 42.250;
  img_70 <<
      50.806, 80.872, 16.168, 58.607, 74.582, 35.398, 30.862, 57.097, 31.000,
      42.485, 53.031, 59.097, 56.107, 35.276, 60.214, 44.781, 65.908, 71.857,
      32.168, 18.908, 15.342, 91.071, 33.020, 78.939, 46.148, 57.745, 36.643,
      47.571, 34.036, 80.393, 53.750, 47.071, 36.179, 78.393, 18.821, 31.000,
      79.755, 65.546, 53.582, 79.107, 47.260, 58.684, 19.969, 83.898, 47.643,
      62.240, 31.816, 46.510, 34.464, 43.643, 61.821, 55.791, 15.597, 28.786,
      38.153, 50.658, 39.949, 40.679, 75.500, 68.923, 44.112, 41.658, 30.714,
      64.821, 22.587, 83.934, 45.429, 49.372, 58.036, 56.638, 74.668, 42.286,
      40.189, 27.883, 51.658, 54.143, 24.526, 34.046, 51.158, 36.281, 36.786,
      92.485, 52.638, 27.913, 64.357, 63.260, 82.847, 92.944, 36.296, 31.357,
      62.714, 40.786, 54.250, 56.000, 10.393, 11.750, 42.286, 66.107, 44.500,
      47.071, 39.214, 62.214, 75.500, 12.643, 12.286, 57.357, 82.786, 27.000;
  img_40 << 56.125, 28.062, 65.500, 31.188, 65.688,
            34.688, 37.938, 52.625, 58.438, 38.625,
            70.125, 63.875, 43.312, 44.000, 54.625,
            30.500, 25.625, 73.625, 56.625, 41.562,
            44.062, 63.812, 47.938, 50.688, 57.812,
            82.938, 36.312, 54.312, 78.562, 37.750,
            27.250, 62.750, 26.750, 25.750, 44.750;

  ArrayD out;
  Resize(img, 0.8, &out);
  EXPECT_ARRAY_NEAR(img_80, out, 1e-3);
  Resize(img, 0.75, &out);
  EXPECT_ARRAY_NEAR(img_75, out, 1e-3);
  Resize(img, &out);
  EXPECT_ARRAY_NEAR(img_75, out, 1e-3);
  Resize(img, 0.7, &out);
  EXPECT_ARRAY_NEAR(img_70, out, 1e-3);
  Resize(img, 0.4, &out);
  EXPECT_ARRAY_NEAR(img_40, out, 1e-3);

  ArrayD img_s10(16, 12);
  img_s10 << 26.243, 39.971, 42.036, 35.034, 35.661, 40.677, 39.458, 31.624,
                 25.852, 28.044, 31.656, 24.225,
             31.893, 47.807, 52.793, 48.170, 49.331, 53.822, 50.520, 44.607,
                 41.413, 42.660, 46.898, 38.283,
             28.266, 39.279, 45.076, 49.259, 55.609, 58.502, 54.680, 55.576,
                 55.838, 52.939, 52.052, 40.422,
             26.406, 33.392, 37.338, 46.947, 58.455, 62.726, 59.475, 60.084,
                 59.289, 54.891, 47.877, 32.851,
             29.945, 37.198, 41.755, 51.726, 59.229, 60.584, 56.413, 52.980,
                 52.526, 52.113, 42.277, 26.207,
             36.539, 45.191, 49.716, 58.841, 62.134, 55.235, 46.705, 44.256,
                 47.195, 49.893, 42.742, 27.747,
             42.427, 50.442, 51.658, 58.458, 61.316, 52.344, 45.016, 46.076,
                 47.608, 47.409, 43.009, 29.420,
             40.201, 45.707, 45.903, 49.572, 49.872, 45.891, 48.853, 54.406,
                 51.949, 44.723, 36.674, 25.127,
             33.991, 39.929, 42.555, 42.318, 39.509, 44.104, 57.413, 63.929,
                 56.598, 44.278, 34.477, 24.486,
             31.364, 38.024, 43.471, 45.179, 42.962, 49.607, 61.317, 64.325,
                 56.980, 49.407, 42.922, 29.347,
             32.881, 37.574, 42.027, 49.266, 50.023, 51.920, 56.046, 56.494,
                 54.024, 55.762, 52.924, 35.197,
             33.203, 37.686, 38.937, 46.706, 51.098, 51.245, 49.124, 47.772,
                 49.464, 55.093, 51.920, 34.528,
             37.090, 43.076, 39.868, 42.598, 48.898, 51.651, 47.802, 46.271,
                 48.954, 51.244, 43.912, 28.329,
             45.374, 51.480, 43.819, 41.769, 47.238, 50.658, 46.765, 46.375,
                 50.775, 51.947, 43.559, 28.304,
             42.574, 47.787, 43.206, 42.310, 45.191, 44.563, 35.257, 31.483,
                 38.139, 46.654, 45.875, 31.731,
             26.756, 30.828, 32.240, 35.245, 37.341, 33.739, 21.517, 15.291,
                 21.794, 34.389, 38.168, 25.879;
  ArrayD img_d = img.cast<double>();
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> out_g;
  GaussianSmoothing(img_d, 1.0, &out_g);
  EXPECT_ARRAY_NEAR(img_s10, out_g, 1e-3);

  MCImage<double, Eigen::Dynamic> img_mcf(0, 0, 3), out_mcf, ref_mcf(0, 0, 3);
  img_mcf.SetAllPlanes(img.cast<double>());

  ResizeImage(img_mcf, 0.8, &out_mcf);
  ref_mcf.SetAllPlanes(img_80);
  EXPECT_ARRAY_NEAR(ref_mcf, out_mcf, 1e-3);

  ResizeImage(img_mcf, 0.75, &out_mcf);
  ref_mcf.SetAllPlanes(img_75);
  EXPECT_ARRAY_NEAR(ref_mcf, out_mcf, 1e-3);

  ResizeImage(img_mcf, &out_mcf);
  EXPECT_ARRAY_NEAR(ref_mcf, out_mcf, 1e-3);

  ResizeImage(img_mcf, 0.7, &out_mcf);
  ref_mcf.SetAllPlanes(img_70);
  EXPECT_ARRAY_NEAR(ref_mcf, out_mcf, 1e-3);

  ResizeImage(img_mcf, 0.4, &out_mcf);
  ref_mcf.SetAllPlanes(img_40);
  EXPECT_ARRAY_NEAR(ref_mcf, out_mcf, 1e-3);

  GaussianSmoothingImage(img_mcf, 1.0, &out_mcf);
  ref_mcf.SetAllPlanes(img_s10);
  EXPECT_ARRAY_NEAR(ref_mcf, out_mcf, 1e-3);
}

/*
uint8_t Interp2Ref(const Image8& image, double x, double y) {
  const int x0 = static_cast<int>(x), y0 = static_cast<int>(y);
  const int x1 = x0 + 1, y1 = y0 + 1;
  const double rx = x - x0, ry = y - y0;
  LOG(INFO) << "Interp2Ref: " << static_cast<int>(image(x0, y0))
      << ", " << static_cast<int>(image(x1, y0))
      << ", " << static_cast<int>(image(x0, y1))
      << ", " << static_cast<int>(image(x1, y1))
      << " - " << rx << "," << ry << " : "
      << (image(x0, y0) * (1 - rx) + image(x1, y0) * rx) * (1 - ry) +
         (image(x0, y1) * (1 - rx) + image(x1, y1) * rx) * ry;
  return static_cast<uint8_t>(
      (image(x0, y0) * (1 - rx) + image(x1, y0) * rx) * (1 - ry) +
      (image(x0, y1) * (1 - rx) + image(x1, y1) * rx) * ry);
}

TEST(ImageUtilTest, TestInterp2) {
  Image8 image8(10, 10);
  for (int r = 0; r < image8.rows(); ++r) {
    for (int c = 0; c < image8.cols(); ++c) {
      image8(r, c) = r + c * image8.rows();
    }
  }
  EXPECT_EQ(Interp2Ref(image8, 0.0, 0.0), Interp2(image8, 0.0, 0.0));
  EXPECT_EQ(Interp2Ref(image8, 5.0, 0.5), Interp2(image8, 5.0, 0.5));
  EXPECT_EQ(Interp2Ref(image8, 0.0, 3.5), Interp2(image8, 0.0, 3.5));
  EXPECT_EQ(Interp2Ref(image8, 3.5, 7.5), Interp2(image8, 3.5, 7.5));
  EXPECT_EQ(Interp2Ref(image8, 5.2, 4.5), Interp2(image8, 5.2, 4.5));
  EXPECT_EQ(Interp2Ref(image8, 8.7, 3.2), Interp2(image8, 8.7, 3.2));
}

TEST(ImageUtilTest, TestResample) {
  Image8 image8;
  EXPECT_TRUE(ReadImage<uint8_t>(FLAGS_image, &image8));
  const int rows = image8.rows(), cols = image8.cols();
  LOG(INFO) << "image8 " << rows << "x" << cols;

  const int resize_rows = rows * 0.7, resize_cols = cols * 0.7;
  Image8 resized(resize_rows, resize_cols);
  Resample(image8, &resized);
  EXPECT_TRUE(WriteImage8(resized, "test_gray8_resize.png"));
}

TEST(ImageUtilTest, TestConvolve) {
  Image8 image8;
  EXPECT_TRUE(ReadImage<uint8_t>(FLAGS_image, &image8));
  const int rows = image8.rows(), cols = image8.cols();
  LOG(INFO) << "image8 " << rows << "x" << cols;

  ImageD kernel = ImageD::Constant(5, 1, 1.f);
//  kernel /= kernel.sum();
  BuildGaussianKernel(1, &kernel);
LOG(INFO) << kernel;
  Image8 out(rows, cols), tmp(rows, cols);
//  Convolve<uint8_t, double, uint8_t>(image8, kernel, &out);
  LOG(INFO) << "convolute " << image8.rows() << "x" << image8.cols() << ", "
            << kernel.rows() << "x" << kernel.cols();
//  Convolve(image8, kernel, &out);
  Convolve(image8, kernel, &tmp);
  Convolve(tmp, kernel.transpose(), &out);
  LOG(INFO) << "done. " << kernel.rows() << "x" << kernel.cols();
  EXPECT_TRUE(WriteImage8(out, "test_gray8_conv.png"));

  image8.resize(15, 10);
  image8.fill(255);
  out.resize(image8.rows(), image8.cols());
  Convolve(image8, kernel, &out);
  EXPECT_TRUE(WriteImage8(out, "test_gray8_conv2.png"));
}
*/
}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
