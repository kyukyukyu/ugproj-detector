#include <iostream>
#include <limits>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "image_file.h"

using namespace std;
using namespace rvslam;

namespace {

TEST(ImageFileTest, TestReadImageGray8) {
  ImageGray8 gray;
  EXPECT_TRUE(ReadImage("test_gray8.png", &gray));
  EXPECT_TRUE(WriteImage(gray, "test_gray8_out.png"));
  EXPECT_TRUE(ReadImage("test_gray16.png", &gray));
  EXPECT_TRUE(WriteImage(gray, "test_gray16_gray8_out.png"));
  EXPECT_TRUE(ReadImage("test_gray16.png", &gray, PIXEL_TYPE_ADJUST));
  EXPECT_TRUE(WriteImage(gray, "test_gray16_gray8_out2.png"));
  EXPECT_TRUE(ReadImage("test_rgb8.png", &gray));
  EXPECT_TRUE(WriteImage(gray, "test_rgb_gray8_out.png"));

  MCImageGray8 mc_gray;
  EXPECT_TRUE(ReadImage("test_gray8.png", &mc_gray));
  EXPECT_TRUE(WriteImage(mc_gray, "test_mc8_out.png"));
  EXPECT_TRUE(ReadImage("test_gray16.png", &mc_gray));
  EXPECT_TRUE(WriteImage(mc_gray, "test_gray16_mc8_out.png"));
  EXPECT_TRUE(ReadImage("test_gray16.png", &mc_gray, PIXEL_TYPE_ADJUST));
  EXPECT_TRUE(WriteImage(mc_gray, "test_gray16_mc8_out2.png"));
  EXPECT_TRUE(ReadImage("test_rgb8.png", &mc_gray));
  EXPECT_TRUE(WriteImage(mc_gray, "test_rgb8_mc8_out.png"));
}

TEST(ImageFileTest, TestReadImageGray16) {
  ImageGray16 gray;
  EXPECT_TRUE(ReadImage("test_gray16.png", &gray));
  EXPECT_TRUE(WriteImage(gray, "test_gray16_out.png"));
  EXPECT_TRUE(ReadImage("test_gray8.png", &gray));
  EXPECT_TRUE(WriteImage(gray, "test_gray8_gray16_out.png"));
  EXPECT_TRUE(ReadImage("test_gray8.png", &gray, PIXEL_TYPE_ADJUST));
  EXPECT_TRUE(WriteImage(gray, "test_gray8_gray16_out2.png"));
  EXPECT_TRUE(ReadImage("test_rgb8.png", &gray));
  EXPECT_TRUE(WriteImage(gray, "test_rgb_gray16_out.png"));

  MCImageGray16 mc_gray;
  EXPECT_TRUE(ReadImage("test_gray16.png", &mc_gray));
  EXPECT_TRUE(WriteImage(mc_gray, "test_mc16_out.png"));
  EXPECT_TRUE(ReadImage("test_gray8.png", &mc_gray));
  EXPECT_TRUE(WriteImage(mc_gray, "test_gray8_mc16_out.png"));
  EXPECT_TRUE(ReadImage("test_rgb8.png", &mc_gray));
  EXPECT_TRUE(WriteImage(mc_gray, "test_rgb8_mc16_out.png"));
}

TEST(ImageFileTest, TestReadImageRGB8) {
  MCImageRGB8 rgb;
  EXPECT_TRUE(ReadImage("test_rgb8.png", &rgb));
  EXPECT_TRUE(WriteImage(rgb, "test_rgb8_out.png"));
  EXPECT_TRUE(ReadImage("test_gray8.png", &rgb));
  EXPECT_TRUE(WriteImage(rgb, "test_gray8_rgb8_out.png"));
  EXPECT_TRUE(ReadImage("test_gray16.png", &rgb));
  EXPECT_TRUE(WriteImage(rgb, "test_gray16_rgb8_out.png"));
  EXPECT_TRUE(ReadImage("test_gray16.png", &rgb, PIXEL_TYPE_ADJUST));
  EXPECT_TRUE(WriteImage(rgb, "test_gray16_rgb8_out2.png"));
}

TEST(ImageFileTest, TestReadImageFloat) {
  ImageFloat gray;
  EXPECT_TRUE(ReadImage("test_gray8.png", &gray, PIXEL_TYPE_ADJUST));
  EXPECT_TRUE(WriteImage(ConvertImageTo<uint8_t>(gray),
                         "test_gray8_float_out.png"));
  EXPECT_TRUE(WriteImage(ConvertImageTo<uint16_t>(gray),
                         "test_gray8_float_out2.png"));
/*
  EXPECT_TRUE(ReadImage("test_gray16.png", &gray));
  EXPECT_TRUE(WriteImage(gray, "test_gray16_gray8_out.png"));
  EXPECT_TRUE(ReadImage("test_gray16.png", &gray, PIXEL_TYPE_ADJUST));
  EXPECT_TRUE(WriteImage(gray, "test_gray16_gray8_out2.png"));
  EXPECT_TRUE(ReadImage("test_rgb8.png", &gray));
  EXPECT_TRUE(WriteImage(gray, "test_rgb_gray8_out.png"));

  MCImageFloat mc_gray;
  EXPECT_TRUE(ReadImage("test_gray8.png", &mc_gray));
  EXPECT_TRUE(WriteImage(mc_gray, "test_mc8_out.png"));
  EXPECT_TRUE(ReadImage("test_gray16.png", &mc_gray));
  EXPECT_TRUE(WriteImage(mc_gray, "test_gray16_mc8_out.png"));
  EXPECT_TRUE(ReadImage("test_gray16.png", &mc_gray, PIXEL_TYPE_ADJUST));
  EXPECT_TRUE(WriteImage(mc_gray, "test_gray16_mc8_out2.png"));
  EXPECT_TRUE(ReadImage("test_rgb8.png", &mc_gray));
  EXPECT_TRUE(WriteImage(mc_gray, "test_rgb8_mc8_out.png"));
*/
}

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
