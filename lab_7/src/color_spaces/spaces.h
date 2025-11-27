#ifndef __SPACES_H__
#define __SPACE_H__

#include "opencv2/opencv.hpp"

class RGB {
public:
  float r;
  float g;
  float b;

  RGB(float r, float g, float b);
  RGB(uchar r, uchar g, uchar b);

  uchar R();
  uchar G();
  uchar B();
};

class HSV {
public:
  float h; // 0-360
  float s; // 0-1
  float v; // 0-1

  HSV(float h, float s, float v);
  HSV(uchar r, uchar g, uchar b);
  HSV(RGB rgb);
};

#endif