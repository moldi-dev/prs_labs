#include "spaces.h"
#include "../common/logger/logger.h"
#include <algorithm>

// Currently this function is not exposed.
HSV rgb_to_hsv(float r, float g, float b)
{
  r = r / 255.0;
  g = g / 255.0;
  b = b / 255.0;

  // h, s, v = hue, saturation, value
  float cmax = std::max(r, std::max(g, b)); // maximum of r, g, b
  float cmin = std::min(r, std::min(g, b)); // minimum of r, g, b
  float diff = cmax - cmin; // diff of cmax and cmin.
  float h = -1, s = -1;
    
  // if cmax and cmax are equal then h = 0
  if (cmax == cmin)
    h = 0;

  // if cmax equal r then compute h
  else if (cmax == r)
    h = int(60 * ((g - b) / diff) + 360) % 360;

  // if cmax equal g then compute h
  else if (cmax == g)
    h = int(60 * ((b - r) / diff) + 120) % 360;

  // if cmax equal b then compute h
  else if (cmax == b)
    h = int(60 * ((r - g) / diff) + 240) % 360;

  // if cmax equal zero
  if (cmax == 0)
    s = 0;
  else
    s = (diff / cmax) * 100;

  // compute v
  float v = cmax * 100;

  return HSV(h, s, v);
}

RGB::RGB(float r, float g, float b)
  :r(r), g(g), b(b)
{}

RGB::RGB(uchar r, uchar g, uchar b)
  :r((float)r), g((float)g), b((float)b)
{}

uchar RGB::R()
{
  return (uchar) r;
}

uchar RGB::G()
{
  return (uchar) g;
}

uchar RGB::B()
{
  return (uchar) b;
}

HSV::HSV(float h, float s, float v)
  :h(h), s(s), v(v)
{}

HSV::HSV(uchar r, uchar g, uchar b)
  :h(0), s(0), v(0)
{
  HSV tmp = rgb_to_hsv((float)r, (float)g, (float)b);
  this->h = tmp.h;
  this->s = tmp.s;
  this->v = tmp.v;
}

HSV::HSV(RGB rgb)
  :h(0), s(0), v(0)
{
  HSV tmp = rgb_to_hsv(rgb.r, rgb.g, rgb.b);
  this->h = tmp.h;
  this->s = tmp.s;
  this->v = tmp.v;
}