#pragma once

#include "layer.h"

namespace ncnn
{
class ParamDict;
class Mat;

class Reshape : public Layer
{
public:
    Reshape();

    virtual int load_param(const ParamDict& pd);
    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    // reshape flag
    // 0 = copy from bottom
    // -1 = remaining
    // -233 = drop this dim (default)
    int w;
    int h;
    int d;
    int c;

    // flag permute chw->hwc or hw->wh before and after reshape
    int permute;

    int ndim;
};
}