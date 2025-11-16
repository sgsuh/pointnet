#pragma once

#include "layer.h"

namespace ncnn
{
class ParamDict;
class Mat;

class Input : public Layer
{
public:
    Input();

    virtual int load_param(const ParamDict& pd);
    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

    int w;
    int h;
    int d;
    int c;
};
}