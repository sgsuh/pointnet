#pragma once

#include "layer.h"

namespace ncnn
{
class ParamDict;
class Mat;

class ReLU : public Layer
{
public:
    ReLU();

    virtual int load_param(const ParamDict& pd);
    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

    float slope;
};
}