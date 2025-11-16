#pragma once

#include "layer.h"

namespace ncnn
{
class ParamDict;

class Permute : public Layer
{
public:
    Permute();

    virtual int load_param(const ParamDict& pd);
    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    int order_type;
};
}