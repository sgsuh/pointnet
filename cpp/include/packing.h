#pragma once

#include "layer.h"

namespace ncnn
{
class Packing : public Layer
{
public:
    Packing();

    virtual int load_param(const ParamDict& pd);
    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    int out_elempack;
    int use_padding;

    // element type
    // 0 = auto
    // 1 = fp32
    // 2 = fp16p
    // 3 = fp16s
    int cast_type_from;
    int cast_type_to;

    // storage type
    // 0 = buffer
    // 1 = image
    int storage_type_from;
    int storage_type_to;
};
}