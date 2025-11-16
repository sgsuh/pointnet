#pragma once

#include "layer.h"

#include "mat.h"

namespace ncnn
{
class ParamDict;
class ModelBin;

class Padding : public Layer
{
public:
    Padding();

    virtual int load_param(const ParamDict& pd);
    virtual int load_model(const ModelBin& mb);
    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    int top;
    int bottom;
    int left;
    int right;
    int type;       // 0 = CONSTANT 1 = REPLICATE 2 = REFLECT

    float value;

    int front;
    int behind;

    // per channel pad value
    int per_channel_pad_data_size;

    Mat per_channel_pad_data;
};
}