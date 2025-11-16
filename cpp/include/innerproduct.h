#pragma once

#include "layer.h"
#include "mat.h"

namespace ncnn
{
class ParamDict;
class ModelBin;

class InnerProduct : public Layer
{
public:
    InnerProduct();

    virtual int load_param(const ParamDict& pd);
    virtual int load_model(const ModelBin& mb);
    virtual int create_pipeline(const Option& opt);
    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    // param
    int num_output;
    int bias_term;

    int weight_data_size;

    int int8_scale_term;

    // 0 = none 1 = relu 2 = leakyrelu 3 = clip 4 = sigmoid
    int activation_type;
    Mat activation_params;

    // model
    Mat weight_data;
    Mat bias_data;
};
}