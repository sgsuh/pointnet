#pragma once

#include "layer.h"

#include "mat.h"

namespace ncnn
{
class ParamDict;
class ModelBin;

class Convolution1D : public Layer
{
public:
    Convolution1D();

    virtual int load_param(const ParamDict& pd);
    virtual int load_model(const ModelBin& mb);
    virtual int create_pipeline(const Option& opt);
    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

    // param
    int num_output;
    int kernel_w;
    int dilation_w;
    int stride_w;
    int pad_left;
    int pad_right;

    float pad_value;

    int bias_term;

    int weight_data_size;

    // 0 = none 1 = relu 2 = leakyrelu 3 = clip 4 = sigmoid
    int activation_type;

    Mat activation_params;

    int dynamic_weight;

    // model
    Mat weight_data;
    Mat bias_data;

protected:
    void make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered, const Option& opt) const;
    void make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered, int kernel_w, const Option& opt) const;
};
}