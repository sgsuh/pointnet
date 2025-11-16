#pragma once

#include "layer.h"

namespace ncnn
{
class ParamDict;

class BinaryOp : public Layer
{
public:
    BinaryOp();

    virtual int load_param(const ParamDict& pd);

    using Layer::forward;
    using Layer::forward_inplace;

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

    enum OperationType
    {
        Operation_ADD = 0,
        Operation_SUB = 1,
        Operation_MUL = 2,
        Operation_DIV = 3,
        Operation_MAX = 4,
        Operation_MIN = 5,
        Operation_POW = 6,
        Operation_RSUB = 7,
        Operation_RDIV = 8
    };

    // param
    int op_type;
    int with_scalar;

    float b;
};
}