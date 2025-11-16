#pragma once

#include "layer.h"

namespace ncnn
{
class ParamDict;

class Gemm : public Layer
{
public:
    Gemm();

    virtual int load_param(const ParamDict& pd);
    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

    float alpha;
    float beta;

    int transA;
    int transB;
};
}