#pragma once

#include <vector>

#include "layer.h"
#include "mat.h"

namespace ncnn
{
class ParamDict;
class ModelBin;

class MemoryData : public Layer
{
public:
    MemoryData();

    virtual int load_param(const ParamDict& pd);
    virtual int load_model(const ModelBin& mb);
    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

    int w;
    int h;
    int c;

    Mat data;
};
}