#pragma once

#include <vector>

#include "layer.h"

namespace ncnn
{
class Mat;

class Split : public Layer
{
public:
    Split();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};
}