#pragma once

#include "layer.h"

namespace ncnn
{
class Mat;

class Flatten : public Layer
{
public:
    Flatten();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};
}