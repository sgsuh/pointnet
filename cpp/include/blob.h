#pragma once

#include <string>

#include "mat.h"

namespace ncnn
{
class Blob
{
public:
    // empty
    Blob();

    // blob name
    std::string name;

    // layer index which produce this blob as output
    int producer;

    // layer index which need this blob as input
    int consumer;

    // shape hint
    Mat shape;
};
}