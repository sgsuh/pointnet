#pragma once

namespace ncnn
{
namespace LayerType
{
enum LayerType
{
#include "layer_type_enum.h"

    CustomBit = (1 << 8)
};
}
}