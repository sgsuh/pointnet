#pragma once

#include <vector>
#include <string>

#include "option.h"

namespace ncnn
{
class Mat;
class ParamDict;
class ModelBin;

class Layer
{
public:
    // empty
    Layer();

    // virtual destructor
    virtual ~Layer();

    // load layer specific parameter from parsed dict
    // return 0 if success
    virtual int load_param(const ParamDict& pd);

    // load layer specific weight data from model binary
    // return 0 if success
    virtual int load_model(const ModelBin& mb);

    // layer implementation specific setup
    // return 0 if success
    virtual int create_pipeline(const Option& opt);

    // layer implementation specific clean
    // return 0 if success
    virtual int destroy_pipeline(const Option& opt);

    // implement inference
    // return 0 if success
    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    // implement inplace inference
    // return 0 if success
    virtual int forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const;
    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

    // one input and one output blob
    bool one_blob_only;

    // support inplace inference
    bool support_inplace;

    // support vulkan compute
    bool support_vulkan;

    // accept input blob with packed storage
    bool support_packing;

    // accept bf16
    bool support_bf16_storage;

    // accept fp16;
    bool support_fp16_storage;

    // accept int8
    bool support_int8_storage;

    // shader image storage
    bool support_image_storage;

    // shader tensor storage
    bool support_tensor_storage;

    bool support_reserved_00;

    // custom user data
    void* userdata;

    // layer type index
    int typeindex;

    // layer type name
    std::string type;

    // layer name
    std::string name;

    // blob index which this layer needs as input
    std::vector<int> bottoms;

    // blob index which this layer produces as output
    std::vector<int> tops;

    // shape hint
    std::vector<Mat> bottom_shapes;
    std::vector<Mat> top_shapes;
};

// layer factoty function
typedef Layer* (*layer_creator_func)(void*);
typedef void (*layer_destroyer_func)(Layer*, void*);

struct layer_registry_entry
{
    // layer type name
    const char* name;

    // layer factory entry
    layer_creator_func creator;
};

struct custom_layer_registry_entry
{
    // layer type name
    const char* name;

    // layer factory entry
    layer_creator_func creator;
    layer_destroyer_func destroyer;

    void* userdata;
};

// get layer type from type name
int layer_to_index(const char* type);

// create layer from type name
Layer* create_layer(const char* type);

// create layer from layer type
Layer* create_layer(int index);

#define DEFINE_LAYER_CREATOR(name)              \
    ::ncnn::Layer* name##_layer_creator(void*)  \
    {                                           \
        return new name;                        \
    }
}