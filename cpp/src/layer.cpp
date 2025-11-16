#include "layer.h"

#include <cstring>

#include "mat.h"
#include "paramdict.h"
#include "layer_declaration.h"

namespace ncnn
{
Layer::Layer()
{
    one_blob_only = false;
    support_inplace = false;
    support_vulkan = false;
    support_packing = false;

    support_bf16_storage = false;
    support_fp16_storage = false;
    support_int8_storage = false;
    support_image_storage = false;
    support_tensor_storage = false;

    support_reserved_00 = false;

    typeindex = -1;

    userdata = 0;
}

Layer::~Layer()
{
    
}

int Layer::load_param(const ParamDict&)
{
    return 0;
}

int Layer::load_model(const ModelBin&)
{
    return 0;
}

int Layer::create_pipeline(const Option&)
{
    return 0;
}

int Layer::destroy_pipeline(const Option&)
{
    return 0;
}

int Layer::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    if(!support_inplace) {
        return -1;
    }

    top_blobs = bottom_blobs;

    for(int i = 0 ; i < (int)top_blobs.size(); i++) {
        top_blobs[i] = bottom_blobs[i].clone(opt.blob_allocator);

        if(top_blobs[i].empty()) {
            return -100;
        }
    }

    return forward_inplace(top_blobs, opt);
}

int Layer::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if(!support_inplace) {
        return -1;
    }

    top_blob = bottom_blob.clone(opt.blob_allocator);

    if(top_blob.empty()) {
        return -100;
    }

    return forward_inplace(top_blob, opt);
}

int Layer::forward_inplace(std::vector<Mat>&, const Option&) const
{
    return -1;
}

int Layer::forward_inplace(Mat&, const Option&) const
{
    return -1;
}

#include "layer_registry.h"

static const int layer_registry_entry_count = sizeof(layer_registry) / sizeof(layer_registry_entry);

int layer_to_index(const char* type)
{
    for(int i = 0; i < layer_registry_entry_count; i++) {
        if(strcmp(type, layer_registry[i].name) == 0) {
            return i;
        }       
    }

    return -1;
}

Layer* create_layer(int index)
{
    if(index < 0 || index >= layer_registry_entry_count) {
        return 0;
    }

    layer_creator_func layer_creator = 0;

    // NCNN_RUNTIME_CPU && NCNN_RVV
    {
        layer_creator = layer_registry[index].creator;
    }

    // *INDENT-ON*
    // clang-format on
    if(!layer_creator) {
        return 0;
    }

    Layer* layer = layer_creator(0);

    layer->typeindex = index;

    return layer;
}

Layer* create_layer(const char* type)
{
    int index = layer_to_index(type);

    if(index == -1) {
        return 0;
    }

    return create_layer(index);
}
}