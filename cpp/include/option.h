#pragma once

namespace ncnn
{
class Allocator;

class Option
{
public:
    // default option
    Option();

    // light mode
    // intermediate blob will be recycled when enabled
    // enabled by default
    bool lightmode;

    // thread count
    // default value is the one returned by get_cpu_count()
    int num_threads;

    // blob memory allocator
    Allocator* blob_allocator;

    // workspace memory allocator
    Allocator* workspace_allocator;

    // the time openmp threads busy-wait for more work before going to sleep
    // default value is 20ms to keep the cores enabled
    // without too much extra power consumption afterwards
    int openmp_blocktime;

    // enabled winograd convolution optimization
    // improve convolution 3x3 stride1 performance, may consume more memory
    // changes should be applied before loading network structure and weight
    // enabled by default
    bool use_winograd_convolution;

    // enable sgemm convolution optimization
    // improve convolution 1x1 stride1 performance, may consume more memory
    // changes should be applied before loading network structure and weight
    // enabled by default
    bool use_sgemm_convolution;

    // enable quantized int8 inference
    // use low-precision int8 path for quantized model
    // changes should be applied before loading network structure and weight
    // enabled by default
    bool use_int8_inference;

    // enable vulkan compute
    bool use_vulkan_compute;

    // enable bf16 data type for storage
    // improve most operator performance on all arm devices, may consume more memory
    bool use_bf16_storage;

    // enable options for gpu inference
    bool use_fp16_packed;
    bool use_fp16_storage;
    bool use_fp16_arithmetic;
    bool use_int8_packed;
    bool use_int8_storage;
    bool use_int8_arithmetic;

    // enable simd-friendly packed memory layout
    // improve all operator performance on all arm devices, will consume more memory
    // changes should be applied before loading network structure and weight
    // enabled by default
    bool use_packing_layout;

    bool use_shader_pack8;

    // subgroup option
    bool use_subgroup_basic;
    bool use_subgroup_vote;
    bool use_subgroup_ballot;
    bool use_subgroup_shuffle;

    // turn on for adreno
    bool use_image_storage;
    bool use_tensor_storage;

    bool use_reserved_0;

    // enable DAZ(Denormals-Are-Zero) and FTZ(Flush-To-Zero)
    // default value is 3
    // 0 = DAZ OFF, FTZ OFF
    // 1 = DAZ On , FTZ OFF
    // 2 = DAZ OFF, FTZ ON
    // 3 = DAZ ON , FTZ ON
    int flush_denormals;

    bool use_local_pool_allocator;

    // enable local memory optimization for gpu inference
    bool use_shader_local_memory;

    // enable cooperative matrix optimization for gpu inference
    bool use_cooperative_matrix;

    // more fine-grained control of winograd convolution
    bool use_winograd23_convolution;
    bool use_winograd43_convolution;
    bool use_winograd63_convolution;

    bool use_reserved_6;
    bool use_reserved_7;
    bool use_reserved_8;
    bool use_reserved_9;
    bool use_reserved_10;
    bool use_reserved_11;    
};
}