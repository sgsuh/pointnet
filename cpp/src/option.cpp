#include "option.h"

#include "cpu.h"

namespace ncnn
{
Option::Option()
{
    lightmode = true;
    num_threads = get_big_cpu_count();
    blob_allocator = 0;
    workspace_allocator = 0;

    openmp_blocktime = 20;

    use_winograd_convolution = true;
    use_sgemm_convolution = true;
    use_int8_inference = true;
    use_vulkan_compute = false;

    use_bf16_storage = false;

    use_fp16_packed = true;
    use_fp16_storage = true;
    use_fp16_arithmetic = true;
    use_int8_packed = true;
    use_int8_storage = true;
    use_int8_arithmetic = false;

    use_packing_layout = true;

    use_shader_pack8 = false;

    use_subgroup_basic = false;
    use_subgroup_vote = false;
    use_subgroup_ballot = false;
    use_subgroup_shuffle = false;

    use_image_storage = false;
    use_tensor_storage = false;

    use_reserved_0 = false;

    flush_denormals = 3;

    use_local_pool_allocator = true;

    use_shader_local_memory = true;
    use_cooperative_matrix = true;

    use_winograd23_convolution = true;
    use_winograd43_convolution = true;
    use_winograd63_convolution = true;
}
}