#include "innerproduct.h"

#include "paramdict.h"
#include "modelbin.h"
#include "fused_activation.h"

namespace ncnn
{
InnerProduct::InnerProduct()
{
    one_blob_only = true;
    support_inplace = false;
}

int InnerProduct::load_param(const ParamDict& pd)
{
    num_output = pd.get(0, 0);
    bias_term = pd.get(1, 0);
    weight_data_size = pd.get(2, 0);
    int8_scale_term = pd.get(8, 0);
    activation_type = pd.get(9, 0);
    activation_params = pd.get(10, Mat());

    if(int8_scale_term) {
        printf("please build ncnn with NCNN_INT8 enabled for int8 inference\n");

        return -1;
    }

    return 0;
}

int InnerProduct::load_model(const ModelBin& mb)
{
    weight_data = mb.load(weight_data_size, 0);

    if(weight_data.empty()) {
        return -100;
    }

    if(bias_term) {
        bias_data = mb.load(num_output, 1);

        if(bias_data.empty()) {
            return -100;
        }
    }

    return 0;
}

int InnerProduct::create_pipeline(const Option& opt)
{
    (void)(opt);

    return 0;
}

int InnerProduct::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int num_input = weight_data_size / num_output;

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    
    size_t elemsize = bottom_blob.elemsize;

    int size = w * h;

    if(bottom_blob.dims == 2 && w == num_input && h > 1) {
        // gemm
        top_blob.create(num_output, h, elemsize, opt.blob_allocator);

        if(top_blob.empty()) {
            return -100;
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for(int j = 0; j < h; j++) {
            const float* m = bottom_blob.row(j);
            float* outptr = top_blob.row(j);

            for(int p = 0; p < num_output; p++) {
                const float* kptr = (const float*)weight_data + w * p;

                float sum = 0.f;

                if(bias_term) {
                    sum = bias_data[p];
                }

                for(int i = 0; i < w; i++) {
                    sum += m[i] * kptr[i];
                }

                outptr[p] = activation_ss(sum, activation_type, activation_params);
            }
        }

        return 0;
    }

    top_blob.create(num_output, elemsize, opt.blob_allocator);

    if(top_blob.empty()) {
        return -100;
    }

    // num_output
    #pragma omp parallel for num_threads(opt.num_threads)
    for(int p = 0; p < num_output; p++) {
        float sum = 0.f;

        if(bias_term) {
            sum = bias_data[p];
        }

        // channels
        for(int q = 0; q < channels; q++) {
            const float* w = (const float*)weight_data + size * channels * p + size * q;
            const float* m = bottom_blob.channel(q);

            for(int i = 0; i < size; i++) {
                sum += m[i] * w[i];
            }
        }

        top_blob[p] = activation_ss(sum, activation_type, activation_params);
    }

    return 0;
}
}