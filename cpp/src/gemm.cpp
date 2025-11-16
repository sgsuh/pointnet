#include "gemm.h"

#include "paramdict.h"
#include "mat.h"

namespace ncnn
{
Gemm::Gemm()
{
    one_blob_only = false;
    support_inplace = false;
}

int Gemm::load_param(const ParamDict& pd)
{
    alpha = pd.get(0, 1.f);
    beta = pd.get(1, 1.f);
    transA = pd.get(2, 0);
    transB = pd.get(3, 0);

    return 0;
}

int Gemm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& A0 = bottom_blobs[0];
    const Mat& B0 = bottom_blobs[1];

    size_t elemsize = A0.elemsize;

    Mat A;

    if(transA == 0) {
        A = A0;
    }
    else {
        // transpose A to row-major
        A.create(A0.h, A0.w, elemsize, opt.workspace_allocator);

        for(int i = 0; i < A.h; i++) {
            float* ptr = A.row(i);

            for(int j = 0; j < A.w; j++) {
                ptr[j] = A0.row(j)[i];
            }
        }
    }

    Mat B;

    if(transB == 0) {
        // transpose B to col-major
        B.create(B0.h, B0.w, elemsize, opt.workspace_allocator);

        for(int i = 0; i < B.h; i++) {
            float* ptr = B.row(i);

            for(int j = 0; j < B.w; j++) {
                ptr[j] = B0.row(j)[i];
            }
        }
    }
    else {
        B = B0;
    }

    int M = A.h;
    int K = A.w;
    int N = B.h;

    bool has_C = bottom_blobs.size() == 3;

    const float* ptrC = 0;
    int broadcast_type_C = 0;

    if(has_C) {
        const Mat& C = bottom_blobs[2];

        ptrC = C;

        if(C.dims == 1 && C.w == 1) {
            // scalar
            broadcast_type_C = 0;
        }

        if(C.dims == 1 && C.w == M) {
            // M
            // auto broadcast from h to w is the ncnn-style convention
            broadcast_type_C = 1;
        }

        if(C.dims == 1 && C.w == N) {
            // N
            broadcast_type_C = 4;
        }

        if(C.dims == 2 && C.w == 1 && C.h == M) {
            // M x 1
            broadcast_type_C = 2;
        }

        if(C.dims == 2 && C.w == N && C.h == M) {
            // M x N
            broadcast_type_C = 3;
        }

        if(C.dims == 2 && C.w == N && C.h == 1) {
            // 1 x N
            broadcast_type_C = 4;
        }
    }

    Mat& top_blob = top_blobs[0];

    top_blob.create(N, M, elemsize, opt.blob_allocator);

    if(top_blob.empty()) {
        return -100;
    }

    float* outptr = top_blob;

    for(int i = 0; i < M; i++) {
        const float* ptrA = A.row(i);

        for(int j = 0; j < N; j++) {
            const float* ptrB = B.row(j);

            float sum = 0.f;

            if(has_C) {
                if(broadcast_type_C == 0) {
                    sum = ptrC[0];
                }

                if(broadcast_type_C == 1) {
                    sum = ptrC[i];
                }

                if(broadcast_type_C == 2) {
                    sum = ptrC[i];
                }

                if(broadcast_type_C == 3) {
                    sum = ptrC[i * N + j];
                }

                if(broadcast_type_C == 4) {
                    sum = ptrC[j];
                }

                sum *= beta;
            }

            for(int k = 0; k < K; k++) {
                sum += ptrA[k] * ptrB[k];
            }

            *outptr++ = sum * alpha;
        }
    }

    return 0;
}
}