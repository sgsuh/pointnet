#include "batchnorm.h"

namespace ncnn
{
class BatchNorm_final : virtual public BatchNorm
{
public:
    virtual int create_pipeline(const Option& opt)
    {
        {
            int ret = BatchNorm::create_pipeline(opt);

            if(ret) {
                return ret;
            }
        }

        return 0;
    }

    virtual int destroy_pipeline(const Option& opt)
    {
        {
            int ret = BatchNorm::destroy_pipeline(opt);

            if(ret) {
                return ret;
            }
        }

        return 0;
    }
};
DEFINE_LAYER_CREATOR(BatchNorm_final)
}

#include "flatten.h"

namespace ncnn
{
class Flatten_final : virtual public Flatten
{
public:
    virtual int create_pipeline(const Option& opt)
    {
        {
            int ret = Flatten::create_pipeline(opt);

            if(ret) {
                return ret;
            }
        }

        return 0;
    }

    virtual int destroy_pipeline(const Option& opt) 
    {
        {
            int ret = Flatten::destroy_pipeline(opt);

            if(ret) {
                return ret;
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(Flatten_final)
}

#include "innerproduct.h"

namespace ncnn
{
class InnerProduct_final : virtual public InnerProduct
{
public:
    virtual int create_pipeline(const Option& opt) 
    {
        {
            int ret = InnerProduct::create_pipeline(opt);

            if(ret) {
                return ret;
            }
        }

        return 0;
    }

    virtual int destroy_pipeline(const Option& opt)
    {
        {
            int ret = InnerProduct::destroy_pipeline(opt);

            if(ret) {
                return ret;
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(InnerProduct_final)
}

#include "input.h"

namespace ncnn
{
class Input_final : virtual public Input
{
public:
    virtual int create_pipeline(const Option& opt)
    {
        {
            int ret = Input::create_pipeline(opt);

            if(ret) {
                return ret;
            }
        }

        return 0;
    }

    virtual int destroy_pipeline(const Option& opt)
    {
        {
            int ret = Input::destroy_pipeline(opt);

            if(ret) {
                return ret;
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(Input_final)
}

#include "memorydata.h"

namespace ncnn
{
class MemoryData_final : virtual public MemoryData
{
public:
    virtual int create_pipeline(const Option& opt)
    {
        {
            int ret = MemoryData::create_pipeline(opt);

            if(ret) {
                return ret;
            }
        }

        return 0;
    }

    virtual int destroy_pipeline(const Option& opt)
    {
        {
            int ret = MemoryData::destroy_pipeline(opt);

            if(ret) {
                return ret;
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(MemoryData_final)
}

#include "relu.h"

namespace ncnn
{
class ReLU_final : virtual public ReLU
{
public:
    virtual int create_pipeline(const Option& opt)
    {
        {
            int ret = ReLU::create_pipeline(opt);

            if(ret) {
                return ret;
            }
        }

        return 0;
    }

    virtual int destroy_pipeline(const Option& opt)
    {
        {
            int ret = ReLU::destroy_pipeline(opt);

            if(ret) {
                return ret;
            }
        }

        return 0; 
    }
};

DEFINE_LAYER_CREATOR(ReLU_final)
}

#include "reshape.h"

namespace ncnn
{
class Reshape_final : virtual public Reshape
{
public:
    virtual int create_pipeline(const Option& opt)
    {
        {
            int ret = Reshape::create_pipeline(opt);

            if(ret) {
                return ret;
            }
        }

        return 0;
    }

    virtual int destroy_pipeline(const Option& opt)
    {
        {
            int ret = Reshape::destroy_pipeline(opt);

            if(ret) {
                return ret;
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(Reshape_final)
}

#include "split.h"

namespace ncnn
{
class Split_final : virtual public Split
{
public:
    virtual int create_pipeline(const Option& opt)
    {
        {
            int ret = Split::create_pipeline(opt);

            if(ret) {
                return ret;
            }
        }

        return 0;
    }

    virtual int destroy_pipeline(const Option& opt) 
    {
        {
            int ret = Split::destroy_pipeline(opt);

            if(ret) {
                return ret;
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(Split_final)
}

#include "binaryop.h"

namespace ncnn
{
class BinaryOp_final : virtual public BinaryOp
{
public:
    virtual int create_pipeline(const Option& opt)
    {
        {
            int ret = BinaryOp::create_pipeline(opt);

            if(ret) {
                return ret;
            }
        }

        return 0;
    }

    virtual int destroy_pipeline(const Option& opt)
    {
        {
            int ret = BinaryOp::destroy_pipeline(opt);

            if(ret) {
                return ret;
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(BinaryOp_final)
}

#include "padding.h"

namespace ncnn
{
class Padding_final : virtual public Padding
{
public:
    virtual int create_pipeline(const Option& opt)
    {
        {
            int ret = Padding::create_pipeline(opt);

            if(ret) {
                return ret;
            }
        }

        return 0;
    }

    virtual int destroy_pipeline(const Option& opt)
    {
        {
            int ret = Padding::destroy_pipeline(opt);

            if(ret) {
                return ret;
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(Padding_final)
}

#include "permute.h"

namespace ncnn
{
class Permute_final : virtual public Permute
{
public:
    virtual int create_pipeline(const Option& opt)
    {
        {
            int ret = Permute::create_pipeline(opt);

            if(ret) {
                return ret;
            }
        }

        return 0;
    }

    virtual int destroy_pipeline(const Option& opt)
    {
        {
            int ret = Permute::destroy_pipeline(opt);

            if(ret) {
                return ret;
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(Permute_final)
}

#include "packing.h"

namespace ncnn
{
class Packing_final : virtual public Packing
{
public:
    virtual int create_pipeline(const Option& opt)
    {
        {
            int ret = Packing::create_pipeline(opt);
            
            if(ret) {
                return ret;
            }
        }

        return 0;
    }

    virtual int destroy_pipeline(const Option& opt)
    {
        {
            int ret = Packing::destroy_pipeline(opt);

            if(ret) {
                return ret;
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(Packing_final)
}

#include "cast.h"

namespace ncnn
{
class Cast_final : virtual public Cast
{
public:
    virtual int create_pipeline(const Option& opt)
    {
        {
            int ret = Cast::create_pipeline(opt);

            if(ret) {
                return ret;
            }
        }

        return 0;
    }

    virtual int destroy_pipeline(const Option& opt)
    {
        {
            int ret = Cast::destroy_pipeline(opt);

            if(ret) {
                return ret;
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(Cast_final)
}

#include "gemm.h"

namespace ncnn
{
class Gemm_final : virtual public Gemm
{
public:
    virtual int create_pipeline(const Option& opt)
    {
        {
            int ret = Gemm::create_pipeline(opt);

            if(ret) {
                return ret;
            }
        }

        return 0;
    }

    virtual int destroy_pipeline(const Option& opt)
    {
        {
            int ret = Gemm::destroy_pipeline(opt);

            if(ret) {
                return ret;
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(Gemm_final)
}

#include "convolution1d.h"

namespace ncnn
{
class Convolution1D_final : virtual public Convolution1D
{
public:
    virtual int create_pipeline(const Option& opt)
    {
        {
            int ret = Convolution1D::create_pipeline(opt);

            if(ret) {
                return ret;
            }
        }

        return 0;
    }

    virtual int destroy_pipeline(const Option& opt)
    {
        {
            int ret = Convolution1D::destroy_pipeline(opt);

            if(ret) {
                return ret;
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(Convolution1D_final)
}

#include "pooling1d.h"

namespace ncnn
{
class Pooling1D_final : virtual public Pooling1D
{
public:
    virtual int create_pipeline(const Option& opt)
    {
        {
            int ret = Pooling1D::create_pipeline(opt);

            if(ret) {
                return ret;
            }
        }

        return 0;
    }

    virtual int destroy_pipeline(const Option& opt)
    {
        {
            int ret = Pooling1D::destroy_pipeline(opt);

            if(ret) {
                return ret;
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(Pooling1D_final)
}