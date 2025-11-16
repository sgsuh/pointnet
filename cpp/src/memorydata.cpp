#include "memorydata.h"

#include "paramdict.h"
#include "modelbin.h"

namespace ncnn
{
MemoryData::MemoryData()
{
    one_blob_only = false;
    support_inplace = false;
}

int MemoryData::load_param(const ParamDict& pd)
{
    w = pd.get(0, 0);
    h = pd.get(1, 0);
    c = pd.get(2, 0);

    return 0;
}

int MemoryData::load_model(const ModelBin& mb)
{
    if(c != 0) {
        data = mb.load(w, h, c, 1);
    }
    else if(h != 0) {
        data = mb.load(w, h, 1);
    }
    else if(w != 0) {
        data = mb.load(w, 1);
    }
    else {
        data.create(1);
    }

    if(data.empty()) {
        return -100;
    }

    return 0;
}

int MemoryData::forward(const std::vector<Mat>&, std::vector<Mat>& top_blobs, const Option& opt) const
{
    Mat& top_blob = top_blobs[0];

    top_blob = data.clone(opt.blob_allocator);

    if(top_blob.empty()) {
        return -100;
    }

    return 0;
}
}