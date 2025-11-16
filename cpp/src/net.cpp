#include "net.h"

#include <vector>
#include <cstring>

#include "allocator.h"
#include "blob.h"
#include "layer.h"
#include "layer_type.h"
#include "datareader.h"
#include "paramdict.h"
#include "mat.h"
#include "modelbin.h"
#include "cpu.h"

namespace ncnn
{
class NetPrivate
{
public:
    NetPrivate(Option& _opt);

    int forward_layer(int layer_index, std::vector<Mat>& blob_mats, const Option& opt) const;

    int convert_layout(Mat& bottom_blob, const Layer* layer, const Option& opt) const;

    int do_forward_layer(const Layer* layer, std::vector<Mat>& blob_mats, const Option& opt) const;

    void update_input_output_indexes();
    void update_input_output_names();

    Option& opt;

    std::vector<Blob> blobs;
    std::vector<Layer*> layers;

    std::vector<int> input_blob_indexes;
    std::vector<int> output_blob_indexes;

    std::vector<const char*> input_blob_names;
    std::vector<const char*> output_blob_names;

    std::vector<custom_layer_registry_entry> custom_layer_registry;

    PoolAllocator* local_blob_allocator;
    PoolAllocator* local_workspace_allocator;
};

NetPrivate::NetPrivate(Option& _opt)
    : opt(_opt)
{
    local_blob_allocator = 0;
    local_workspace_allocator = 0;
}

void NetPrivate::update_input_output_indexes()
{
    input_blob_indexes.clear();
    output_blob_indexes.clear();

    for(size_t i = 0; i < layers.size(); i++) {
        if(layers[i]->typeindex == LayerType::Input) {
            int blob_index = layers[i]->tops[0];

            input_blob_indexes.push_back(blob_index);
        }
    }

    for(size_t i = 0; i < blobs.size(); i++) {
        if(blobs[i].producer != -1 && blobs[i].consumer == -1) {
            output_blob_indexes.push_back(i);
        }
    }
}

void NetPrivate::update_input_output_names()
{
    input_blob_names.clear();
    output_blob_names.clear();

    for(size_t i = 0; i < input_blob_indexes.size(); i++) {
        int blob_index = input_blob_indexes[i];

        input_blob_names.push_back(blobs[blob_index].name.c_str());
    }

    for(size_t i = 0; i < output_blob_indexes.size(); i++) {
        int blob_index = output_blob_indexes[i];

        output_blob_names.push_back(blobs[blob_index].name.c_str());
    }
}

int NetPrivate::forward_layer(int layer_index, std::vector<Mat>& blob_mats, const Option& opt) const
{
    // if(layer_index == 4) {
    //     printf("Layer Index: %d\n", layer_index);
    // }

    printf("Layer Index: %d\n", layer_index);

    const Layer* layer = layers[layer_index];

    if(layer->one_blob_only) {
        // load bottom blob
        int bottom_blob_index = layer->bottoms[0];

        if(blob_mats[bottom_blob_index].dims == 0) {
            int ret = forward_layer(blobs[bottom_blob_index].producer, blob_mats, opt);

            if(ret != 0) {
                return ret;
            }
        }
    }
    else {
        // load bottom blob
        for(size_t i = 0; i < layer->bottoms.size(); i++) {
            int bottom_blob_index = layer->bottoms[i];

            if(blob_mats[bottom_blob_index].dims == 0) {
                int ret = forward_layer(blobs[bottom_blob_index].producer, blob_mats, opt);

                if(ret != 0) {
                    return ret;
                }
            }
        }
    }

    int ret = do_forward_layer(layer, blob_mats, opt);

    if(ret != 0) {
        return ret;
    }

    return 0;
}

int NetPrivate::convert_layout(Mat& bottom_blob, const Layer* layer, const Option& opt) const
{
    if(opt.use_packing_layout) {
        // resolve dst_elempack
        int dims = bottom_blob.dims;
        int elemcount = 0;

        if(dims == 1) {
            elemcount = bottom_blob.elempack * bottom_blob.w;
        }

        if(dims == 2) {
            elemcount = bottom_blob.elempack * bottom_blob.h;
        }

        if(dims == 3 || dims == 4) {
            elemcount == bottom_blob.elempack * bottom_blob.c;
        }

        int elembits = bottom_blob.elembits();

        int dst_elempack = 1;

        if(layer->support_packing) {
            if(elembits == 32) {
                if(elemcount % 4 == 0) {
                    dst_elempack = 4;
                }
            }
            
            if(elembits == 16) {
                if(elemcount % 4 == 0) {
                    dst_elempack = 4;
                }
            }

            if(elembits == 8) {
                if(elemcount % 8 == 0) {
                    dst_elempack = 8;
                }
            }
        }

        if(bottom_blob.elempack != dst_elempack) {
            Mat bottom_blob_packed;

            convert_packing(bottom_blob, bottom_blob_packed, dst_elempack, opt);

            bottom_blob = bottom_blob_packed;
        }
    }

    return 0;
}

int NetPrivate::do_forward_layer(const Layer* layer, std::vector<Mat>& blob_mats, const Option& opt) const
{
    if(layer->one_blob_only) {
        int bottom_blob_index = layer->bottoms[0];
        int top_blob_index = layer->tops[0];

        Mat& bottom_blob_ref = blob_mats[bottom_blob_index];
        Mat bottom_blob;

        if(opt.lightmode) {
            // deep copy for inplace forward if data is shared
            if(layer->support_inplace && *bottom_blob_ref.refcount != 1) {
                bottom_blob = bottom_blob_ref.clone(opt.blob_allocator);
            }
        }

        if(bottom_blob.dims == 0) {
            bottom_blob = bottom_blob_ref;
        }

        convert_layout(bottom_blob, layer, opt);

        // forward
        if(opt.lightmode && layer->support_inplace) {
            Mat& bottom_top_blob = bottom_blob;
            int ret = layer->forward_inplace(bottom_top_blob, opt);

            if(ret != 0) {
                return ret;
            }

            // store top blob
            blob_mats[top_blob_index] = bottom_top_blob;
        }
        else {
            Mat top_blob;
            int ret = layer->forward(bottom_blob, top_blob, opt);

            if(ret != 0) {
                return ret;
            }

            // store top blob
            blob_mats[top_blob_index] = top_blob;
        }

        if(opt.lightmode) {
            // delete after taken in light mode
            blob_mats[bottom_blob_index].release();
        }
    }
    else {
        std::vector<Mat> bottom_blobs(layer->bottoms.size());

        for(size_t i = 0; i < layer->bottoms.size(); i++) {
            int bottom_blob_index = layer->bottoms[i];
            
            Mat& bottom_blob_ref = blob_mats[bottom_blob_index];
            bottom_blobs[i].release();

            if(opt.lightmode) {
                // deep copy for inplace forward if data is shared
                if(layer->support_inplace && *bottom_blob_ref.refcount != 1) {
                    bottom_blobs[i] = bottom_blob_ref.clone(opt.blob_allocator);
                }
            }

            if(bottom_blobs[i].dims == 0) {
                bottom_blobs[i] = bottom_blob_ref;
            }

            convert_layout(bottom_blobs[i], layer, opt);
        }

        // forward
        if(opt.lightmode && layer->support_inplace) {
            std::vector<Mat>& bottom_top_blobs = bottom_blobs;

            int ret = layer->forward_inplace(bottom_top_blobs, opt);

            if(ret != 0) {
                return ret;
            }

            // store top blobs
            for(size_t i = 0; i < layer->tops.size(); i++) {
                int top_blob_index = layer->tops[i];

                blob_mats[top_blob_index] = bottom_top_blobs[i];
            }
        }
        else {
            std::vector<Mat> top_blobs(layer->tops.size());

            int ret = layer->forward(bottom_blobs, top_blobs, opt);

            if(ret != 0) {
                return ret;
            }

            // store top blobs
            for(size_t i = 0; i < layer->tops.size(); i++) {
                int top_blob_index = layer->tops[i];

                blob_mats[top_blob_index] = top_blobs[i];
            }
        }

        for(size_t i = 0; i < layer->bottoms.size(); i++) {
            int bottom_blob_index = layer->bottoms[i];

            if(opt.lightmode) {
                // delete after taken in light mode
                blob_mats[bottom_blob_index].release();
            }
        }
    }

    return 0;
}

Net::Net()
    : d(new NetPrivate(opt))
{

}

Net::~Net()
{
    clear();

    delete d;
}

void Net::clear()
{
    d->blobs.clear();

    for(size_t i = 0; i < d->layers.size(); i++) {
        Layer* layer = d->layers[i];

        Option opt1 = opt;

        if(!layer->support_image_storage) {
            opt1.use_image_storage = false;
        }

        int dret = layer->destroy_pipeline(opt1);

        if(dret != 0) {
            printf("layer destroy_pipeline failed\n");
        }

        if(layer->typeindex & ncnn::LayerType::CustomBit) {
            int custom_index = layer->typeindex & ~ncnn::LayerType::CustomBit;

            if(d->custom_layer_registry[custom_index].destroyer) {
                d->custom_layer_registry[custom_index].destroyer(layer, d->custom_layer_registry[custom_index].userdata);
            }
            else {
                delete layer;
            }
        }
        else {
            delete layer;
        }
    }

    d->layers.clear();

    if(d->local_blob_allocator) {
        delete d->local_blob_allocator;

        d->local_blob_allocator = 0;
    }

    if(d->local_workspace_allocator) {
        delete d->local_workspace_allocator;

        d->local_workspace_allocator = 0;
    }
}

Extractor Net::create_extractor() const
{
    return Extractor(this, d->blobs.size());
}

const std::vector<int>& Net::input_indexes() const
{
    return d->input_blob_indexes;
}

const std::vector<int>& Net::output_indexes() const
{
    return d->output_blob_indexes;
}

const std::vector<const char*>& Net::input_names() const
{
    return d->input_blob_names;
}

const std::vector<const char*>& Net::output_names() const
{
    return d->output_blob_names;
}

const std::vector<Blob>& Net::blobs() const
{
    return d->blobs;
}

int Net::custom_layer_to_index(const char* type)
{
    const size_t custom_layer_registry_entry_count = d->custom_layer_registry.size();

    for(size_t i = 0; i < custom_layer_registry_entry_count; i++) {
        if(strcmp(type, d->custom_layer_registry[i].name) == 0) {
            return static_cast<int>(i);
        }       
    }

    return -1;
}

Layer* Net::create_custom_layer(int index)
{
    const size_t custom_layer_registry_entry_count = d->custom_layer_registry.size();

    if(index < 0 || static_cast<unsigned int>(index) >= custom_layer_registry_entry_count) {
        return 0;
    }

    layer_creator_func layer_creator = d->custom_layer_registry[index].creator;

    if(!layer_creator) {
        return 0;
    }

    Layer* layer = layer_creator(d->custom_layer_registry[index].userdata);

    layer->typeindex = ncnn::LayerType::CustomBit | index;

    return layer;
}

Layer* Net::create_custom_layer(const char* type)
{
    int index = custom_layer_to_index(type);

    if(index == -1) {
        return 0;
    }

    return create_custom_layer(index);
}

int Net::find_blob_index_by_name(const char* name) const
{
    for(size_t i = 0; i < d->blobs.size(); i++) {
        const Blob& blob = d->blobs[i];

        if(blob.name == name) {
            return static_cast<int>(i);
        }
    }

    printf("find_blob_index_by_name %s failed\n", name);

    return -1;
}

int Net::load_param(const DataReader& dr)
{
#define SCAN_VALUE(fmt, v)              \
    if(dr.scan(fmt, &v) != 1) {         \
        printf("parse " #v " failed");  \
        return -1;                      \
    }

    int magic = 0;

    SCAN_VALUE("%d", magic)

    if(magic != 7767517) {
        printf("param is too old, please regenerate\n");

        return -1;
    }

    // parse
    int layer_count = 0;
    int blob_count = 0;

    SCAN_VALUE("%d", layer_count)
    SCAN_VALUE("%d", blob_count)

    if(layer_count <= 0 || blob_count <= 0) {
        printf("invalid layer_count or blob_count\n");

        return -1;
    }

    d->layers.resize((size_t)layer_count);
    d->blobs.resize((size_t)blob_count);

    ParamDict pd;

    int blob_index = 0;

    for(int i = 0; i < layer_count; i++) {
        char layer_type[256];
        char layer_name[256];

        int bottom_count = 0;
        int top_count = 0;

        SCAN_VALUE("%255s", layer_type)
        SCAN_VALUE("%255s", layer_name)
        SCAN_VALUE("%d", bottom_count)
        SCAN_VALUE("%d", top_count)

        Layer* layer = create_layer(layer_type);

        if(!layer) {
            layer = create_custom_layer(layer_type);
        }

        if(!layer) {
            printf("layer %s not exists or registered\n", layer_type);
            clear();

            return -1;
        }

        layer->type = std::string(layer_type);
        layer->name = std::string(layer_name);

        layer->bottoms.resize(bottom_count);

        for(int j = 0; j < bottom_count; j++) {
            char bottom_name[256];

            SCAN_VALUE("%255s", bottom_name)

            int bottom_blob_index = find_blob_index_by_name(bottom_name);

            if(bottom_blob_index == -1) {
                Blob& blob = d->blobs[blob_index];

                bottom_blob_index = blob_index;

                blob_index++;
            }

            Blob& blob = d->blobs[bottom_blob_index];

            blob.consumer = i;

            layer->bottoms[j] = bottom_blob_index;
        }

        layer->tops.resize(top_count);

        for(int j = 0; j < top_count; j++) {
            Blob& blob = d->blobs[blob_index];

            char blob_name[256];

            SCAN_VALUE("%255s", blob_name);

            blob.name = std::string(blob_name);
            blob.producer = i;

            layer->tops[j] = blob_index;

            blob_index++;
        }

        // layer specific params
        int pdlr = pd.load_param(dr);

        if(pdlr != 0) {
            printf("ParamDict load_param %d %s failed\n", i, layer->name.c_str());

            continue;
        }

        if(layer->support_int8_storage) {
            // no int8 gpu support yet
            opt.use_vulkan_compute = false;
        }

        // pull out top shape hints
        Mat shape_hints = pd.get(30, Mat());

        if(!shape_hints.empty()) {
            const int* psh = shape_hints;

            for(int j = 0; j < top_count; j++) {
                Blob& blob = d->blobs[layer->tops[j]];

                int dims = psh[0];

                if(dims == 1) {
                    blob.shape = Mat(psh[1], (void*)0, 4u, 1);
                }

                if(dims == 2) {
                    blob.shape = Mat(psh[1], psh[2], (void*)0, 4u, 1);
                }

                if(dims == 3) {
                    blob.shape = Mat(psh[1], psh[2], psh[3], (void*)0, 4u, 1);
                }

                psh += 4;
            }
        }

        // set bottom and top shape hints
        layer->bottom_shapes.resize(bottom_count);

        for(int j = 0; j < bottom_count; j++) {
            layer->bottom_shapes[j] = d->blobs[layer->bottoms[j]].shape;
        }

        layer->top_shapes.resize(top_count);

        for(int j = 0; j < top_count; j++) {
            layer->top_shapes[j] = d->blobs[layer->tops[j]].shape;
        }

        int lr = layer->load_param(pd);

        if(lr != 0) {
            printf("layer load_param %d %s failed\n", i, layer->name.c_str());

            continue;
        }

        d->layers[i] = layer;
    }

    d->update_input_output_indexes();
    d->update_input_output_names();

#undef SCAN_VALUE

    return 0;
}

int Net::load_param(FILE* fp)
{
    DataReaderFromStdio dr(fp);

    return load_param(dr);
}

int Net::load_param(const char* protopath)
{
    FILE* fp = fopen(protopath, "rb");

    if(!fp) {
        printf("fopen %s failed\n", protopath);

        return -1;
    }

    int ret = load_param(fp);

    fclose(fp);

    return ret;
}

int Net::load_model(const DataReader& dr)
{
    if(d->layers.empty()) {
        printf("network graph not ready\n");

        return -1;
    }

    int layer_count = (int)d->layers.size();

    // load file
    int ret = 0;

    ModelBinFromDataReader mb(dr);

    for(int i = 0; i < layer_count; i++) {
        Layer* layer = d->layers[i];

        // Here we found inconsistent content in the parameter file.
        if(!layer) {
            printf("load_model error at layer %d, parameter file has inconsistent content.\n", i);

            ret = -1;

            break;
        }

        int lret = layer->load_model(mb);

        if(lret != 0) {
            printf("layer load_model %d %s failed\n", i, layer->name.c_str());

            ret = -1;

            break;
        }

        if(layer->support_int8_storage) {
            // not int8 gpu support yet
            opt.use_vulkan_compute = false;
        }
    }

    for(int i = 0; i < layer_count; i++) {
        Layer* layer = d->layers[i];

        Option opt1 = opt;

        int cret = layer->create_pipeline(opt1);

        if(cret != 0) {
            printf("layer create_pipeline %d %s failed\n", i, layer->name.c_str());

            ret = -1;

            break;
        }
    }

    if(opt.use_local_pool_allocator) {
        if(opt.blob_allocator == 0) {
            if(!d->local_blob_allocator) {
                d->local_blob_allocator = new PoolAllocator;
                d->local_blob_allocator->set_size_compare_ratio(0.f);
            }
        }

        if(opt.workspace_allocator == 0) {
            if(!d->local_workspace_allocator) {
                d->local_workspace_allocator = new PoolAllocator;
                d->local_workspace_allocator->set_size_compare_ratio(0.5f);
            }
        }
    }

    return ret;
}

int Net::load_model(FILE* fp)
{
    DataReaderFromStdio dr(fp);

    return load_model(dr);
}

int Net::load_model(const char* modelpath)
{
    FILE* fp = fopen(modelpath, "rb");

    if(!fp) {
        printf("fopen %s failed\n", modelpath);

        return -1;
    }

    int ret = load_model(fp);

    fclose(fp);

    return ret;
}

class ExtractorPrivate
{
public:
    ExtractorPrivate(const Net* _net)
        : net(_net)
    {

    }

    const Net* net;
    std::vector<Mat> blob_mats;
    Option opt;
};

Extractor::~Extractor()
{
    clear();

    delete d;
}

Extractor::Extractor(const Extractor& rhs)
    : d(new ExtractorPrivate(0))
{
    d->net = rhs.d->net;
    d->blob_mats = rhs.d->blob_mats;
    d->opt = rhs.d->opt;
}

Extractor& Extractor::operator=(const Extractor& rhs)
{
    if(this == &rhs) {
        return *this;
    }

    d->net = rhs.d->net;
    d->blob_mats = rhs.d->blob_mats;
    d->opt = rhs.d->opt;

    return *this;
}

void Extractor::clear()
{
    d->blob_mats.clear();
}

Extractor::Extractor(const Net* _net, size_t blob_count)
    : d(new ExtractorPrivate(_net))
{
    d->blob_mats.resize(blob_count);
    d->opt = d->net->opt;
}

int Extractor::input(const char* blob_name, const Mat& in)
{
    int blob_index = d->net->find_blob_index_by_name(blob_name);

    if(blob_index == -1) {
        printf("Try\n");

        const std::vector<const char*>& input_names = d->net->input_names();

        for(size_t i = 0; i < input_names.size(); i++) {
            printf("    ex.input(\"%s\", in%d);\n", input_names[i], (int)i);
        }

        return -1;
    }

    return input(blob_index, in);
}

int Extractor::input(int blob_index, const Mat& in)
{
    if(blob_index < 0 || blob_index >= (int)d->blob_mats.size()) {
        return -1;
    }

    d->blob_mats[blob_index] = in;

    return 0;
}

int Extractor::extract(const char* blob_name, Mat& feat, int type)
{
    int blob_index = d->net->find_blob_index_by_name(blob_name);

    if(blob_index == -1) {
        printf("Try\n");

        const std::vector<const char*>& output_names = d->net->output_names();

        for(size_t i = 0; i < output_names.size(); i++) {
            printf("    ex.extract(\"%s\", out%d);\n", output_names[i], (int)i);
        }

        return -1;
    }

    return extract(blob_index, feat, type);
}

int Extractor::extract(int blob_index, Mat& feat, int type)
{
    if(blob_index < 0 || blob_index >= (int)d->blob_mats.size()) {
        return -1;
    }

    int old_blocktime = get_kmp_blocktime();
    set_kmp_blocktime(d->opt.openmp_blocktime);

    int old_flush_denormals = get_flush_denormals();
    set_flush_denormals(d->opt.flush_denormals);

    int ret = 0;

    if(d->blob_mats[blob_index].dims == 0) {
        int layer_index = d->net->blobs()[blob_index].producer;

        // use local allocator
        if(d->opt.use_local_pool_allocator) {
            if(!d->opt.blob_allocator) {
                d->opt.blob_allocator = d->net->d->local_blob_allocator;
            }

            if(!d->opt.workspace_allocator) {
                d->opt.workspace_allocator = d->net->d->local_workspace_allocator;
            }
        }

        ret = d->net->d->forward_layer(layer_index, d->blob_mats, d->opt);
    }

    feat = d->blob_mats[blob_index];

    if(d->opt.use_packing_layout && (type == 0) && feat.elempack != 1) {
        Mat bottom_blob_unpacked;

        convert_packing(feat, bottom_blob_unpacked, 1, d->opt);

        feat = bottom_blob_unpacked;
    }

    if(feat.elembits() == 8 && (type == 0)) {
        Mat feat_fp32;

        cast_int8_to_float32(feat, feat_fp32, d->opt);

        feat = feat_fp32;
    }

    if(d->opt.use_local_pool_allocator && feat.allocator == d->net->d->local_blob_allocator) {
        // detach the returned mat from local pool allocator
        // so we could destroy net instance mush earlier
        feat = feat.clone();
    }

    set_kmp_blocktime(old_blocktime);
    set_flush_denormals(old_flush_denormals);

    return ret;
}
}