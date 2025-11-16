#pragma once

#include <iostream>
#include <vector>

#include "option.h"

namespace ncnn
{
class NetPrivate;
class DataReader;
class Layer;
class Extractor;
class Mat;
class Blob;

class Net
{
public:
    // empty init
    Net();

    // clear and destroy
    virtual ~Net();

    virtual int custom_layer_to_index(const char* type);

    int load_param(const DataReader& dr);

    int load_model(const DataReader& dr);

    // load network structure from plain param file
    // return 0 if success
    int load_param(FILE* fp);
    int load_param(const char* protopath);

    // load network weight data from model file
    // return 0 if success
    int load_model(FILE* fp);
    int load_model(const char* modelpath);

    // unload network structure and weight data
    void clear();

    // construct an Extractor from network
    Extractor create_extractor() const;

    // get input / output indexes / names
    const std::vector<int>& input_indexes() const;
    const std::vector<int>& output_indexes() const;

    const std::vector<const char*>& input_names() const;
    const std::vector<const char*>& output_names() const;

    const std::vector<Blob>& blobs() const;

    // option can be changed before loading
    Option opt;

protected:
    friend class Extractor;

    int find_blob_index_by_name(const char* name) const;

    virtual Layer* create_custom_layer(const char* type);
    virtual Layer* create_custom_layer(int index);

private:
    NetPrivate* const d;
};

class ExtractorPrivate;
class Extractor
{
public:
    virtual ~Extractor();

    // copy
    Extractor(const Extractor&);

    // assign
    Extractor& operator=(const Extractor&);

    // clear blob mats and allocators
    void clear();

    // set input by blob name
    // return 0 if success
    int input(const char* blob_name, const Mat& in);

    // get result by blob name
    // return 0 if success
    // type = 0, default
    // type = 1, do not convert fp16 / bf16 or / and packing
    int extract(const char* blob_name, Mat& feat, int type = 0);

    // set input by blob index
    // return 0 if success
    int input(int blob_index, const Mat& in);

    // get result by blob index
    // return 0 if success
    // type = 0, default
    // type = 1, do not convert fp16 / bf16 or / and packing
    int extract(int blob_index, Mat& feat, int type = 0);

protected:
    friend Extractor Net::create_extractor() const;
    Extractor(const Net* net, size_t blob_count);

private:
    ExtractorPrivate* const d;
};
}