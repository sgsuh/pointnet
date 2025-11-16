#pragma once

#include "mat.h"

namespace ncnn
{
class DataReader;

class ModelBin
{
public:
    ModelBin();
    virtual ~ModelBin();

    // element type
    // 0 = auto
    // 1 = float32
    // 2 = float16
    // 3 = int8
    // load vec
    virtual Mat load(int w, int type) const = 0;

    // load image
    virtual Mat load(int w, int h, int type) const;

    // load dim
    virtual Mat load(int w, int h, int c, int type) const;
};

class ModelBinFromDataReaderPrivate;

class ModelBinFromDataReader : public ModelBin
{
public:
    explicit ModelBinFromDataReader(const DataReader& dr);
    virtual ~ModelBinFromDataReader();

    virtual Mat load(int w, int type) const;

private:
    ModelBinFromDataReader(const ModelBinFromDataReader&);
    ModelBinFromDataReader& operator=(const ModelBinFromDataReader&);
    ModelBinFromDataReaderPrivate* const d;
};
};