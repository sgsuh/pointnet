#pragma once

#include "mat.h"

// at most 32 parameters
#define NCNN_MAX_PARAM_COUNT    32

namespace ncnn
{
class DataReader;

class ParamDictPrivate;

class ParamDict
{
public:
    // empty
    ParamDict();

    virtual ~ParamDict();

    // get int
    int get(int id, int def) const;

    // get float
    float get(int id, float def) const;

    // get array
    Mat get(int id, const Mat& def) const;

    // set int
    void set(int id, int i);

    // set float
    void set(int id, float f);

    // set array
    void set(int id, const Mat& v);

protected:
    friend class Net;

    void clear();

    int load_param(const DataReader& dr);

private:
    ParamDictPrivate* const d;
};
}