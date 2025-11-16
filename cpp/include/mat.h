#pragma once

#include "allocator.h"
#include "option.h"

namespace ncnn
{
class Mat
{
public:
    // empty
    Mat();

    // vec
    Mat(int w, size_t elemsize = 4u, Allocator* allocator = 0);

    // image
    Mat(int w, int h, size_t elemsize = 4u, Allocator* allocator = 0);

    // packed cube
    Mat(int w, int h, int d, int c, size_t elemsize, int elempack, Allocator* allocator = 0);

    // copy
    Mat(const Mat& m);

    // external vec
    Mat(int w, void* data, size_t elemsize = 4u, Allocator* allocator = 0);

    // external packed vec
    Mat(int w, void* data, size_t elemsize, int elempack, Allocator* allocator = 0);

    // external packed image
    Mat(int w, int h, void* data, size_t elemsize, int elempack, Allocator* allocator = 0);

    // external packed dim
    Mat(int w, int h, int c, void* data, size_t elemsize, int elempack, Allocator* allocator = 0);

    // release
    ~Mat();

    // assign
    Mat& operator=(const Mat& m);

    // set all
    template<typename T>
    void fill(T v);

    // deep copy
    Mat clone(Allocator* allocator = 0) const;

    // reshape vec
    Mat reshape(int w, Allocator* allocator = 0) const;

    // reshape image
    Mat reshape(int w, int h, Allocator* allocator = 0) const;

    // reshape dim
    Mat reshape(int w, int h, int c, Allocator* allocator = 0) const;

    // reshape cube
    Mat reshape(int w, int h, int d, int c, Allocator* allocator = 0) const;

    // allocate vec
    void create(int w, size_t elemsize = 4u, Allocator* allocator = 0);

    // allocate image
    void create(int w, int h, size_t elemsize = 4u, Allocator* allocator = 0);

    // allocate packed image
    void create(int w, int h, size_t elemsize, int elempack, Allocator* allocator = 0);

    // allocate packed dim
    void create(int w, int h, int c, size_t elemsize, int elempack, Allocator* allocator = 0);

    // allocate packed cube
    void create(int w, int h, int d, int c, size_t elemsize, int elempack, Allocator* allocator = 0);

    // refcount++
    void addref();

    // refcount--
    void release();

    bool empty() const;
    size_t total() const;

    // bits per element
    int elembits() const;

    // data reference
    Mat channel(int c);
    const Mat channel(int c) const;

    const Mat depth(int z) const;

    float* row(int y);
    const float* row(int y) const;

    // access raw data
    template<typename T>
    operator T*();

    template<typename T>
    operator const T*() const;

    // convenient access float vec element
    float& operator[](size_t i);
    const float& operator[](size_t i) const;

    // convenient construct from half precision floating point data
    static Mat from_float16(const unsigned short* data, int size);

    // pointer to the data
    void* data;

    // pointer to the reference counter
    // when points to user-allocated data, the pointer is NULL
    int* refcount;

    // element size in bytes
    // 4 = float32 / int32
    // 2 = float16
    // 1 = int8 / uint8
    // 0 = empty
    size_t elemsize;

    // packed count inside element
    // c/1-d-h-w-1  c/1-h-w-1   h/1-w-1     w/1-1   scalar
    // c/4-d-h-w-4  c/4-h-w-4   h/4-w-4     w/4-4   sse/neon
    // c/8-d-h-w-8  c/8-h-w-8   h/8-w-8     w/8-8   avx/fp16
    int elempack;

    // the allocator
    Allocator* allocator;

    // the dimension rank
    int dims;

    int w;
    int h;
    int d;
    int c;

    size_t cstep;
};

// type conversion
// convert float to half precision floating point
unsigned short float32_to_float16(float value);

// convert half precision floating point to float
float float16_to_float32(unsigned short value);

// convert float to brain half
inline unsigned short float32_to_bfloat16(float value)
{
    // 16 : 16
    union
    {
        unsigned int u;
        float f;
    } tmp;

    tmp.f = value;

    return tmp.u >> 16;
}

// convert brain half to float
inline float bfloat16_to_float32(unsigned short value)
{
    // 16 : 16
    union
    {
        unsigned int u;
        float f;
    } tmp;

    tmp.u = value << 16;

    return tmp.f;
}

// mat process
enum BorderType
{
    BORDER_CONSTANT = 0,
    BORDER_REPLICATE = 1,
    BORDER_REFLECT = 2,
    BORDER_TRANSPARENT = -233,
};

void copy_make_border(const Mat& src, Mat& dst, int top, int bottom, int left, int right, int type, float v, const Option& opt = Option());
void convert_packing(const Mat& src, Mat& dst, int elempack, const Option& opt = Option());
void flatten(const Mat& src, Mat& dst, const Option& opt = Option());
void cast_int8_to_float32(const Mat& src, Mat& dst, const Option& opt = Option());

inline Mat::Mat()
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), d(0), c(0), cstep(0)
{

}

inline Mat::Mat(int _w, size_t _elemsize, Allocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), d(0), c(0), cstep(0)
{
    create(_w, _elemsize, _allocator);
}

inline Mat::Mat(int _w, int _h, size_t _elemsize, Allocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), d(0), c(0), cstep(0)
{
    create(_w, _h, _elemsize, _allocator);
}

inline Mat::Mat(int _w, int _h, int _d, int _c, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), d(0), c(0), cstep(0)
{
    create(_w, _h, _d, _c, _elemsize, _elempack, _allocator);
}

inline Mat::Mat(const Mat& m)
    : data(m.data), refcount(m.refcount), elemsize(m.elemsize), elempack(m.elempack), allocator(m.allocator), dims(m.dims), w(m.w), h(m.h), d(m.d), c(m.c), cstep(m.cstep)
{
    addref();
}

inline Mat::Mat(int _w, void* _data, size_t _elemsize, Allocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), dims(1), w(_w), h(1), d(1), c(1)
{
    cstep = w;
}

inline Mat::Mat(int _w, void* _data, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), dims(1), w(_w), h(1), d(1), c(1)
{
    cstep = w;
}

inline Mat::Mat(int _w, int _h, void* _data, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), dims(2), w(_w), h(_h), d(1), c(1)
{
    cstep = (size_t)w * h;
}

inline Mat::Mat(int _w, int _h, int _c, void* _data, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), dims(2), w(_w), h(_h), d(1), c(1)
{
    cstep = (size_t)w * h;
}

inline Mat::~Mat()
{
    release();
}

inline void Mat::addref()
{
    if(refcount) {
        NCNN_XADD(refcount, 1);
    }
}

inline void Mat::release()
{
    if(refcount && NCNN_XADD(refcount, -1) == 1) {
        if(allocator) {
            allocator->fastFree(data);
        }
        else {
            fastFree(data);
        }
    }

    data = 0;

    elemsize = 0;
    elempack = 0;
    
    dims = 0;
    w = 0;
    h = 0;
    d = 0;
    c = 0;

    cstep = 0;

    refcount = 0;
}

inline size_t Mat::total() const
{
    return cstep * c;
}

inline bool Mat::empty() const
{
    return data == 0 || total() == 0;
}

template<typename T>
inline Mat::operator T*()
{
    return (T*)data;
}

template<typename T>
inline Mat::operator const T*() const
{
    return (const T*)data;
}

inline float& Mat::operator[](size_t i)
{
    return ((float*)data)[i];
}

inline const float& Mat::operator[](size_t i) const
{
    return ((const float*)data)[i];
}

inline float* Mat::row(int y)
{
    return (float*)((unsigned char*)data + (size_t)w * y * elemsize);
}

inline const float* Mat::row(int y) const
{
    return (const float*)((unsigned char*)data + (size_t)w * y * elemsize);
}

inline Mat Mat::channel(int _c)
{
    Mat m(w, h, d, (unsigned char*)data + cstep * _c * elemsize, elemsize, elempack, allocator);

    m.dims = dims - 1;

    if(dims == 4) {
        m.cstep = (size_t)w * h;
    }

    return m;
}

inline const Mat Mat::channel(int _c) const
{
    Mat m(w, h, d, (unsigned char*)data + cstep * _c * elemsize, elemsize, elempack, allocator);

    m.dims = dims - 1;

    if(dims == 4) {
        m.cstep = (size_t)w * h;
    }

    return m;
}

inline Mat& Mat::operator=(const Mat& m)
{
    if(this == &m) {
        return *this;
    }

    if(m.refcount) {
        NCNN_XADD(m.refcount, 1);
    }

    release();

    data = m.data;
    refcount = m.refcount;
    elemsize = m.elemsize;
    elempack = m.elempack;
    allocator = m.allocator;

    dims = m.dims;
    w = m.w;
    h = m.h;
    d = m.d;
    c = m.c;

    cstep = m.cstep;

    return *this;
}

inline const Mat Mat::depth(int z) const
{
    return Mat(w, h, (unsigned char*)data + (size_t)w * h * z * elemsize, elemsize, elempack, allocator);
}

template<typename T>
inline void Mat::fill(T _v)
{
    int size = (int)total();
    T* ptr = (T*)data;

    for(int i = 0; i < size; i++) {
        ptr[i] = _v;
    }
}

inline int Mat::elembits() const
{
    return elempack ? static_cast<int>(elemsize * 8) / elempack : 0;
}
}