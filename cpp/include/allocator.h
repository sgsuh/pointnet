#pragma once

#include <iostream>

namespace ncnn
{
// the alignment of all the allocated buffers
#define NCNN_MALLOC_ALIGN   16

// we have some optimized kernels that may overread buffer a bit in loop
// it is common to interleave next-loop data load with arithmetic instruction
// allocating more bytes keeps us safe from SEGV_ACCERR failure
#define NCNN_MALLOC_OVERREAD    64

// version for gcc >= 4.7
#define NCNN_XADD(addr, delta)  (int)__atomic_fetch_add((unsigned*)addr, (unsigned)(delta), __ATOMIC_ACQ_REL)

// Aligns a buffer size to the specified number of bytes
// The function returns the minimum number that is greater or equal to sz and is divisible by n
// sz Buffer size to align
// n Alignment size that must be a power of two
static inline size_t alignSize(size_t sz, int n)
{
    return (sz + n - 1) & -n;
}

static inline void* fastMalloc(size_t size)
{
    void* ptr = 0;

    if(posix_memalign(&ptr, NCNN_MALLOC_ALIGN, size + NCNN_MALLOC_OVERREAD)) {
        ptr = 0;
    }

    return ptr;
}

static inline void fastFree(void* ptr)
{
    if(ptr) {
        free(ptr);
    }
}

class Allocator
{
public:
    virtual ~Allocator();
    virtual void* fastMalloc(size_t size) = 0;
    virtual void fastFree(void* ptr) = 0;
};

class PoolAllocatorPrivate;

class PoolAllocator : public Allocator
{
public:
    PoolAllocator();
    ~PoolAllocator();

    // ratio range 0 ~ 1
    // default cr = 0.75
    void set_size_compare_ratio(float scr);

    // release all budgets immediately
    void clear();

    virtual void* fastMalloc(size_t size);
    virtual void fastFree(void* ptr);

private:
    PoolAllocator(const PoolAllocator&);
    PoolAllocator& operator=(const PoolAllocator&);
    PoolAllocatorPrivate* const d;
};
}