#include "allocator.h"

#include <list>

#include "platform.h"

namespace ncnn
{
Allocator::~Allocator()
{
    
}

class PoolAllocatorPrivate
{
public:
    unsigned int size_compare_ratio;    // 0 ~ 256

    Mutex budgets_lock;
    Mutex payouts_lock;

    std::list<std::pair<size_t, void*>> budgets;
    std::list<std::pair<size_t, void*>> payouts;
};

PoolAllocator::PoolAllocator()
    : Allocator(), d(new PoolAllocatorPrivate)
{
    d->size_compare_ratio = 192;
}

PoolAllocator::~PoolAllocator()
{
    clear();

    if(!d->payouts.empty()) {
        printf("FATAL ERROR! pool allocator destroyed too early\n");

        std::list<std::pair<size_t, void*>>::iterator it = d->payouts.begin();

        for(; it != d->payouts.end(); ++it) {
            void* ptr = it->second;

            printf("%p still in use\n", ptr);
        }
    }

    delete d;
}

PoolAllocator::PoolAllocator(const PoolAllocator&)
    : d(0)
{

}

PoolAllocator& PoolAllocator::operator=(const PoolAllocator&)
{
    return *this;
}

void PoolAllocator::clear()
{
    d->budgets_lock.lock();

    std::list<std::pair<size_t, void*>>::iterator it = d->budgets.begin();

    for(; it != d->budgets.end(); ++it) {
        void* ptr = it->second;

        ncnn::fastFree(ptr);
    }

    d->budgets.clear();
    d->budgets_lock.unlock();
}

void* PoolAllocator::fastMalloc(size_t size)
{
    d->budgets_lock.lock();

    // find free budget
    std::list<std::pair<size_t, void*>>::iterator it = d->budgets.begin();

    for(; it != d->budgets.end(); ++it) {
        size_t bs = it->first;

        // size_compare_ratio ~ 100%
        if(bs >= size && ((bs * d->size_compare_ratio) >> 8) <= size) {
            void* ptr = it->second;

            d->budgets.erase(it);
            d->budgets_lock.unlock();
            d->payouts_lock.lock();
            d->payouts.push_back(std::make_pair(bs, ptr));
            d->payouts_lock.unlock();

            return ptr;
        }
    }

    d->budgets_lock.unlock();

    // new
    void* ptr = ncnn::fastMalloc(size);

    d->payouts_lock.lock();
    d->payouts.push_back(std::make_pair(size, ptr));
    d->payouts_lock.unlock();

    return ptr;
}

void PoolAllocator::fastFree(void* ptr)
{
    d->payouts_lock.lock();

    // return to budgets
    std::list<std::pair<size_t, void*>>::iterator it = d->payouts.begin();

    for(; it != d->payouts.end(); ++it) {
        if(it->second == ptr) {
            size_t size = it->first;

            d->payouts.erase(it);
            d->payouts_lock.unlock();
            d->budgets_lock.lock();
            d->budgets.push_back(std::make_pair(size, ptr));
            d->budgets_lock.unlock();

            return;
        }
    }

    d->payouts_lock.unlock();

    printf("FATAL ERROR! pool allocator get wild %p\n", ptr);
    ncnn::fastFree(ptr);
}

void PoolAllocator::set_size_compare_ratio(float scr)
{
    if(scr < 0.f || scr > 1.f) {
        printf("invalid size compare ratio %f\n", scr);

        return;
    }

    d->size_compare_ratio = (unsigned int)(scr * 256);
}
}