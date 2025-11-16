#pragma once

#include <pthread.h>

namespace ncnn
{
class Mutex
{
public:
    Mutex()
    {
        pthread_mutex_init(&mutex, 0);
    }

    ~Mutex()
    {
        pthread_mutex_destroy(&mutex);
    }

    void lock()
    {
        pthread_mutex_lock(&mutex);
    }

    void unlock()
    {
        pthread_mutex_unlock(&mutex);
    }

private:
    pthread_mutex_t mutex;
};
}