#pragma once

#include <sched.h>

namespace ncnn
{
class CpuSet
{
public:
    CpuSet();

    void disable_all();
    void enable(int cpu);

    int num_enabled() const;

    bool is_enabled(int cpu) const;

    cpu_set_t cpu_set;
};

// test optional cpu features
// asimdhp = aarch64 asimd half precision
int cpu_support_arm_asimdhp();

// zfh = riscv half-precision float
int cpu_support_riscv_zfh();

// cpu info
int get_big_cpu_count();

// convenient wrapper
const CpuSet& get_cpu_thread_affinity_mask(int powersave);

int get_kmp_blocktime();
void set_kmp_blocktime(int time_ms);

// need to flush denormals on Intel Chipset.
// Other architectures such as ARM can be added as needed.
// 0 = DAZ OFF, FTZ OFF
// 1 = DAZ ON , FTZ OFF
// 2 = DAZ OFF, FTZ ON
// 3 = DAZ ON , FTZ ON
int get_flush_denormals();
int set_flush_denormals(int flush_denormals);
}