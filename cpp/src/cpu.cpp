#include "cpu.h"

#include <iostream>
#include <limits.h>
#include <cstring>
#include <vector>

namespace ncnn
{
CpuSet::CpuSet()
{
    disable_all();
}

void CpuSet::disable_all()
{
    CPU_ZERO(&cpu_set);
}

void CpuSet::enable(int cpu)
{
    CPU_SET(cpu, &cpu_set);
}

bool CpuSet::is_enabled(int cpu) const
{
    return CPU_ISSET(cpu, &cpu_set);
}

int CpuSet::num_enabled() const
{
    int num_enabled = 0;

    for(int i = 0; i < (int)sizeof(cpu_set_t) * 8; i++) {
        if(is_enabled(i)) {
            num_enabled++;
        }
    }
}

static int get_cpucount()
{
    int count = 0;

    FILE* fp = fopen("/proc/cpuinfo", "rb");

    if(!fp) {
        return 1;
    }

    char line[1024];

    while(!feof(fp)) {
        char* s = fgets(line, 1024, fp);

        if(!s) {
            break;
        }

        if(memcmp(line, "processor", 9) == 0) {
            count++;
        }
    }

    fclose(fp);

    if(count < 1) {
        count = 1;
    }

    return count;
}

static int g_cpucount = get_cpucount();

static int get_max_freq_khz(int cpuid)
{
    char path[256];

    sprintf(path, "/sys/devices/system/cpu/cpufreq/stats/cpu%d/time_in_state", cpuid);

    FILE* fp = fopen(path, "rb");

    if(!fp) {
        sprintf(path, "/sys/devices/system/cpu/cpu%d/cpufreq/stats/time_in_state", cpuid);

        fp = fopen(path, "rb");

        if(fp) {
            int max_freq_khz = 0;

            while(!feof(fp)) {
                int freq_khz = 0;
                int nscan = fscanf(fp, "%d %*d", &freq_khz);

                if(nscan != 1) {
                    break;
                }

                if(freq_khz > max_freq_khz) {
                    max_freq_khz = freq_khz;
                }
            }

            fclose(fp);

            if(max_freq_khz != 0) {
                return max_freq_khz;
            }

            fp = NULL;
        }

        if(!fp) {
            sprintf(path, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpuid);

            fp = fopen(path, "rb");

            if(!fp) {
                return -1;
            }

            int max_freq_khz = -1;
            int nscan = fscanf(fp, "%d", &max_freq_khz);

            if(nscan != 1) {
                printf("fscanf cpuinfo_max_freq error %d\n", nscan);
            }

            fclose(fp);

            return max_freq_khz;
        }
    }

    int max_freq_khz = 0;

    while(!feof(fp)) {
        int freq_khz = 0;
        int nscan = fscanf(fp, "%d %*d", &freq_khz);

        if(nscan != 1) {
            break;
        }

        if(freq_khz > max_freq_khz) {
            max_freq_khz = freq_khz;
        }
    }

    fclose(fp);

    return max_freq_khz;
}

static CpuSet g_thread_affinity_mask_all;
static CpuSet g_thread_affinity_mask_little;
static CpuSet g_thread_affinity_mask_big;

static int setup_thread_affinity_masks()
{
    g_thread_affinity_mask_all.disable_all();

    int max_freq_khz_min = INT_MAX;
    int max_freq_khz_max = 0;

    std::vector<int> cpu_max_freq_khz(g_cpucount);

    for(int i = 0; i < g_cpucount; i++) {
        int max_freq_khz = get_max_freq_khz(i);

        cpu_max_freq_khz[i] = max_freq_khz;

        if(max_freq_khz > max_freq_khz_max) {
            max_freq_khz_max = max_freq_khz;
        }

        if(max_freq_khz < max_freq_khz_min) {
            max_freq_khz_min = max_freq_khz;
        }
    }

    int max_freq_khz_medium = (max_freq_khz_min + max_freq_khz_max) / 2;

    if(max_freq_khz_medium == max_freq_khz_max) {
        g_thread_affinity_mask_little.disable_all();

        g_thread_affinity_mask_big = g_thread_affinity_mask_all;

        return 0;
    }

    for(int i = 0; i < g_cpucount; i++) {
        if(cpu_max_freq_khz[i] < max_freq_khz_medium) {
            g_thread_affinity_mask_little.enable(i);
        }
        else {
            g_thread_affinity_mask_big.enable(i);
        }
    }

    return 0;
}

const CpuSet& get_cpu_thread_affinity_mask(int powersave)
{
    setup_thread_affinity_masks();

    if(powersave == 0) {
        return g_thread_affinity_mask_all;
    }

    if(powersave == 1) {
        return g_thread_affinity_mask_little;
    }

    if(powersave == 2) {
        return g_thread_affinity_mask_big;
    }

    printf("powersave %d not supported\n", powersave);

    return g_thread_affinity_mask_all;
}

int get_big_cpu_count()
{
    int big_cpu_count = get_cpu_thread_affinity_mask(2).num_enabled();

    return big_cpu_count ? big_cpu_count : g_cpucount;
}

int cpu_support_arm_asimdhp()
{
    return 0;
}

int cpu_support_riscv_zfh()
{
    return 0;
}

int get_kmp_blocktime()
{
    return 0;
}

void set_kmp_blocktime(int time_ms)
{
    (void)time_ms;
}

int get_flush_denormals()
{
    return 0;
}

int set_flush_denormals(int flush_denormals)
{
    if(flush_denormals < 0 || flush_denormals > 3) {
        printf("denormals_zero %d not supported\n", flush_denormals);

        return -1;
    }

    return 0;
}

}