/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include "nvcxx/core/affinity.h"

#include <algorithm>
#include <glog/logging.h>
#include "nvml.h"

#define   test_bit(_n,_p)     ( _n & ( 1UL << _p))

namespace yais
{

struct nvmlState
{
    nvmlState() { CHECK_EQ(nvmlInit(), NVML_SUCCESS) << "Failed to initialize NVML"; }
    ~nvmlState() { CHECK_EQ(nvmlShutdown(), NVML_SUCCESS) << "Failed to Shutdown NVML"; }
};

static auto nvmlInstatnce = std::make_unique<nvmlState>();

static cpuaff::affinity_manager s_Manager;
std::vector<int> ParseIDs(const std::string data);

void Affinity::SetAffinity(const CpuSet cpus)
{
    CHECK(s_Manager.set_affinity(cpus))
        << "SetAffinity failed for cpu_set: " << cpus;
}

CpuSet Affinity::GetAffinity()
{
    CpuSet cpus;
    CHECK(s_Manager.get_affinity(cpus)) << "GetAffinity failed";
    return cpus;
}

CpuSet Affinity::GetDeviceAffinity(int device_id)
{
    CpuSet cpus;
    nvmlDevice_t gpu;
    unsigned long cpu_mask = 0;

    CHECK_EQ(nvmlDeviceGetHandleByIndex(device_id, &gpu), NVML_SUCCESS)
        << "Failed to get Device for index=" << device_id;

    CHECK_EQ(nvmlDeviceGetCpuAffinity(gpu, sizeof(cpu_mask), &cpu_mask), NVML_SUCCESS)
        << "Failed to retrieve CpusSet for GPU=" << device_id;

    for (int i=0; i<8*sizeof(cpu_mask); i++)
    {
        if (test_bit(cpu_mask, i)) {
            cpus = cpus.Union(Affinity::GetCpuFromId(i));
        }
    }

    DLOG(INFO) << "CPU Affinity for GPU " << device_id << ": " << cpus.GetCpuString();
    return cpus;
}

CpuSet Affinity::GetCpusByNuma(int numa_id)
{
    CpuSet cpus;
    CHECK(s_Manager.get_cpus_by_numa(cpus, numa_id))
        << "GetCpusByNuma failed for numa_id: " << numa_id;
    return cpus;
}

CpuSet Affinity::GetCpusBySocket(int socket_id)
{
    CpuSet cpus;
    CHECK(s_Manager.get_cpus_by_socket(cpus, socket_id))
        << "GetCpusBySocket failed for socket_id: " << socket_id;
    return cpus;
}

CpuSet Affinity::GetCpusByCore(int core_id)
{
    CpuSet cpus;
    CHECK(s_Manager.get_cpus_by_core(cpus, core_id))
        << "GetCpusByCore failed for core_id: " << core_id;
    return cpus;
}

CpuSet Affinity::GetCpusByProcessingUnit(int thread_id)
{
    CpuSet cpus;
    CHECK(s_Manager.get_cpus_by_processing_unit(cpus, thread_id))
        << "GetCpusByProcessingUnit failed for thread_id: " << thread_id;
    return cpus;
}

CpuSet Affinity::GetCpuFromId(int id)
{
    CpuSet cpus;
    cpuaff::cpu cpu;
    CHECK(s_Manager.get_cpu_from_id(cpu, id))
        << "GetCpuById failed for cpu_id: " << id;
    cpus.insert(cpu);
    return cpus;
}

CpuSet Affinity::GetCpusFromString(std::string ids)
{
    CpuSet cpus;
    auto int_ids = ParseIDs(ids);
    for (const auto id : int_ids)
    {
        cpus = cpus.Union(Affinity::GetCpuFromId(id));
    }
    return cpus;
}

CpuSet CpuSet::Intersection(const CpuSet &other) const
{
    CpuSet intersect_;
    set_intersection(begin(), end(), other.begin(), other.end(), std::inserter(intersect_, intersect_.begin()));
    return intersect_;
}

CpuSet CpuSet::Union(const CpuSet &other) const
{
    CpuSet union_;
    set_union(begin(), end(), other.begin(), other.end(), std::inserter(union_, union_.begin()));
    return union_;
}

CpuSet CpuSet::Difference(const CpuSet &other) const
{
    CpuSet diff_;
    set_difference(begin(), end(), other.begin(), other.end(), std::inserter(diff_, diff_.begin()));
    return diff_;
}

std::string CpuSet::GetCpuString() const
{
    auto allocator = GetAllocator();
    std::ostringstream convert;
    for (size_t i = 0; i < allocator.size(); i++)
    {
        auto j = allocator.allocate();
        convert << j.id().get() << " ";
    }
    return convert.str();
}

int ConvertString2Int(const std::string &str)
{
    int x;
    std::stringstream ss(str);
    CHECK(ss >> x) << "Error converting " << str << " to integer";
    return x;
}

std::vector<std::string> SplitStringToArray(const std::string &str, char splitter)
{
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string temp;
    while (getline(ss, temp, splitter)) // split into new "lines" based on character
    {
        tokens.push_back(temp);
    }
    return tokens;
}

std::vector<int> ParseIDs(const std::string data)
{
    std::vector<int> result;
    std::vector<std::string> tokens = SplitStringToArray(data, ',');
    for (std::vector<std::string>::const_iterator it = tokens.begin(), end_it = tokens.end(); it != end_it; ++it)
    {
        const std::string &token = *it;
        std::vector<std::string> range = SplitStringToArray(token, '-');
        if (range.size() == 1)
        {
            result.push_back(ConvertString2Int(range[0]));
        }
        else if (range.size() == 2)
        {
            int start = ConvertString2Int(range[0]);
            int stop = ConvertString2Int(range[1]);
            for (int i = start; i <= stop; i++)
            {
                result.push_back(i);
            }
        }
        else
        {
            LOG(FATAL) << "Error parsing token " << token;
        }
    }
    return result;
}

} // namespace yais
