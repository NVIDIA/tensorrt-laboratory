/* Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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
#include "trtlab/core/affinity.h"

#include <algorithm>
#include <glog/logging.h>

#define test_bit(_n, _p) (_n & (1UL << _p))

namespace trtlab {

static cpuaff::affinity_manager s_Manager;
std::vector<int> ParseIDs(const std::string data);

void Affinity::SetAffinity(const CpuSet& cpus)
{
    CHECK(s_Manager.set_affinity(cpus)) << "SetAffinity failed for cpu_set: " << cpus;
}

CpuSet Affinity::GetAffinity()
{
    CpuSet cpus;
    CHECK(s_Manager.get_affinity(cpus)) << "GetAffinity failed";
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

cpuaff::cpu Affinity::GetCpuFromId(int id)
{
    cpuaff::cpu cpu;
    CHECK(s_Manager.get_cpu_from_id(cpu, id)) << "GetCpuById failed for cpu_id: " << id;
    return cpu;
}

CpuSet Affinity::GetCpusFromString(const std::string& ids)
{
    CpuSet cpus;
    auto int_ids = ParseIDs(ids);
    for(const auto id : int_ids)
    {
        cpus.insert(Affinity::GetCpuFromId(id));
        // cpus = cpus.Union(Affinity::GetCpuFromId(id));
    }
    return cpus;
}

CpuSet CpuSet::Intersection(const CpuSet& other) const
{
    CpuSet value;
    set_intersection(begin(), end(), other.begin(), other.end(),
                     std::inserter(value, value.begin()));
    return value;
}

CpuSet CpuSet::Union(const CpuSet& other) const
{
    CpuSet value;
    set_union(begin(), end(), other.begin(), other.end(), std::inserter(value, value.begin()));
    return value;
}

CpuSet CpuSet::Difference(const CpuSet& other) const
{
    CpuSet value;
    set_difference(begin(), end(), other.begin(), other.end(), std::inserter(value, value.begin()));
    return value;
}

std::string CpuSet::GetCpuString() const
{
    auto allocator = GetAllocator();
    std::ostringstream convert;
    for(size_t i = 0; i < allocator.size(); i++)
    {
        auto j = allocator.allocate();
        convert << j.id().get() << " ";
    }
    return convert.str();
}

int ConvertString2Int(const std::string& str)
{
    int x;
    std::stringstream ss(str);
    CHECK(ss >> x) << "Error converting " << str << " to integer";
    return x;
}

std::vector<std::string> SplitStringToArray(const std::string& str, char splitter)
{
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string temp;
    while(getline(ss, temp, splitter)) // split into new "lines" based on character
    {
        tokens.push_back(temp);
    }
    return tokens;
}

std::vector<int> ParseIDs(const std::string data)
{
    std::vector<int> result;
    std::vector<std::string> tokens = SplitStringToArray(data, ',');
    for(std::vector<std::string>::const_iterator it = tokens.begin(), end_it = tokens.end();
        it != end_it; ++it)
    {
        const std::string& token = *it;
        std::vector<std::string> range = SplitStringToArray(token, '-');
        if(range.size() == 1)
        {
            result.push_back(ConvertString2Int(range[0]));
        }
        else if(range.size() == 2)
        {
            int start = ConvertString2Int(range[0]);
            int stop = ConvertString2Int(range[1]);
            for(int i = start; i <= stop; i++)
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

} // namespace trtlab
