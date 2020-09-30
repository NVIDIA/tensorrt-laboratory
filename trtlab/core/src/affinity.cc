/* Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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
#include "affinity.h"

#include <algorithm>
#include <functional>

#include <glog/logging.h>
#include <boost/fiber/numa/topology.hpp>

#include "ranges.h"

#define test_bit(_n, _p) (_n & (1UL << _p))

using namespace trtlab;

static cpuaff::affinity_manager s_manager;
static std::string              cpu_int_string(const cpu_set& cpus, std::function<int(const cpuaff::cpu&)> extract);
static std::vector<int>         parse_ints(const std::string data);

cpu_set cpu_set::get_intersection(const cpu_set& other) const
{
    cpu_set cpus;
    set_intersection(begin(), end(), other.begin(), other.end(), std::inserter(cpus, cpus.begin()));
    return cpus;
}

cpu_set cpu_set::get_union(const cpu_set& other) const
{
    cpu_set cpus;
    set_union(begin(), end(), other.begin(), other.end(), std::inserter(cpus, cpus.begin()));
    return cpus;
}

cpu_set cpu_set::get_difference(const cpu_set& other) const
{
    cpu_set cpus;
    set_difference(begin(), end(), other.begin(), other.end(), std::inserter(cpus, cpus.begin()));
    return cpus;
}

cpu_set cpu_set::from_string(std::string ids)
{
    cpu_set cpus;
    auto    ints = parse_ints(ids);
    for (const auto id : ints)
    {
        cpus.insert(affinity::system::cpu_from_logical_id(id));
    }
    return cpus;
}

std::string cpu_set::cpus_string() const
{
    auto extract = [](const cpuaff::cpu& cpu) { return int(cpu.id().get()); };
    return cpu_int_string(*this, extract);
}

std::string cpu_set::cores_string() const
{
    auto extract = [](const cpuaff::cpu& cpu) { return int(cpu.core()); };
    return cpu_int_string(*this, extract);
}

std::string cpu_set::sockets_string() const
{
    auto extract = [](const cpuaff::cpu& cpu) { return int(cpu.socket()); };
    return cpu_int_string(*this, extract);
}

std::ostream& trtlab::operator<<(std::ostream& s, const cpu_set& cpus)
{
    s << "[cpu_set: cpus=" << cpus.cpus_string() << "; cores=" << cpus.cores_string() << "; sockets=" << cpus.sockets_string() << "]";
    return s;
}

std::ostream& trtlab::operator<<(std::ostream& s, const numa_node& node)
{
    s << "[numa_node: " << node.id << "; logical_cpus=" << node.cpus.cpus_string()
      << "; distances=(";
    for(const auto& d : node.distances)
    {
        s << " " << d;
    }
    s << " )]";
    return s;
}

std::vector<numa_node> affinity::system::topology()
{
    auto                   topo = boost::fibers::numa::topology();
    std::vector<numa_node> nodes;

    for (const auto& n : topo)
    {
        nodes.emplace_back();
        auto& node = nodes.back();

        // numa node id
        node.id = n.id;

        // logical cpus
        for (auto cpu : n.logical_cpus)
        {
            node.cpus.insert(affinity::system::cpu_from_logical_id(cpu));
        }

        // distance
        for (auto d : n.distance)
        {
            node.distances.push_back(d);
        }
    }
    return nodes;
}

void affinity::this_thread::set_affinity(const cpu_set& cpus)
{
    //DVLOG(1) << "Affinity: set thread affinity to: " << cpus;
    CHECK(s_manager.set_affinity(cpus)) << "SetAffinity failed for cpu_set: " << cpus;
}

cpu_set affinity::this_thread::get_affinity()
{
    cpu_set cpus;
    CHECK(s_manager.get_affinity(cpus)) << "GetAffinity failed";
    //DVLOG(1) << "Affinity: cpu affinity of calling thread: " << cpus;
    return cpus;
}

/*
cpu_set Affinity::GetCpusByNuma(int numa_id)
{
    cpuaff::cpu_set cpus;
    CHECK(s_manager.get_cpus_by_numa(cpus, numa_id)) << "GetCpusByNuma failed for numa_id: " << numa_id;
    return cpu_set(cpus);
}

cpu_set Affinity::GetCpusBySocket(int socket_id)
{
    cpuaff::cpu_set cpus;
    CHECK(s_manager.get_cpus_by_socket(cpus, socket_id)) << "GetCpusBySocket failed for socket_id: " << socket_id;
    return cpu_set(cpus);
}

cpu_set Affinity::GetCpusByCore(int core_id)
{
    cpuaff::cpu_set cpus;
    CHECK(s_manager.get_cpus_by_core(cpus, core_id)) << "GetCpusByCore failed for core_id: " << core_id;
    return cpu_set(cpus);
}

cpu_set Affinity::GetCpusByProcessingUnit(int thread_id)
{
    cpuaff::cpu_set cpus;
    CHECK(s_manager.get_cpus_by_processing_unit(cpus, thread_id)) << "GetCpusByProcessingUnit failed for thread_id: " << thread_id;
    return cpu_set(cpus);
}
*/

affinity_guard::affinity_guard()
{
    m_original_cpus = affinity::this_thread::get_affinity();
}
affinity_guard::affinity_guard(const cpu_set& cpus) : affinity_guard()
{
    CHECK(!cpus.empty());
    affinity::this_thread::set_affinity(cpus);
}

affinity_guard::~affinity_guard()
{
    affinity::this_thread::set_affinity(m_original_cpus);
}

// static implementations

cpuaff::cpu affinity::system::cpu_from_logical_id(int id)
{
    cpuaff::cpu cpu;
    CHECK(s_manager.get_cpu_from_id(cpu, id)) << "GetCpuById failed for cpu_id: " << id;
    return cpu;
}

int ConvertString2Int(const std::string& str)
{
    int               x;
    std::stringstream ss(str);
    CHECK(ss >> x) << "Error converting " << str << " to integer";
    return x;
}

std::vector<std::string> SplitStringToArray(const std::string& str, char splitter)
{
    std::vector<std::string> tokens;
    std::stringstream        ss(str);
    std::string              temp;
    while (getline(ss, temp, splitter)) // split into new "lines" based on character
    {
        tokens.push_back(temp);
    }
    return tokens;
}

std::vector<int> parse_ints(const std::string data)
{
    std::vector<int>         result;
    std::vector<std::string> tokens = SplitStringToArray(data, ',');
    for (std::vector<std::string>::const_iterator it = tokens.begin(), end_it = tokens.end(); it != end_it; ++it)
    {
        const std::string&       token = *it;
        std::vector<std::string> range = SplitStringToArray(token, '-');
        if (range.size() == 1)
        {
            result.push_back(ConvertString2Int(range[0]));
        }
        else if (range.size() == 2)
        {
            int start = ConvertString2Int(range[0]);
            int stop  = ConvertString2Int(range[1]);
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

std::string cpu_int_string(const cpu_set& cpus, std::function<int(const cpuaff::cpu&)> extract)
{
    std::set<int> uniques;

    for (const auto& cpu : cpus)
    {
        uniques.insert(extract(cpu));
    }

    auto ranges = find_ranges(std::vector<int>(uniques.begin(), uniques.end()));
    return print_ranges(ranges);
}

