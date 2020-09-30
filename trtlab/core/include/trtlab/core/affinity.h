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
#pragma once
#include <vector>

#include "cpuaff/cpuaff.hpp"

namespace trtlab
{
    struct affinity;

    class cpu_set final : public cpuaff::cpu_set
    {
    public:
        using cpuaff::cpu_set::cpu_set;

        cpu_set get_intersection(const cpu_set& other) const;
        cpu_set get_union(const cpu_set& other) const;
        cpu_set get_difference(const cpu_set& other) const;

        static cpu_set from_string(std::string);

        std::string cpus_string() const;
        std::string cores_string() const;
        std::string sockets_string() const;

        auto get_allocator() const -> cpuaff::round_robin_allocator
        {
            return cpuaff::round_robin_allocator(*this);
        };

        friend std::ostream& operator<<(std::ostream& s, const cpu_set& cpus);
    };

    std::ostream& operator<<(std::ostream& s, const cpu_set& cpus);

    class affinity_guard final
    {
        // hold the original affinity of the calling thread
        // the original affinity will be restored on destruction
        cpu_set m_original_cpus;

    public:
        affinity_guard();
        explicit affinity_guard(const cpu_set&);
        ~affinity_guard();

        affinity_guard(const affinity_guard&) = delete;
        affinity_guard& operator=(const affinity_guard&) = delete;

        affinity_guard(affinity_guard&&) noexcept = delete;
        affinity_guard& operator=(affinity_guard&&) noexcept = delete;
    };

    struct numa_node
    {
        unsigned              id;
        cpu_set               cpus;
        std::vector<unsigned> distances;

        friend std::ostream& operator<<(std::ostream& s, const numa_node& cpus);
    };

    std::ostream& operator<<(std::ostream& s, const numa_node& cpus);

    struct affinity final
    {
        struct this_thread final
        {
            static cpu_set get_affinity();
            static void    set_affinity(const cpu_set&);
        };

        struct system final
        {
            // static cpu_set cpus_by_numa(int numa_id);
            // static cpu_set cpus_by_socket(int socket_id);
            // static cpu_set cpus_by_core(int core_id);
            // static cpu_set cpus_by_hyperthread(int thread_id);

            static cpuaff::cpu cpu_from_logical_id(int id);

            static std::vector<numa_node> topology();
        };
    };

} // namespace trtlab
