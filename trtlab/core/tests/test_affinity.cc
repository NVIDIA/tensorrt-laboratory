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

#include "glog/logging.h"
#include "gtest/gtest.h"

using namespace trtlab;

class TestAffinity : public ::testing::Test {};

TEST_F(TestAffinity, Basics)
{
    auto cpus = Affinity::GetAffinity();
    LOG(INFO) << cpus;
    LOG(INFO) << cpus.GetCpuString();

    // docker desktop uses a VM that screws up all affinity settings
    // this is similar to how aws vms disguise numa information
    // in both casses, each logical gpu shows up as a separate socket
    // and the numa_nodes is -1

    cpus = Affinity::GetCpusBySocket(0);
    LOG(INFO) << cpus;
    LOG(INFO) << "socket 0: " << cpus.GetCpuString();

    cpus = Affinity::GetCpusBySocket(1);
    LOG(INFO) << cpus;
    LOG(INFO) << "Socket 1: " << cpus.GetCpuString();
/*
    cpus = Affinity::GetCpusByNuma(0);
    LOG(INFO) << "numa 0: " << cpus.GetCpuString();

    cpus = Affinity::GetCpusByNuma(1);
    LOG(INFO) << "numa 1: " << cpus.GetCpuString();
*/
}

TEST_F(TestAffinity, IntString)
{
    std::set<int> ints = { 0, 1, 3, 4, 5, 9, 10 };
    std::string str = CpuSet::IntString(ints);
    CHECK_EQ(str, std::string("0-1,3-5,9-10"));

    ints = { 0 };
    str = CpuSet::IntString(ints);
    CHECK_EQ(str, std::string("0"));

    ints = { 1 };
    str = CpuSet::IntString(ints);
    CHECK_EQ(str, std::string("1"));

    ints = { 1,2,3 };
    str = CpuSet::IntString(ints);
    CHECK_EQ(str, std::string("1-3"));

    ints = { 1,2,3,4,5,7,6 };
    str = CpuSet::IntString(ints);
    CHECK_EQ(str, std::string("1-7"));

    ints = { 1,2,4,5,7,6 };
    str = CpuSet::IntString(ints);
    CHECK_EQ(str, std::string("1-2,4-7"));
}