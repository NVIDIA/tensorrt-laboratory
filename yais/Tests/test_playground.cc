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
#include <iostream>
#include <string>
#include <vector>
#include <memory>

#include "glog/logging.h"

#include "YAIS/Memory.h"
#include "YAIS/Affinity.h"

using yais::Affinity;
using yais::CudaHostAllocator;
using yais::CudaDeviceAllocator;

struct Object
{
    Object(std::string name) : m_Name(name) {}
    Object(Object &&other) : m_Name(std::move(other.m_Name)) {}
    ~Object()
    {
        LOG(INFO) << "Destroying Object " << m_Name;
    }

    void SetName(std::string name) { m_Name = name; }
    std::string GetName() { return m_Name; }

  private:
    std::string m_Name;
};

void Test(const std::shared_ptr<Object> ptr)
{
    ptr->SetName("Bar");
}

int main(int argc, char *argv[])
{
    std::vector<std::unique_ptr<Object>> objects;

    Affinity affinity;

    auto socket_0_no_hyperthreads = Affinity::GetAffinity().Intersection(
        Affinity::GetCpusBySocket(0).Intersection(
            Affinity::GetCpusByProcessingUnit(0).Intersection(
                Affinity::GetCpusFromString("2,4-6")
        )));

    LOG(INFO) << socket_0_no_hyperthreads;

/*
    auto affinity = Affinity()::Create();
    auto socket0 = affinity->Split(affinity->GetSocket(0));
    auto socket1 = affinity->Split(affinity->GetSocket(1));
    CHECK_EQ(affinity.GetSize(), 0) << "More than 2 sockets?";
    auto service_cpus = socket0.GetExclusive().Union(socket1.Pop()).GetSharedAllocator();


    auto tp0 = std::make_unique<ThreadPool>(socket0.GetAvailable(), socket0.GetExclusiveAllocator());
    auto tp1 = std::make_unique<ThreadPool>(socket1.GetAvailable()*4, socket1.GetSharedAllocator());
*/
//  auto gpu_sptr = HostAllocator::make_shared(1024*1024*10);
//  LOG(INFO) << "shared " << gpu_sptr->GetSize();
//  auto gpu_uptr = HostAllocator::make_unique(1024*1024*10);
//  LOG(INFO) << "unique " << gpu_uptr->GetSize();

/*
    LOG(INFO) << "creating host mem stack";
    auto stack = yais::MemoryStack<HostAllocator>(1024*1024*10);
    LOG(INFO) << "host mem stack initialized";

    auto ptr01 = stack.Allocate(1024*1024*3);
    auto ptr02 = stack.Allocate(1024*1024*3);
    auto ptr03 = stack.Allocate(1024*1024*3);
    stack.ResetAllocations();
    auto ptr04 = stack.Allocate(1024*1024*5);
    auto ptr05 = stack.Allocate(1024*1024*5);

    // auto host = yais::MemoryStack<HostAllocator>(1024); */
    LOG(INFO) << "hi";
    auto host = yais::MemoryStack<CudaHostAllocator>(6946048);
    auto dev = yais::MemoryStack<CudaDeviceAllocator>(6946048);
    auto dev2 = yais::MemoryStack<CudaDeviceAllocator>(6946048);

    auto h1 = CudaHostAllocator::make_shared(6946048);
    auto d1 = CudaDeviceAllocator::make_shared(6946048);
    auto d2 = CudaDeviceAllocator::make_shared(6946048);
    auto d3 = CudaDeviceAllocator::make_unique(6946048);

    // C++17 should return a reference to the inserted by emplace_back
    // prior to 17, emplace_back return void
    // this fails on -std=gu++1z, not sure if it's 17 compliant for this fn
    // http://en.cppreference.com/w/cpp/container/vector/emplace_back
    // auto retval = objects.emplace_back(std::make_unique<Object>("Foo"));

    auto t = std::make_shared<Object>("Foo");
    Test(t);
    CHECK_EQ(t->GetName(), "Bar");
    return 0;
}
