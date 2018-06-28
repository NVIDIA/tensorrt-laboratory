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
#include "YAIS/Affinity.h"
#include "YAIS/ThreadPool.h"
#include "YAIS/Memory.h"
#include "YAIS/MemoryStack.h"
#include "YAIS/Pool.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <glog/logging.h>

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>

using yais::Affinity;
using yais::CpuSet;
using yais::CudaDeviceAllocator;
using yais::CudaHostAllocator;
using yais::MemoryStack;
using yais::Pool;
using yais::ThreadPool;

int main(int argc, char *argv[])
{
    auto one_gib = 1024 * 1024 * 1024;
    auto zeroMemory = true;

    auto gpu_0 = Affinity::GetDeviceAffinity(0);

    // Socket 0 - non-hyperthreads on a DGX-1 or Station
    auto socket_0 = Affinity::GetAffinity().Intersection(
        Affinity::GetCpusBySocket(0).Intersection(
            Affinity::GetCpusByProcessingUnit(0)));

    // Socket 1 - non-hyperthreads on a DGX-1, or
    // Socket 0 - hyperthreads on a DGX-Station
    auto socket_1 = Affinity::GetAffinity().Intersection(
        Affinity::GetCpusFromString("20-39"));

    auto workers_0 = std::make_shared<ThreadPool>(socket_0);
    auto workers_1 = std::make_shared<ThreadPool>(socket_1);

    std::shared_ptr<CudaHostAllocator> pinned_0, pinned_1;

    auto future_0 = workers_0->enqueue([=, &pinned_0] {
        pinned_0 = CudaHostAllocator::make_shared(one_gib);
        pinned_0->WriteZeros();
    });

    auto future_1 = workers_1->enqueue([=, &pinned_1] {
        pinned_1 = CudaHostAllocator::make_shared(one_gib);
        pinned_1->WriteZeros();
    });

    LOG(INFO) << socket_0;

    future_0.get();
    CHECK(pinned_0) << "pinned_0 got deallocated - fail";
    LOG(INFO) << "pinned_0 (ptr, size): ("
              << pinned_0->Data() << ", "
              << pinned_0->Size() << ")";
    future_1.get();

    std::shared_ptr<MemoryStack<CudaDeviceAllocator>> gpu_stack_on_socket0;
    std::shared_ptr<MemoryStack<CudaDeviceAllocator>> gpu_stack_on_socket1;

    future_0 = workers_0->enqueue([=, &gpu_stack_on_socket0] {
        CHECK_EQ(cudaSetDevice(0), CUDA_SUCCESS) << "Set Device 0 failed";
        gpu_stack_on_socket0 = MemoryStack<CudaDeviceAllocator>::make_shared(one_gib);
        gpu_stack_on_socket0->Reset(zeroMemory);
    });

    future_0.get(); // thread allocating gpu_stack_on_socket0 finished with task
    LOG(INFO) << "Push Binding 0 - 10MB - stack_ptr = "
              << gpu_stack_on_socket0->Allocate(10 * 1024 * 1024);
    LOG(INFO) << "Push Binding 1 - 128MB - stack_ptr = "
              << gpu_stack_on_socket0->Allocate(128 * 1024 * 1024);
    gpu_stack_on_socket0->Reset();

    struct Buffer
    {
        Buffer(
            std::shared_ptr<CudaHostAllocator> pinned_,
            std::shared_ptr<MemoryStack<CudaDeviceAllocator>> gpu_stack_,
            std::shared_ptr<ThreadPool> workers_) : pinned(pinned_), gpu_stack(gpu_stack_), workers(workers_) {}

        std::shared_ptr<CudaHostAllocator> pinned;
        std::shared_ptr<MemoryStack<CudaDeviceAllocator>> gpu_stack;
        std::shared_ptr<ThreadPool> workers;
    };

    auto buffers = Pool<Buffer>::Create();

    buffers->EmplacePush(new Buffer(pinned_0, gpu_stack_on_socket0, workers_0));
    buffers->EmplacePush(new Buffer(pinned_1, gpu_stack_on_socket1, workers_1));

    for (int i = 0; i < 6; i++)
    {
        auto buffer = buffers->Pop();
        buffer->workers->enqueue([buffer] {
            // perform some work - regardless of which buffer you got, you are working
            // on a thread properly assocated with the resources
            LOG(INFO) << Affinity::GetAffinity();
        });
    }

    workers_0.reset();
    workers_1.reset();

    std::this_thread::sleep_for(std::chrono::seconds(3));

    return 0;
}
