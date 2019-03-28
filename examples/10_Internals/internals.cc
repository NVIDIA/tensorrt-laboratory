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
#include "trtlab/core/memory/allocator.h"
#include "trtlab/core/memory/memory_stack.h"
#include "trtlab/core/pool.h"
#include "trtlab/core/thread_pool.h"
#include "trtlab/cuda/device_info.h"
#include "trtlab/cuda/memory/cuda_device.h"
#include "trtlab/cuda/memory/cuda_managed.h"
#include "trtlab/cuda/memory/cuda_pinned_host.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <glog/logging.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using trtlab::Affinity;
using trtlab::Allocator;
using trtlab::CpuSet;
using trtlab::CudaDeviceMemory;
using trtlab::CudaPinnedHostMemory;
using trtlab::DeviceInfo;
using trtlab::MemoryStack;
using trtlab::Pool;
using trtlab::ThreadPool;

int main(int argc, char* argv[])
{
    FLAGS_alsologtostderr = 1; // Log to console
    ::google::InitGoogleLogging("example10/internals.x");

    auto one_gib = 1024 * 1024 * 1024;
    auto zeroMemory = true;

    const auto& gpu_0 = DeviceInfo::Affinity(0);

    // Socket 0 - non-hyperthreads on a DGX-1 or Station
    const auto& socket_0 = Affinity::GetAffinity().Intersection(
        Affinity::GetCpusBySocket(0).Intersection(Affinity::GetCpusByProcessingUnit(0)));

    // Socket 1 - non-hyperthreads on a DGX-1, or
    // Socket 0 - hyperthreads on a DGX-Station
    const auto& socket_1 =
        Affinity::GetAffinity().Intersection(Affinity::GetCpusFromString("20-39"));

    auto workers_0 = std::make_shared<ThreadPool>(socket_0);
    auto workers_1 = std::make_shared<ThreadPool>(socket_1);

    std::shared_ptr<CudaPinnedHostMemory> pinned_0, pinned_1;

    auto future_0 = workers_0->enqueue([=, &pinned_0] {
        pinned_0 = std::make_shared<Allocator<CudaPinnedHostMemory>>(one_gib);
        pinned_0->Fill(0);
    });

    auto future_1 = workers_1->enqueue([=, &pinned_1] {
        pinned_1 = std::make_shared<Allocator<CudaPinnedHostMemory>>(one_gib);
        pinned_1->Fill(0);
    });

    LOG(INFO) << socket_0;

    future_0.get();
    CHECK(pinned_0) << "pinned_0 got deAllocator - fail";
    LOG(INFO) << "pinned_0 (ptr, size): (" << pinned_0->Data() << ", " << pinned_0->Size() << ")";
    future_1.get();

    std::shared_ptr<MemoryStack<CudaDeviceMemory>> gpu_stack_on_socket0;
    std::shared_ptr<MemoryStack<CudaDeviceMemory>> gpu_stack_on_socket1;

    // It's not strictly necessary to alloaction GPU memory from threads near the GPU
    // this just drives home the point that we want to align CPU worker thread to GPU affinity.
    future_0 = workers_0->enqueue([=, &gpu_stack_on_socket0] {
        CHECK_EQ(cudaSetDevice(0), CUDA_SUCCESS) << "Set Device 0 failed";
        gpu_stack_on_socket0 = std::make_shared<MemoryStack<CudaDeviceMemory>>(one_gib);
        gpu_stack_on_socket0->Reset(zeroMemory);
    });

    // On a dual-socket system, we could use workers_1 to allocation device memory.
    // Leaving this as an exercise to the reader.

    future_0.get(); // thread allocating gpu_stack_on_socket0 finished with task
    LOG(INFO) << "Push Binding 0 - 10MB - stack_ptr = "
              << gpu_stack_on_socket0->Allocate(10 * 1024 * 1024);
    LOG(INFO) << "Push Binding 1 - 128MB - stack_ptr = "
              << gpu_stack_on_socket0->Allocate(128 * 1024 * 1024);
    // Try allocating 1 byte. Notice how the memory is aligned. Default alignment
    // is defined by the MemoryType in Memory.h
    gpu_stack_on_socket0->Reset();

    /**
     * Create a Buffer object associates a worker threads, host memory and device memory
     * that are properly aligned to the hardware topology.
     */
    struct Buffer
    {
        Buffer(std::shared_ptr<CudaPinnedHostMemory> pinned_,
               std::shared_ptr<MemoryStack<CudaDeviceMemory>> gpu_stack_,
               std::shared_ptr<ThreadPool> workers_)
            : pinned(pinned_), gpu_stack(gpu_stack_), workers(workers_)
        {
        }

        std::shared_ptr<CudaPinnedHostMemory> pinned;
        std::shared_ptr<MemoryStack<CudaDeviceMemory>> gpu_stack;
        std::shared_ptr<ThreadPool> workers;

        // Normally, we'd associate some GPU index value to the buffer.
    };

    // Now create a Pool of Buffers
    auto buffers = Pool<Buffer>::Create();

    // Here we push two buffers, one for each socket.
    buffers->EmplacePush(new Buffer(pinned_0, gpu_stack_on_socket0, workers_0));
    buffers->EmplacePush(new Buffer(pinned_1, gpu_stack_on_socket1, workers_1));

    // Exercise: add more buffer objects.  Which of the three objects per Buffer
    // will you reuse, which will you make new instances of?

    // If you have arbituray work which is not necesasry topology aligned, say an incoming
    // inference request, you can pull a buffer object from the pool and queue work to the
    // proper set of threads best associated with that device
    for(int i = 0; i < 6; i++)
    {
        auto buffer = buffers->Pop();
        buffer->workers->enqueue([buffer] {
            // perform some work - regardless of which buffer you got, you are working
            // on a thread properly assocated with the resources
            LOG(INFO) << Affinity::GetAffinity();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        });
    }

    workers_0.reset();
    workers_1.reset();

    return 0;
}
