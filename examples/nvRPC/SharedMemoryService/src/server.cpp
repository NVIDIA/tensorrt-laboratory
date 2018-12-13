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
#include <chrono>
#include <map>
#include <memory>
#include <thread>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "nvrpc/executor.h"
#include "nvrpc/server.h"
#include "nvrpc/service.h"
#include "tensorrt/playground/core/memory.h"
#include "tensorrt/playground/core/pool.h"
#include "tensorrt/playground/core/resources.h"
#include "tensorrt/playground/core/thread_pool.h"

#include "echo.grpc.pb.h"
#include "echo.pb.h"

using yais::AsyncRPC;
using yais::AsyncService;
using yais::Context;
using yais::Executor;
using yais::Resources;
using yais::Server;
using yais::ThreadPool;

using yais::SystemV;

// CLI Options
DEFINE_int32(thread_count, 1, "Size of thread pool");

/**
 * @brief SystemV Memory Manager
 * 
 * This object does not allocate system v shared memory segments.  Instead, it attaches and manages
 * descriptors into shared memory segments allocated by an external source.
 */
class SystemVManager
{
  protected:
    class SystemVDescriptor : public SystemV
    {
      public:
        SystemVDescriptor(std::shared_ptr<const SystemV> memory, size_t offset, size_t size)
            : SystemV((*memory)[offset], size, false), m_Memory(memory)
        {
        }

        SystemVDescriptor(SystemVDescriptor&& other) : SystemV(std::move(other)) {}

      private:
        std::shared_ptr<const SystemV> m_Memory;
    };

  public:
    using Descriptor = std::unique_ptr<SystemVDescriptor>;

    SystemVManager() = default;

    Descriptor Acquire(size_t shm_id, size_t offset, size_t size)
    {
        std::unique_lock<std::mutex> l(m_Mutex);
        std::shared_ptr<const SystemV> memory;
        auto search = m_AttachedSegments.find(shm_id);
        if(search == m_AttachedSegments.end())
        {
            DLOG(INFO) << "SystemV Manager: attaching to shm_id: " << shm_id;
            memory = std::make_shared<SystemV>(shm_id);
            m_AttachedSegments[shm_id] = memory;
        }
        else
        {
            memory = search->second;
        }
        CHECK_LE(offset + size, memory->Size());
        return std::make_unique<SystemVDescriptor>(memory, offset, size);
    }

  private:
    std::map < size_t, std::shared_ptr<const SystemV>> m_AttachedSegments;
    std::mutex m_Mutex;
};



struct SimpleResources : public Resources
{
    SimpleResources(int numThreadsInPool = 3) : m_ThreadPool(numThreadsInPool) {}

    ThreadPool& GetThreadPool()
    {
        return m_ThreadPool;
    }

    SystemVManager& GetSystemVManager()
    {
        return m_SystemVManager;
    }

  private:
    ThreadPool m_ThreadPool;
    SystemVManager m_SystemVManager;
};


class SimpleContext final : public Context<simple::Input, simple::Output, SimpleResources>
{
    void ExecuteRPC(RequestType& input, ResponseType& output) final override
    {
        // We could do work here, but we'd block the TPS, i.e. the threads pulling messages
        // off the incoming recieve queue.  Very quick responses are best done here; however,
        // longer running workload should be offloaded so the TPS can avoid being blocked.
        // GetResources()->GetThreadPool().enqueue([this, &input, &output]{
        // Now running on a worker thread of the ThreadPool defined in SimpleResources.
        // Here we are just echoing back the incoming // batch_id; however, in later
        // examples, we'll show how to run an async cuda pipline.
        LOG_FIRST_N(INFO, 20) << "Tag = " << Tag() << " Thread = " << std::this_thread::get_id();
        SystemVManager::Descriptor mdesc;
        if(input.has_sysv()) {
            mdesc = GetResources()->GetSystemVManager().Acquire(
                input.sysv().shm_id(),
                input.sysv().offset(),
                input.sysv().size()
            );
        }
        mdesc->cast_to_array<size_t>()[0];
        LOG_IF(ERROR, mdesc->cast_to_array<size_t>()[0] != input.batch_id());
        output.set_batch_id(input.batch_id());
        this->FinishResponse();
        // });
        // The TPS thread is now free to continue processing message - async ftw!
    }
};

int main(int argc, char* argv[])
{
    FLAGS_alsologtostderr = 1; // Log to console

    ::google::InitGoogleLogging("simpleServer");
    ::google::ParseCommandLineFlags(&argc, &argv, true);

    // A server will bind an IP:PORT to listen on
    Server server("0.0.0.0:50051");

    // A server can host multiple services
    LOG(INFO) << "Register Service (simple::Inference) with Server";
    auto simpleInference = server.RegisterAsyncService<simple::Inference>();

    LOG(INFO) << "Register RPC (simple::Inference::Compute) with Service (simple::Inference)";
    auto rpcCompute = simpleInference->RegisterRPC<SimpleContext>(
        &simple::Inference::AsyncService::RequestCompute);

    LOG(INFO) << "Initializing Resources for RPC (simple::Inference::Compute)";
    auto rpcResources = std::make_shared<SimpleResources>();

    LOG(INFO) << "Creating Executor";
    auto executor = server.RegisterExecutor(new Executor(1));

    LOG(INFO) << "Creating Execution Contexts for RPC (simple::Inference::Compute) with Executor";
    executor->RegisterContexts(rpcCompute, rpcResources, 10);

    LOG(INFO) << "Running Server";
    server.Run(std::chrono::milliseconds(2000), [] {});
}
