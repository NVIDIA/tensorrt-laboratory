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
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <chrono>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <grpcpp/grpcpp.h>

#include "tensorrt/playground/core/memory/cyclic_allocator.h"
#include "tensorrt/playground/core/memory/system_v.h"

#include "echo.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using simple::Input;
using simple::Output;
using simple::Inference;

using playground::Memory::CyclicAllocator;
using playground::Memory::SystemV;

static constexpr size_t one_mb = 1024*1024;

DEFINE_int32(count, 1, "number of grpc messages to send");

class SimpleClient final {
 public:
  SimpleClient(std::shared_ptr<Channel> channel)
      : m_Stub(Inference::NewStub(channel)), m_Memory(5, one_mb) {}

  // Generate and send RPC message
  int Compute(const int batch_id) {

    // Allocate some SysV shared memory from the CyclicAllocator
    CyclicAllocator<SystemV>::Descriptor mdesc = RandomAllocation();

    // Populate the request object
    Input request;
    request.set_batch_id(batch_id);
    auto sysv = request.mutable_sysv();
    sysv->set_shm_id(mdesc->Stack().Memory().ShmID());
    sysv->set_offset(mdesc->Offset());
    sysv->set_size(mdesc->Size());

    // Write the batch_id to the shared memory segment
    // This will validated against the batch_id in the message body on the server
    auto data = mdesc->CastToArray<size_t>();
    data[0] = batch_id;
    data[1] = 0xDEADBEEF;

    // Container for the data we expect from the server.
    Output reply;

    // Context for the client. It could be used to convey extra information to
    // the server and/or tweak certain RPC behaviors.
    ClientContext context;

    // The actual RPC.
    Status status = m_Stub->Compute(&context, request, &reply);

    if (status.ok()) {
      CHECK_EQ(data[1], batch_id);
      return reply.batch_id();
    } else {
      LOG(ERROR) << status.error_code() << ": " << status.error_message();
      return -1;
    }
  }

 private:
  CyclicAllocator<SystemV>::Descriptor RandomAllocation() {
    size_t bytes = rand() % (m_Memory.MaxAllocationSize() / 4);
    bytes = std::max(bytes, 16UL); // guarantee at least 16 bytes (2x size_t)
    DLOG(INFO) << "RandomAllocation: " << bytes << " bytes";
    return m_Memory.Allocate(bytes);
  }

  std::unique_ptr<Inference::Stub> m_Stub;
  CyclicAllocator<SystemV> m_Memory;
};

int main(int argc, char** argv) 
{
  FLAGS_alsologtostderr = 1; // It will dump to console
  ::google::ParseCommandLineFlags(&argc, &argv, true);

  SimpleClient client(grpc::CreateChannel(
      "localhost:50051", grpc::InsecureChannelCredentials()));

  auto start = std::chrono::steady_clock::now();
  for(int i=0; i<FLAGS_count; i++) {
      auto reply = client.Compute(i);
      LOG_IF(INFO, reply == -1) << "BatchId received: " << reply;
  }
  auto end = std::chrono::steady_clock::now();
  float elapsed = std::chrono::duration<float>(end - start).count();
  std::cout << FLAGS_count << " requests in " << elapsed << "seconds" << std::endl;
  return 0;
}
