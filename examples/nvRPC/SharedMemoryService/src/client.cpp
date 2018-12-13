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
#include <memory>
#include <string>
#include <chrono>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <grpcpp/grpcpp.h>

#include "tensorrt/playground/core/cyclic_allocator.h"
#include "tensorrt/playground/core/memory.h"

#include "echo.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using simple::Input;
using simple::Output;
using simple::Inference;

using yais::CyclicAllocator;
using yais::SystemV;

static constexpr size_t one_mb = 1024*1024;

class SimpleClient {
 public:
  SimpleClient(std::shared_ptr<Channel> channel)
      : stub_(Inference::NewStub(channel)), m_Memory(5, one_mb) {}

  // Assembles the client's payload, sends it and presents the response back
  // from the server.
  int Compute(const int batch_id) {
    // Data we are sending to the server.

    CyclicAllocator<SystemV>::Descriptor mem = RandomAllocation();
    DLOG(INFO) << "SysV Info - ShmID: " << mem->Stack().Memory().ShmID();
    auto data = mem->cast_to_array<size_t>();
    data[0] = batch_id;

    Input request;
    request.set_batch_id(batch_id);
    auto sysv = request.mutable_sysv();
    sysv->set_shm_id(mem->Stack().Memory().ShmID());
    sysv->set_offset(mem->Offset());
    sysv->set_size(mem->Size());

    // Container for the data we expect from the server.
    Output reply;

    // Context for the client. It could be used to convey extra information to
    // the server and/or tweak certain RPC behaviors.
    ClientContext context;

    // The actual RPC.
    Status status = stub_->Compute(&context, request, &reply);

    // Act upon its status.
    if (status.ok()) {
      return reply.batch_id();
    } else {
      std::cout << status.error_code() << ": " << status.error_message()
                << std::endl;
      return -1;
    }
  }

 private:
  typename CyclicAllocator<SystemV>::Descriptor RandomAllocation() {
    size_t bytes = rand() % (m_Memory.MaxAllocationSize() / 4);
    DLOG(INFO) << "RandomAllocation: " << bytes << " bytes";
    return m_Memory.Allocate(bytes);
  }

  std::unique_ptr<Inference::Stub> stub_;
  CyclicAllocator<SystemV> m_Memory;
};

DEFINE_int32(count, 1, "number of grpc messages to send");

int main(int argc, char** argv) {
  // Instantiate the client. It requires a channel, out of which the actual RPCs
  // are created. This channel models a connection to an endpoint (in this case,
  // localhost at port 50051). We indicate that the channel isn't authenticated
  // (use of InsecureChannelCredentials()).
  FLAGS_alsologtostderr = 1; // It will dump to console
   ::google::ParseCommandLineFlags(&argc, &argv, true);

  SimpleClient client(grpc::CreateChannel(
      "localhost:50051", grpc::InsecureChannelCredentials()));
  auto start = std::chrono::steady_clock::now();
  for(int i=0; i<FLAGS_count; i++) {
      auto reply = client.Compute(i);
      if(reply == -1 || reply != i) std::cout << "BatchId received: " << reply << std::endl;
  }
  auto end = std::chrono::steady_clock::now();
  float elapsed = std::chrono::duration<float>(end - start).count();
  std::cout << FLAGS_count << " requests in " << elapsed << "seconds" << std::endl;
  return 0;
}
