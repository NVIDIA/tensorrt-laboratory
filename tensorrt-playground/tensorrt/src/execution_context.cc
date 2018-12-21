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
#include "tensorrt/playground/execution_context.h"

#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>

#include <glog/logging.h>

#include "tensorrt/playground/core/memory/allocator.h"
#include "tensorrt/playground/cuda/memory/cuda_device.h"

using yais::Memory::Allocator;
using yais::Memory::CudaDeviceMemory;

namespace yais
{
namespace TensorRT
{

/**
 * @brief TensorRT convenience class for wrapping IExecutionContext
 * 
 * Designed to launch, time and sychronized the execution of an async gpu calculation.
 *
 * Note: It is critically important that Reset() is called if the ExecutionContext is not going to
 * be immediately deleted after use.  Most high-performance implementation will maintain a queue/pool
 * of ExecutionContexts.  You mush call Reset() before returning the ExecutionContext to the pool or
 * risk deadlock / resource starvation.
 *  
 * Note: Timing is not implemented yet.  When implemented, do so a the host level and not the GPU
 * level to maximize GPU performance by using cudaEventDisabledTiming.  We don't need super accurate
 * timings, they are simply a nice-to-have, so a reasonable approximation on the host is sufficient.
 */
ExecutionContext::ExecutionContext(size_t workspace_size) 
  : m_Context{nullptr}, m_Workspace{std::make_unique<Allocator<CudaDeviceMemory>>(workspace_size)}
{
    CHECK_EQ(cudaEventCreateWithFlags(&m_ExecutionContextFinished, cudaEventDisableTiming), CUDA_SUCCESS)
        << "Failed to Create Execution Context Finished Event";
}

ExecutionContext::~ExecutionContext()
{
    CHECK_EQ(cudaEventDestroy(m_ExecutionContextFinished), CUDA_SUCCESS);
}

/**
 * @brief Set the ExectionContext
 * @param context 
 */
void ExecutionContext::SetContext(std::shared_ptr<::nvinfer1::IExecutionContext> context)
{
    m_Context = context;
    m_Context->setDeviceMemory(m_Workspace->Data());
}

/**
 * @brief Enqueue an Inference calculation
 * 
 * Initiates a forward pass through a TensorRT optimized graph and registers an event on the stream
 * which is trigged when the compute has finished and the ExecutionContext can be reused by competing threads.
 * Use the Synchronize method to sync on this event.
 * 
 * @param bindings 
 */
void ExecutionContext::Infer(const std::shared_ptr<Bindings> &bindings)
{
    DLOG(INFO) << "Launching Inference Execution";
    auto start = std::chrono::system_clock::now();
    m_ElapsedTimer = [start] { 
        return std::chrono::duration<double>(std::chrono::system_clock::now() - start).count();
    };
    m_Context->enqueue(bindings->BatchSize(), bindings->DeviceAddresses(), bindings->Stream(), nullptr);
    CHECK_EQ(cudaEventRecord(m_ExecutionContextFinished, bindings->Stream()), CUDA_SUCCESS);
}

/**
 * @brief Synchronized on the Completion of the Inference Calculation
 */
auto ExecutionContext::Synchronize() -> double
{
    CHECK_EQ(cudaEventSynchronize(m_ExecutionContextFinished), CUDA_SUCCESS);
    return m_ElapsedTimer();
}

/**
 * @brief Resets the Context
 * 
 * Note: It is very important to call Reset when you are finished using this object.  This is because
 * this object own a reference to the TensorRT IExecutionContext.  Failure to call Reset will maintain
 * a reference to the IExecutionContext, which could eventually starve resources and cause a deadlock.
 * 
 * See the implementation of TensorRT::Resources::GetExecutionContext for an example on how to use
 * with a yais::Pool<ExecutionContext>.
 */
void ExecutionContext::Reset()
{
    m_Context->setDeviceMemory(nullptr);
    m_Context.reset();
    m_ElapsedTimer = [] { return 0.0; };
}

} // namespace TensorRT
} // namespace yais