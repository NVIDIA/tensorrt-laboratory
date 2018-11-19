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
#ifndef _YAIS_TENSORRT_EXECUTIONCONTEXT_H_
#define _YAIS_TENSORRT_EXECUTIONCONTEXT_H_

#include "YAIS/Memory.h"
#include "YAIS/TensorRT/Bindings.h"

#include "NvInfer.h"

namespace yais
{
namespace TensorRT
{

/**
 * @brief Manages the execution of an inference calculation
 * 
 * The ExecutionContext is a limited quanity resource used to control the number
 * of simultaneous calculations allowed on the device at any given time.
 * 
 * A properly configured Bindings object is required to initiate the TensorRT
 * inference calculation.
 */
class ExecutionContext
{
  public:
    virtual ~ExecutionContext();

    DELETE_COPYABILITY(ExecutionContext);
    DELETE_MOVEABILITY(ExecutionContext);

    void SetContext(std::shared_ptr<::nvinfer1::IExecutionContext> context);
    void Infer(const std::shared_ptr<Bindings> &);
    auto Synchronize() -> double;

  private:
    ExecutionContext(size_t workspace_size);
    void Reset();

    std::function<double()> m_ElapsedTimer;
    cudaEvent_t m_ExecutionContextFinished;
    std::shared_ptr<::nvinfer1::IExecutionContext> m_Context;

    std::unique_ptr<CudaDeviceAllocator> m_Workspace;

    friend class ResourceManager;
};

} // namespace TensorRT
} // namespace yais

#endif // _YAIS_TENSORRT_EXECUTIONCONTEXT_H_