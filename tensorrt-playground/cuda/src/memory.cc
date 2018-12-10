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
#include "tensorrt/playground/cuda/memory.h"
#include "tensorrt/playground/cuda/device_info.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <glog/logging.h>

namespace yais
{
size_t DeviceMemory::DefaultAlignment() const
{
    return DeviceInfo::Alignment();
}

void DeviceMemory::Fill(char val)
{
    CHECK_EQ(cudaMemset(Data(), val, Size()), CUDA_SUCCESS);
}

const std::string& DeviceMemory::Type() const
{
    static std::string type = "DeviceMemory";
    return type;
}

// CudaManagedMemory
void* CudaManagedMemory::Allocate(size_t size)
{
    void* ptr;
    CHECK_EQ(cudaMallocManaged((void**)&ptr, size, cudaMemAttachGlobal), CUDA_SUCCESS);
    return ptr;
}

void CudaManagedMemory::Free()
{
    CHECK_EQ(cudaFree(Data()), CUDA_SUCCESS);
}

const std::string& CudaManagedMemory::Type() const
{
    static std::string type = "CudaMallocManaged";
    return type;
}

// CudaDeviceMemory

void* CudaDeviceMemory::Allocate(size_t size)
{
    void* ptr;
    CHECK_EQ(cudaMalloc((void**)&ptr, size), CUDA_SUCCESS);
    return ptr;
}

void CudaDeviceMemory::Free()
{
    CHECK_EQ(cudaFree(Data()), CUDA_SUCCESS);
}

const std::string& CudaDeviceMemory::Type() const
{
    static std::string type = "CudaMalloc";
    return type;
}

// CudaHostMemory

void* CudaHostMemory::Allocate(size_t size)
{
    void* ptr;
    CHECK_EQ(cudaMallocHost((void**)&ptr, size), CUDA_SUCCESS);
    return ptr;
}

void CudaHostMemory::Free()
{
    CHECK_EQ(cudaFreeHost(Data()), CUDA_SUCCESS);
}

const std::string& CudaHostMemory::Type() const
{
    static std::string type = "CudaMallocHost";
    return type;
}

} // namespace yais