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
#include "tensorrt/playground/allocator.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <glog/logging.h>

namespace playground {
namespace TensorRT {

void* NvAllocator::allocate(size_t size, uint64_t alignment, uint32_t flags)
{
    void* ptr;
    std::lock_guard<std::recursive_mutex> lock(m_Mutex);
    if(m_UseWeightAllocator)
    {
        WeightAllocator(&ptr, size);
        m_Pointers.push_back(Pointer{ptr, size});
    }
    else
    {
        CHECK_EQ(cudaMalloc(&ptr, size), CUDA_SUCCESS);
        DLOG(INFO) << "TensorRT cudaMalloc size = " << size;
    }
    return ptr;
}

void NvAllocator::free(void* ptr)
{
    DLOG(INFO) << "TensorRT cudaFree " << ptr;
    CHECK_EQ(cudaFree(ptr), CUDA_SUCCESS) << "Failed to free TensorRT device memory";
}

const std::vector<NvAllocator::Pointer>& NvAllocator::GetPointers()
{
    return m_Pointers;
}

void StandardAllocator::WeightAllocator(void** ptr, size_t size)
{
    CHECK_EQ(cudaMalloc(ptr, size), CUDA_SUCCESS);
    DLOG(INFO) << "TensorRT cudaMalloc size = " << size << "; " << *ptr;
}

void ManagedAllocator::WeightAllocator(void** ptr, size_t size)
{
    CHECK_EQ(cudaMallocManaged(ptr, size), CUDA_SUCCESS);
    DLOG(INFO) << "TensorRT cudaMallocManaged size = " << size << "; " << *ptr;
    CHECK_EQ(cudaMemAdvise(*ptr, size, cudaMemAdviseSetReadMostly, 0), CUDA_SUCCESS);
}

} // namespace TensorRT
} // namespace playground