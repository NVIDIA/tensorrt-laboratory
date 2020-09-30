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
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <trtlab/memory/allocator.h>
#include <trtlab/memory/allocator_traits.h>

#include "device_memory.h"
#include "../device_guard.h"

namespace trtlab
{
    namespace memory
    {
        namespace cuda_detail
        {
            struct cuda_malloc
            {
                using memory_type = device_memory;
                static cudaError_t allocate(void** ptr, std::size_t size)
                {
                    return cudaMalloc(ptr, size);
                }
                static cudaError_t deallocate(void* ptr) noexcept
                {
                    return cudaFree(ptr);
                }
                static allocator_info info()
                {
                    return {"cudaMalloc", nullptr};
                }
            };

            struct cuda_malloc_managed
            {
                using memory_type = device_managed_memory;
                static cudaError_t allocate(void** ptr, std::size_t size)
                {
                    return cudaMallocManaged(ptr, size);
                }
                static cudaError_t deallocate(void* ptr) noexcept
                {
                    return cudaFree(ptr);
                }
                static allocator_info info()
                {
                    return {"cudaMallocManaged", nullptr};
                }
            };

            struct cuda_malloc_host
            {
                using memory_type = host_pinned_memory;
                static cudaError_t allocate(void** ptr, std::size_t size)
                {
                    return cudaMallocHost(ptr, size);
                }
                static cudaError_t deallocate(void* ptr) noexcept
                {
                    return cudaFreeHost(ptr);
                }
                static allocator_info info()
                {
                    return {"cudaMallocHost", nullptr};
                }
            };

            template <typename CudaAllocator>
            struct generic_allocator
            {
                using is_stateful = std::false_type;
                using memory_type = typename CudaAllocator::memory_type;

                static void* allocate_node(std::size_t size, std::size_t)
                {
                    void* addr = nullptr;
                    auto  rc   = CudaAllocator::allocate((void**)&addr, size);
                    if (rc != cudaSuccess)
                    {
                        LOG(ERROR) << info().name << " failed to allocate " << size;
                        throw std::bad_alloc();
                    }
                    return addr;
                }

                static void deallocate_node(void* ptr, std::size_t, std::size_t) noexcept
                {
                    CHECK_EQ(CudaAllocator::deallocate(ptr), cudaSuccess) << "freeing " << info().name << ": " << ptr;
                }

                // note: requires implementation
                static allocator_info info()
                {
                    return CudaAllocator::info();
                }
            };
        } // namespace cuda_detail

        using cuda_malloc         = cuda_detail::generic_allocator<cuda_detail::cuda_malloc>;
        using cuda_malloc_managed = cuda_detail::generic_allocator<cuda_detail::cuda_malloc_managed>;
        using cuda_malloc_host    = cuda_detail::generic_allocator<cuda_detail::cuda_malloc_host>;

        namespace cuda_detail
        {
            template <typename CudaAllocator>
            class device_allocator
            {
                static_assert(std::is_base_of<device_memory, typename CudaAllocator::memory_type>::value, "should be device_memory");

            public:
                using allocator_type = device_allocator<CudaAllocator>;
                using is_stateful    = std::true_type;
                using memory_type    = typename CudaAllocator::memory_type;

                device_allocator(int device_id) : m_DeviceID(device_id) {}
                virtual ~device_allocator() {}

                device_allocator(allocator_type&&)      = default;
                device_allocator(const allocator_type&) = default;

                allocator_type& operator=(allocator_type&&) = default;
                allocator_type& operator=(const allocator_type&) = default;

                void* allocate_node(std::size_t size, std::size_t)
                {
                    device_guard guard(m_DeviceID);
                    return CudaAllocator::allocate_node(size, 0);
                }

                void deallocate_node(void* ptr, std::size_t, std::size_t) noexcept
                {
                    CudaAllocator::deallocate_node(ptr, 0, 0);
                }

                bool operator==(const allocator_type& other)
                {
                    return m_DeviceID == other.DeviceID;
                }

            private:
                int m_DeviceID;
            };
        } // namespace cuda_detail

        using cuda_malloc_allocator         = cuda_detail::device_allocator<cuda_malloc>;
        using cuda_malloc_managed_allocator = cuda_detail::device_allocator<cuda_malloc_managed>;
        using cuda_malloc_host_allocator    = cuda_malloc_host;

        static auto make_cuda_allocator(int device_id = -1)
        {
            if (device_id == -1)
            {
                CHECK_EQ(cudaGetDevice(&device_id), cudaSuccess);
            }
            return memory::make_allocator(cuda_malloc_allocator(device_id));
        };

    } // namespace memory
} // namespace trtlab