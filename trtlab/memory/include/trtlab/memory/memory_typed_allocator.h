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
#include <memory>

#include <glog/logging.h>

#include "trtlab/memory/allocator_storage.h"
#include "trtlab/memory/descriptor.h"

namespace trtlab
{
    namespace memory
    {
        template <typename MemoryType, DLDeviceType DeviceType>
        class memory_typed_allocator final
        {
            std::shared_ptr<iallocator> m_allocator;
            using allocator_type = memory_typed_allocator<MemoryType, DeviceType>;

        public:
            using memory_type = MemoryType;

            memory_typed_allocator(std::shared_ptr<iallocator> alloc) : m_allocator(alloc)
            {
                CHECK_EQ(m_allocator->device_context().device_type, DeviceType);
            }
            ~memory_typed_allocator() = default;

            memory_typed_allocator(const allocator_type&) = default;
            memory_typed_allocator& operator=(const allocator_type&) = default;

            memory_typed_allocator(allocator_type&&) = default;
            memory_typed_allocator& operator=(allocator_type&&) = default;

            void* allocate_node(std::size_t size, std::size_t alignment)
            {
                DCHECK(m_allocator);
                return m_allocator->allocate(size, alignment);
            }

            void deallocate_node(void* ptr, std::size_t size, std::size_t alignment)
            {
                DCHECK(m_allocator);
                return m_allocator->deallocate(ptr, size, alignment);
            }

            std::size_t max_node_size() const
            {
                DCHECK(m_allocator);
                return m_allocator->max_size();
            }

            std::size_t max_alignment() const
            {
                DCHECK(m_allocator);
                return m_allocator->max_alignment();
            }
            
            std::size_t min_alignment() const
            {
                DCHECK(m_allocator);
                return m_allocator->min_alignment();
            }

            DLContext device_context() const
            {
                DCHECK(m_allocator);
                return m_allocator->device_context();
            }

            std::shared_ptr<iallocator> shared() const
            {
                return m_allocator;
            }
        };

        using host_allocator = memory_typed_allocator<host_memory, kDLCPU>;
    } // namespace memory
} // namespace trtlab