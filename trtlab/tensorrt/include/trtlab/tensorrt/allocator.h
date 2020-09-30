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
#include <mutex>
#include <vector>

#include <NvInfer.h>

namespace trtlab
{
    namespace TensorRT
    {
        class NvAllocator : public ::nvinfer1::IGpuAllocator
        {
        public:
            struct Pointer
            {
                void*  addr;
                size_t size;
            };

            NvAllocator() : m_UseWeightAllocator(false) {}
            virtual ~NvAllocator() override {}

            NvAllocator(const NvAllocator&) = delete;
            NvAllocator& operator=(const NvAllocator&) = delete;

            NvAllocator(NvAllocator&&) noexcept = delete;
            NvAllocator& operator=(NvAllocator&&) noexcept = delete;

            // TensorRT IGpuAllocator virtual overrides
            void* allocate(uint64_t size, uint64_t alignment, uint32_t flags) final override;
            void  free(void* ptr) final override;

            template <class F, class... Args>
            auto use_weights_allocator(F&& f, Args&&... args) -> typename std::result_of<F(Args...)>::type
            {
                std::lock_guard<std::recursive_mutex> lock(m_Mutex);
                m_Pointers.clear();
                m_UseWeightAllocator = true;
                auto retval          = f(std::forward<Args>(args)...);
                m_UseWeightAllocator = false;
                m_Pointers.clear();
                return retval;
            }

            const std::vector<Pointer>& get_pointers();

        private:
            bool                 m_UseWeightAllocator;
            std::recursive_mutex m_Mutex;
            std::vector<Pointer> m_Pointers;
            virtual void         weights_allocate(void**, size_t) = 0;
        };

        class StandardAllocator : public NvAllocator
        {
            using NvAllocator::NvAllocator;
            void weights_allocate(void** ptr, size_t) final override;
        };

        class ManagedAllocator : public NvAllocator
        {
            using NvAllocator::NvAllocator;
            void weights_allocate(void** ptr, size_t) override;
        };

    } // namespace TensorRT
} // namespace trtlab