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
#include <cstdlib>
#include <memory>

#include <dlpack/dlpack.h>
#include <glog/logging.h>

namespace trtlab
{
    namespace memory
    {
        struct iallocator;

        class descriptor final
        {
            using storage_type = std::shared_ptr<iallocator>;

        public:
            descriptor();
            descriptor(std::shared_ptr<iallocator> alloc, std::size_t size, std::size_t alignment);

            descriptor(const descriptor& other) noexcept = delete;
            descriptor& operator=(const descriptor& other) noexcept = delete;

            descriptor(descriptor&& other) noexcept
            : m_storage(std::exchange(other.m_storage, nullptr)),
              m_size(std::exchange(other.m_size, 0u)),
              m_alignment(std::exchange(other.m_alignment, 0u)),
              m_data(std::exchange(other.m_data, nullptr))
            {
            }

            descriptor& operator=(descriptor&& other) noexcept
            {
                m_data      = std::exchange(other.m_data, nullptr);
                m_size      = std::exchange(other.m_size, 0u);
                m_alignment = std::exchange(other.m_alignment, 0u);
                m_storage   = std::exchange(other.m_storage, nullptr);
                return *this;
            }

            ~descriptor()
            {
                release();
            }

            void* data() noexcept
            {
                return m_data;
            }
            const void* data() const noexcept
            {
                return m_data;
            }
            std::size_t size() const noexcept
            {
                return m_size;
            };

            DLContext device_context() const;

            void release();

            std::shared_ptr<descriptor> make_shared();

        private:
            storage_type m_storage;
            std::size_t  m_size;
            std::size_t  m_alignment;
            void*        m_data;

            friend std::ostream& operator<<(std::ostream& os, const descriptor& md);
        };

        // clang-format off
        struct iallocator
        {
            virtual ~iallocator() = default;

            inline void* allocate(std::size_t size, std::size_t alignment = 0UL) { return do_allocate(size, alignment); }
            inline void deallocate(void* ptr, std::size_t size = 0UL, std::size_t alignment = 0UL) noexcept { do_deallocate(ptr, size, alignment); }
            inline descriptor allocate_descriptor(std::size_t size, std::size_t alignment = 0UL) { return do_allocate_descriptor(size, alignment); }

            inline std::size_t max_alignment() const { return do_max_alignment(); }
            inline std::size_t min_alignment() const { return do_min_alignment(); }
            inline std::size_t max_size() const { return do_max_size(); }

            inline DLContext device_context() const { return do_device_context();}

        private:
            virtual void*       do_allocate(std::size_t, std::size_t)                   = 0;
            virtual void        do_deallocate(void*, std::size_t, std::size_t) noexcept = 0;
            virtual descriptor  do_allocate_descriptor(std::size_t, std::size_t)        = 0;
            virtual std::size_t do_min_alignment() const                                = 0;
            virtual std::size_t do_max_alignment() const                                = 0;
            virtual std::size_t do_max_size() const                                     = 0;
            virtual DLContext   do_device_context() const                               = 0;
        };
        // clang-format on 

        std::ostream& operator<<(std::ostream& os, const descriptor& md);
    } // namespace memory
} // namespace trtlab