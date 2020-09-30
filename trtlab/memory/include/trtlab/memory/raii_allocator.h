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
#include <set>

#include <glog/logging.h>

#include "trtlab/memory/allocator.h"

namespace trtlab
{
    namespace memory
    {
        namespace raii_detail
        {
            template <typename Compare = std::less<>>
            struct compare_descriptor
            {
                using is_transparent = void;

                constexpr bool operator()(void* addr, const descriptor& md, Compare compare = Compare()) const
                {
                    return compare(reinterpret_cast<const addr_t>(addr), reinterpret_cast<const addr_t>(const_cast<void*>(md.data())));
                }

                constexpr bool operator()(const descriptor& md, void* addr, Compare compare = Compare()) const
                {
                    return compare(reinterpret_cast<const addr_t>(const_cast<void*>(md.data())), reinterpret_cast<const addr_t>(addr));
                }

                constexpr bool operator()(const descriptor& lhs, const descriptor& rhs, Compare compare = Compare()) const
                {
                    return compare(reinterpret_cast<const addr_t>(const_cast<void*>(lhs.data())),
                                   reinterpret_cast<const addr_t>(const_cast<void*>(rhs.data())));
                }
            };

            template <typename RawAllocator, typename Mutex>
            class raii_storage : public iallocator
            {
            public:
                using allocator_type = allocator<RawAllocator, Mutex>;
                using memory_type    = typename allocator_type::memory_type;
                using mutex          = typename allocator_type::mutex;
                using is_stateful    = std::true_type;

                raii_storage(const allocator_type& alloc) : m_allocator(alloc) {}
                ~raii_storage() = default;

                raii_storage(const raii_storage&) = delete;
                raii_storage& operator=(const raii_storage&) = delete;

                raii_storage(raii_storage&&) noexcept = default;
                raii_storage& operator=(raii_storage&&) noexcept = default;

                allocator_type& get_allocator()
                {
                    return m_allocator;
                }

                const allocator_type& get_allocator() const
                {
                    return m_allocator;
                }

            private:
                void* do_allocate(std::size_t size, std::size_t alignment) final override
                {
                    alignment = std::min(std::max(alignment, m_allocator.min_alignment()), m_allocator.max_alignment());
                    auto md = m_allocator.allocate_descriptor(size, alignment);

                    std::lock_guard<mutex> lock(m_mutex);
                    auto [it, rc] = m_descriptors.insert(std::move(md));
                    if (!rc)
                    {
                        LOG(FATAL) << "unable to hold internal descriptor";
                    }
                    return const_cast<void*>(it->data());
                }

                void do_deallocate(void* ptr, std::size_t size, std::size_t alignment) noexcept final override
                {
                    std::lock_guard<mutex> lock(m_mutex);

                    auto md = m_descriptors.find(ptr);
                    if (md == m_descriptors.end())
                    {
                        LOG(FATAL) << "cannot find matching internal descriptor";
                    }
                    m_descriptors.erase(md);
                }

                descriptor do_allocate_descriptor(std::size_t size, std::size_t alignment) final override
                {
                    return m_allocator.allocate_descriptor(size, alignment);
                }

                std::size_t do_max_alignment() const final override
                {
                    return m_allocator.max_alignment();
                }

                std::size_t do_min_alignment() const final override
                {
                    return m_allocator.min_alignment();
                }

                std::size_t do_max_size() const final override
                {
                    return m_allocator.max_node_size();
                }

                DLContext do_device_context() const final override
                {
                    return m_allocator.device_context();
                }

            private:
                mutable mutex                              m_mutex;
                allocator_type                             m_allocator;
                std::set<descriptor, compare_descriptor<>> m_descriptors;
            };

        } // namespace raii_detail

        template <typename RawAllocator, typename Mutex>
        using raii_allocator = allocator_detail::allocator_impl<raii_detail::raii_storage<RawAllocator, Mutex>>;

        template <typename RawAllocator, typename Mutex>
        auto make_raii_allocator(const allocator<RawAllocator, Mutex>& alloc)
        {
            auto storage = std::make_shared<raii_detail::raii_storage<RawAllocator, Mutex>>(alloc);
            return raii_allocator<RawAllocator, Mutex>(std::move(storage));
        }

    } // namespace memory
} // namespace trtlab