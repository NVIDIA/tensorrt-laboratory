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
        namespace allocator_detail
        {
            template <typename RawAllocator, typename Mutex>
            class smart_storage : public iallocator,
                                  public std::enable_shared_from_this<smart_storage<RawAllocator, Mutex>>,
                                  private allocator_storage<direct_storage<RawAllocator>, Mutex>
            {
                using storage_type = allocator_storage<direct_storage<RawAllocator>, Mutex>;

            public:
                using allocator_type = typename storage_type::allocator_type;
                using memory_type    = typename storage_type::memory_type;
                using mutex          = typename storage_type::mutex;

                explicit smart_storage(storage_type&& storage) : storage_type(std::move(storage)),
                  m_min_alignment(storage_type::min_alignment()),
                  m_max_alignment(storage_type::max_alignment()) {}

                smart_storage(smart_storage&& other)
                : storage_type(std::move(other)),
                  m_min_alignment(storage_type::min_alignment()),
                  m_max_alignment(storage_type::max_alignment())
                {
                }
                smart_storage& operator=(smart_storage&& other)
                {
                    storage_type::operator=(std::move(other));
                    m_min_alignment = other.min_alignment();
                    m_max_alignment = other.max_alignment();
                    return *this;
                }

                ~smart_storage() override {}

                allocator_type& get_allocator() noexcept
                {
                    return storage_type::get_allocator();
                }

                const allocator_type& get_allocator() const noexcept
                {
                    return storage_type::get_allocator();
                }

                // disambiguate method in both iallocator and storage_type

                using iallocator::allocate_descriptor;
                using iallocator::device_context;
                using iallocator::max_alignment;
                using iallocator::min_alignment;

            private:
                inline void* do_allocate(std::size_t size, std::size_t alignment) final override
                {
                    return storage_type::allocate_node(size, alignment);
                }

                inline void do_deallocate(void* ptr, std::size_t size, std::size_t alignment) noexcept final override
                {
                    storage_type::deallocate_node(ptr, size, alignment);
                }

                inline descriptor do_allocate_descriptor(std::size_t size, std::size_t alignment) final override
                {
                    alignment = std::min(std::max(alignment, m_min_alignment), m_max_alignment);
                    return descriptor(std::move(this->shared_from_this()), size, alignment);
                }

                inline std::size_t do_max_alignment() const final override
                {
                    return m_max_alignment;
                }

                inline std::size_t do_min_alignment() const final override
                {
                    return m_min_alignment;
                }

                inline std::size_t do_max_size() const final override
                {
                    return storage_type::max_node_size();
                }

                inline DLContext do_device_context() const final override
                {
                    return storage_type::device_context();
                }

                std::size_t m_min_alignment;
                std::size_t m_max_alignment;
            };

            template <typename Mutex, typename RawAllocator>
            auto make_allocator_storage(RawAllocator&& alloc)
            {
                auto storage = allocator_storage<direct_storage<RawAllocator>, Mutex>(std::move(alloc));
                return std::make_shared<smart_storage<RawAllocator, Mutex>>(std::move(storage));
            }

            template <typename StorageType>
            class allocator_impl
            {
                using storage_type = StorageType;

                std::shared_ptr<storage_type> m_storage;

            public:
                using allocator_type = typename storage_type::allocator_type;
                using memory_type    = typename storage_type::memory_type;
                using mutex          = typename storage_type::mutex;
                using is_stateful    = std::true_type;

                explicit allocator_impl(std::shared_ptr<storage_type> storage) : m_storage(storage) 
                {
                    static_assert(std::is_base_of<iallocator, storage_type>::value, "storage must be derived from iallocator");
                }
                virtual ~allocator_impl() = default;

                allocator_impl(const allocator_impl<StorageType>& other) = default;
                allocator_impl& operator=(const allocator_impl<StorageType>& other) = default;

                allocator_impl(allocator_impl&& other) : m_storage(std::exchange(other.m_storage, nullptr)) {}

                allocator_impl& operator=(allocator_impl&& other)
                {
                    m_storage = std::exchange(other.m_storage, nullptr);
                    return *this;
                }

                allocator_impl copy() const noexcept
                {
                    return *this;
                }

                // iallocator

                void* allocate(std::size_t size, std::size_t alignment = 0UL)
                {
                    DCHECK(m_storage);
                    return m_storage->allocate(size, alignment);
                }

                void deallocate(void* ptr, std::size_t size = 0UL, std::size_t alignment = 0UL)
                {
                    DCHECK(m_storage);
                    m_storage->deallocate(ptr, size, alignment);
                }

                descriptor allocate_descriptor(std::size_t size, std::size_t alignment = 0UL)
                {
                    return m_storage->allocate_descriptor(size, alignment);
                }

                // allocator_traits

                void* allocate_node(std::size_t size, std::size_t alignment)
                {
                    DCHECK(m_storage);
                    return m_storage->allocate(size, alignment);
                }

                void* allocate_array(std::size_t count, std::size_t size, std::size_t alignment)
                {
                    DCHECK(m_storage);
                    return m_storage->allocate(count * size, alignment);
                }

                void deallocate_node(void* ptr, std::size_t size, std::size_t alignment) noexcept
                {
                    DCHECK(m_storage);
                    m_storage->deallocate(ptr, size, alignment);
                }

                void deallocate_array(void* ptr, std::size_t count, std::size_t size, std::size_t alignment) noexcept
                {
                    DCHECK(m_storage);
                    m_storage->deallocate(ptr, count * size, alignment);
                }

                DLContext device_context() const
                {
                    DCHECK(m_storage);
                    return m_storage->device_context();
                }

                std::size_t max_alignment() const
                {
                    DCHECK(m_storage);
                    return m_storage->max_alignment();
                }

                std::size_t min_alignment() const
                {
                    DCHECK(m_storage);
                    return m_storage->min_alignment();
                }

                std::size_t max_node_size() const
                {
                    DCHECK(m_storage);
                    return m_storage->max_size();
                }

                std::size_t max_array_size() const
                {
                    DCHECK(m_storage);
                    return m_storage->max_size();
                }

                // storage policy methods

                allocator_type& get_allocator() noexcept
                {
                    return m_storage->get_allocator();
                }

                const allocator_type& get_allocator() const noexcept
                {
                    return *m_storage->get_allocator();
                }

                // access raw allocator

                auto use_count() const noexcept
                {
                    return m_storage.use_count();
                }

                // allocator interface

                iallocator& get_iallocator()
                {
                    return *m_storage;
                }

                std::shared_ptr<storage_type> shared() const
                {
                    return m_storage;
                }

            private:
                template <typename T>
                friend bool operator==(const allocator_impl<T>& lhs, const allocator_impl<T>& rhs) noexcept;
            };

            template <typename T>
            bool operator==(const allocator_impl<T>& lhs, const allocator_impl<T>& rhs) noexcept
            {
                return lhs.m_storage.get() == rhs.m_storage.get();
            }

        } // namespace allocator_detail

        template <typename RawAllocator, typename Mutex>
        using allocator = allocator_detail::allocator_impl<allocator_detail::smart_storage<RawAllocator, Mutex>>;

        // convenience methods to create allocators from raw allocators

        template <typename RawAllocator>
        auto make_allocator(RawAllocator&& alloc)
        {
            auto storage = allocator_detail::make_allocator_storage<detail::mutex_for<RawAllocator, std::mutex>>(std::move(alloc));
            return allocator<RawAllocator, detail::mutex_for<RawAllocator, std::mutex>>(std::move(storage));
        }

        template <typename Mutex, typename RawAllocator>
        auto make_allocator(RawAllocator&& alloc)
        {
            auto storage = allocator_detail::make_allocator_storage<detail::mutex_for<RawAllocator, Mutex>>(std::move(alloc));
            return allocator<RawAllocator, detail::mutex_for<RawAllocator, Mutex>>(std::move(storage));
        }

        // only stateless or threadsafe RawAllocators can be used with this method
        template <typename RawAllocator>
        auto make_thread_unsafe_allocator(RawAllocator&& alloc)
        {
            auto storage = allocator_detail::make_allocator_storage<detail::mutex_for<RawAllocator, no_mutex>>(std::move(alloc));
            return allocator<RawAllocator, no_mutex>(std::move(storage));
        }

        // this specialization of is_shared_allocator
        // the allocator is shared, if - like \ref allocator_reference -
        //   if multiple objects refer to the same internal allocator and if it can be copied.
        template <typename RawAllocator, typename Mutex>
        struct is_shared_allocator<allocator<RawAllocator, Mutex>> : std::true_type
        {
        };

        template <typename RawAllocator, typename Mutex>
        struct is_thread_safe_allocator<allocator<RawAllocator, Mutex>> : std::true_type
        {
        };

    } // namespace memory
} // namespace trtlab