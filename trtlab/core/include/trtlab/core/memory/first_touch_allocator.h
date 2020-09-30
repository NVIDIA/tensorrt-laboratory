#pragma once
#include <cstring>
#include <set>
#include <utility>

#include <trtlab/memory/allocator_traits.h>

#include "../affinity.h"

#if HAS_LIBNUMA
#include <numaif.h>
#endif

namespace trtlab
{
    namespace memory
    {
        namespace traits_detail
        {
            //=== page_size() ===//
            // first try Allocator::page_size()
            // then return maximum value
            template <class Allocator>
            auto page_size(full_concept, const Allocator& alloc) -> TRTLAB_AUTO_RETURN_TYPE(alloc.page_size(), std::size_t)
/*
                template <class Allocator>
                std::size_t page_size(min_concept, const Allocator&)
            {
                return std::size_t(0);
            }
*/
        } // namespace traits_detail

        template <typename RawAllocator, char Fill = 0x42>
        class first_touch_allocator : TRTLAB_EBO(allocator_traits<RawAllocator>::allocator_type)
        {
            using traits            = allocator_traits<RawAllocator>;
            using composable_traits = composable_allocator_traits<RawAllocator>;
            using composable        = is_composable_allocator<typename traits::allocator_type>;

        public:
            using allocator_type = typename allocator_traits<RawAllocator>::allocator_type;
            using memory_type    = typename allocator_traits<RawAllocator>::memory_type;
            using is_stateful    = std::true_type;

            static_assert(is_host_memory<memory_type>::value, "currently only implemented for host_memory");

            explicit first_touch_allocator(int numa_node_id, RawAllocator&& alloc = {}) : allocator_type(std::move(alloc))
            {
                auto topology = affinity::system::topology();
                CHECK_LE(numa_node_id, topology.size());
                m_numa_node = topology[numa_node_id];
            }

            first_touch_allocator(first_touch_allocator&& other) noexcept
            : allocator_type(std::move(other)), m_numa_node(std::move(other.m_numa_node))
            {
            }

            first_touch_allocator& operator=(first_touch_allocator&& other) noexcept
            {
                allocator_type::operator=(std::move(other));
                m_numa_node             = std::move(other.m_numa_node);
                return *this;
            }

            void* allocate_node(std::size_t size, std::size_t alignment)
            {
                
                affinity_guard scoped_affinity(m_numa_node.cpus);
                auto           ptr = traits::allocate_node(get_allocator(), size, alignment);
                return first_touch(ptr, size);
            }

            void* allocate_array(std::size_t count, std::size_t size, std::size_t alignment)
            {
                affinity_guard scoped_affinity(m_numa_node.cpus);
                auto           ptr = traits::allocate_array(get_allocator(), count, size, alignment);
                return first_touch(ptr, count * size);
            }

            // TODO: turn all the default implemenations into macros
            // DEFAULT_TRAITS_DEALLOCATE_NODE
            // DEFAULT_TRAITS_DEALLOCATE_ARRAY
            // etc

            void deallocate_node(void* ptr, std::size_t size, std::size_t alignment) noexcept
            {
                traits::deallocate_node(get_allocator(), ptr, size, alignment);
            }

            void deallocate_array(void* ptr, std::size_t count, std::size_t size, std::size_t alignment) noexcept
            {
                traits::deallocate_array(get_allocator(), ptr, count, size, alignment);
            }

            std::size_t max_node_size() const
            {
                return traits::max_node_size(get_allocator());
            }

            std::size_t max_array_size() const
            {
                return traits::max_array_size(get_allocator());
            }

            std::size_t max_alignment() const
            {
                return traits::max_alignment(get_allocator());
            }

            std::size_t min_alignment() const
            {
                return traits::min_alignment(get_allocator());
            }

            /// @{
            /// \returns A reference to the underlying allocator.
            allocator_type& get_allocator() noexcept
            {
                return *this;
            }

            const allocator_type& get_allocator() const noexcept
            {
                return *this;
            }
            /// @}

        private:
            void* first_touch(void* ptr, std::size_t size)
            {
                auto page_size = traits_detail::page_size(traits_detail::full_concept{}, get_allocator());
                if (page_size && is_aligned(ptr, page_size))
                {
                    DVLOG(2) << "allocator provides memory aligned to page_size: " << page_size;
                    first_touch_aligned_pages(ptr, size, page_size);
                }
                else
                {
                    DVLOG(2) << "allocator did not provide page_size or is not aligned on a page boundry";
                    std::memset(ptr, Fill, size);
                }
                return ptr;
            }

            void first_touch_aligned_pages(void* ptr, std::size_t size, std::size_t page_size)
            {
                auto page_count = size / page_size + (size % page_size ? 1 : 0);
                auto nodes      = 1;
#if HAS_LIBNUMA
                void* pages[page_count];
                int   nodes[page_count];
                int   status[page_count];
#endif
                auto pages_per_node  = page_count / nodes;
                auto pages_remaining = page_count % nodes;

                for (std::size_t n = 0; n < nodes; n++)
                {
                    auto pages = pages_per_node + (n < pages_remaining ? 1 : 0);
                    DVLOG(1) << "assign " << pages << " to numa_ndoe " << n;

                    affinity_guard scoped_affinity(m_numa_node.cpus);
                    for (std::size_t p = n; p < pages; p += nodes)
                    {
                        char* page = static_cast<char*>(ptr);
                        page += p * page_size;
                        *page = Fill; // touch the first byte of the page
#if HAS_LIBNUMA
                        pages[p] = static_cast<void*>(page);
                        nodes[p] = n; // the node the page should be on
#endif
                    }
                }
#if HAS_NUMA
                // get status of pages
                auto rc = move_pages(0 /*self memory */, page_count, &pages, NULL, status, 0);
                if (rc)
                {
                    LOG(WARNING) << "page_check: move_pages returned :" << std::strerror(errno);
                }

                std::vector<int> move;
                for (std::size_t p = 0; p < page_count; p++)
                {
                    LOG_IF(WARNING, status[p] < 0) << "page " << p << ": " << std::strerror(std::abs(errno));
                    if (status[p] >= 0 && status[p] != nodes[p])
                    {
                        move.push_back(p);
                        LOG(WARNING) << "page " << p << ": expected on node " << node[p] "; found on " << status[p];
                    }
                }

                // move any misaligned pages
                if (move.size())
                {
                    // overwrite pages/nodes with pages o be moved
                    for (std::size_t p = 0; p < move.size; p++)
                    {
                        page_id  = move[p];
                        pages[p] = pages[page_id];
                        nodes[p] = nodes[page_id];
                    }
                    rc = move_pages(0, move.size(), &pages, nodes, status, MPOL_MF_MOVE);
                }
#endif
            }

            numa_node m_numa_node;
        };

    } // namespace memory
} // namespace trtlab