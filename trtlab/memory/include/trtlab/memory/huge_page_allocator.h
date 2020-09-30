#pragma once

#include "posix_aligned_allocator.h"

namespace trtlab
{
    namespace memory
    {
        template <std::size_t HugePageSize>
        struct transparent_huge_page_allocator : public posix_aligned_allocator<HugePageSize>
        {
            static_assert(is_valid_alignment(HugePageSize), "must be a power of 2");

            static void* allocate_node(std::size_t size, std::size_t)
            {
                // ensure complete pages are allocated
                auto addl_bytes_for_complete_page = size % HugePageSize;
                DLOG_IF(WARNING, addl_bytes_for_complete_page)
                    << "transparent_huge_page_allocator allocates complete pages; " << addl_bytes_for_complete_page
                    << " additional bytes were added to allocation";
                size += addl_bytes_for_complete_page;

                // allocate and advise
                void* ptr = posix_aligned_allocator<HugePageSize>::allocate_node(size, HugePageSize);
                auto ret = madvise(ptr, size, MADV_HUGEPAGE);
                if (ret)
                {
                    LOG(WARNING) << "madvise returned an error: " << std::strerror(errno);
                }
                return ptr;
            }

            constexpr static std::size_t page_size()
            {
                return HugePageSize;
            }
        };
    } // namespace memory
} // namespace trtlab