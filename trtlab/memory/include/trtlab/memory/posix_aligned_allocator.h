#pragma once

#include <cerrno>
#include <stdlib.h>
#include <sys/mman.h>

#include "align.h"

namespace trtlab
{
    namespace memory
    {
        template <std::size_t Alignment>
        struct posix_aligned_allocator
        {
            static_assert(is_valid_alignment(Alignment), "must be a power of 2");

            using memory_type = host_memory;
            using is_stateful = std::false_type;

            static void* allocate_node(std::size_t size, std::size_t)
            {
                void* ptr = NULL;
                int   ret = posix_memalign(&ptr, Alignment, size);
                if (ret)
                {
                    throw std::bad_alloc();
                }
                return ptr;
            }

            static void deallocate_node(void* ptr, std::size_t, std::size_t) noexcept
            {
                free(ptr);
            }

            constexpr static std::size_t max_alignment()
            {
                return Alignment;
            }

            constexpr static std::size_t min_alignment()
            {
                return Alignment;
            }
        };
    } // namespace memory
} // namespace trtlab