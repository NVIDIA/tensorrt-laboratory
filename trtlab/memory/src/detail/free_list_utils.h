// Copyright (C) 2015-2016 Jonathan Müller <jonathanmueller.dev@gmail.com>
// This file is subject to the license terms in the LICENSE file
// found in the top-level directory of this distribution.

#ifndef TRTLAB_MEMORY_SRC_DETAIL_FREE_LIST_UTILS_H_INCLUDED
#define TRTLAB_MEMORY_SRC_DETAIL_FREE_LIST_UTILS_H_INCLUDED

#include <cstdint>

#include "config.h"
#include "align.h"
#include "detail/assert.h"

#include <cstring>
#include <functional>

namespace trtlab
{
    namespace memory
    {
        namespace detail
        {
            //=== storage ===///
            // reads stored integer value
            inline std::uintptr_t get_int(void* address) noexcept
            {
                TRTLAB_MEMORY_ASSERT(address);
                std::uintptr_t res;
                std::memcpy(&res, address, sizeof(std::uintptr_t));
                return res;
            }

            // sets stored integer value
            inline void set_int(void* address, std::uintptr_t i) noexcept
            {
                TRTLAB_MEMORY_ASSERT(address);
                std::memcpy(address, &i, sizeof(std::uintptr_t));
            }

            // pointer to integer
            inline std::uintptr_t to_int(char* ptr) noexcept
            {
                return reinterpret_cast<std::uintptr_t>(ptr);
            }

            // integer to pointer
            inline char* from_int(std::uintptr_t i) noexcept
            {
                return reinterpret_cast<char*>(i);
            }

            //=== intrusive linked list ===//
            // reads a stored pointer value
            inline char* list_get_next(void* address) noexcept
            {
                return from_int(get_int(address));
            }

            // stores a pointer value
            inline void list_set_next(void* address, char* ptr) noexcept
            {
                set_int(address, to_int(ptr));
            }

            //=== intrusive xor linked list ===//
            // returns the other pointer given one pointer
            inline char* xor_list_get_other(void* address, char* prev_or_next) noexcept
            {
                return from_int(get_int(address) ^ to_int(prev_or_next));
            }

            // sets the next and previous pointer (order actually does not matter)
            inline void xor_list_set(void* address, char* prev, char* next) noexcept
            {
                set_int(address, to_int(prev) ^ to_int(next));
            }

            // changes other pointer given one pointer
            inline void xor_list_change(void* address, char* old_ptr,
                                        char* new_ptr) noexcept
            {
                TRTLAB_MEMORY_ASSERT(address);
                auto other = xor_list_get_other(address, old_ptr);
                xor_list_set(address, other, new_ptr);
            }

            // advances a pointer pair forward/backward
            inline void xor_list_iter_next(char*& cur, char*& prev) noexcept
            {
                auto next = xor_list_get_other(cur, prev);
                prev      = cur;
                cur       = next;
            }

            // links new node between prev and next
            inline void xor_list_insert(char* new_node, char* prev, char* next) noexcept
            {
                xor_list_set(new_node, prev, next);
                xor_list_change(prev, next, new_node); // change prev's next to new_node
                xor_list_change(next, prev, new_node); // change next's prev to new_node
            }

            //=== sorted list utils ===//
            // if std::less/std::greater not available compare integer representation and hope it works
            inline bool less(void* a, void* b) noexcept
            {
                return std::less<void*>()(a, b);
            }

            inline bool less_equal(void* a, void* b) noexcept
            {
                return a == b || less(a, b);
            }

            inline bool greater(void* a, void* b) noexcept
            {
                return std::greater<void*>()(a, b);
            }

            inline bool greater_equal(void* a, void* b) noexcept
            {
                return a == b || greater(a, b);
            }
        } // namespace detail
    }     // namespace memory
} // namespace trtlab

#endif // TRTLAB_MEMORY_SRC_DETAIL_FREE_LIST_UTILS_H_INCLUDED
