// Copyright (C) 2015-2016 Jonathan Müller <jonathanmueller.dev@gmail.com>
// This file is subject to the license terms in the LICENSE file
// found in the top-level directory of this distribution.

#ifndef TRTLAB_MEMORY_DETAILL_FREE_LIST_H_INCLUDED
#define TRTLAB_MEMORY_DETAILL_FREE_LIST_H_INCLUDED

#include <cstddef>
#include <cstdint>

#include "../align.h"
#include "utility.h"
#include "../config.h"

namespace trtlab
{
    namespace memory
    {
        namespace detail
        {
            // stores free blocks for a memory pool
            // memory blocks are fragmented and stored in a list
            // debug: fills memory and uses a bigger node_size for fence memory
            class free_memory_list
            {
            public:
                // minimum element size
                static constexpr auto min_element_size = sizeof(char*);
                // alignment
                static constexpr auto min_element_alignment = alignof(char*);

                //=== constructor ===//
                free_memory_list(std::size_t node_size) noexcept;

                // calls other constructor plus insert
                free_memory_list(std::size_t node_size, void* mem, std::size_t size) noexcept;

                free_memory_list(free_memory_list&& other) noexcept;
                ~free_memory_list() noexcept = default;

                free_memory_list& operator=(free_memory_list&& other) noexcept;

                friend void swap(free_memory_list& a, free_memory_list& b) noexcept;

                //=== insert/allocation/deallocation ===//
                // inserts a new memory block, by splitting it up and setting the links
                // does not own memory!
                // mem must be aligned for alignment()
                // pre: size != 0
                void insert(void* mem, std::size_t size) noexcept;

                // returns the usable size
                // i.e. how many memory will be actually inserted and usable on a call to insert()
                std::size_t usable_size(std::size_t size) const noexcept
                {
                    return size;
                }

                // returns a single block from the list
                // pre: !empty()
                void* allocate() noexcept;

                // returns a memory block big enough for n bytes
                // might fail even if capacity is sufficient
                void* allocate(std::size_t n) noexcept;

                // deallocates a single block
                void deallocate(void* ptr) noexcept;

                // deallocates multiple blocks with n bytes total
                void deallocate(void* ptr, std::size_t n) noexcept;

                //=== getter ===//
                std::size_t node_size() const noexcept
                {
                    return node_size_;
                }

                // alignment of all nodes
                std::size_t alignment() const noexcept
                {
                    return alignment_;
                }

                // number of nodes remaining
                std::size_t capacity() const noexcept
                {
                    return capacity_;
                }

                bool empty() const noexcept
                {
                    return first_ == nullptr;
                }

            private:
                std::size_t fence_size() const noexcept
                {
                #if TRTLAB_MEMORY_DEBUG_FILL
                    #error "this is broken; disable debug fill"
                    return debug_fence_size ? node_size_ : 0u;
                #else
                    return 0u;
                #endif
                }

                //std::size_t fence_size() const noexcept;
                void insert_impl(void* mem, std::size_t size) noexcept;

                char*       first_;
                std::size_t node_size_, capacity_, alignment_;
            };

            void swap(free_memory_list& a, free_memory_list& b) noexcept;

            // same as above but keeps the nodes ordered
            // this allows array allocations, that is, consecutive nodes
            // debug: fills memory and uses a bigger node_size for fence memory
            class ordered_free_memory_list
            {
            public:
                // minimum element size
                static constexpr auto min_element_size = sizeof(char*);
                // alignment
                static constexpr auto min_element_alignment = alignof(char*);

                //=== constructor ===//
                ordered_free_memory_list(std::size_t node_size) noexcept;

                ordered_free_memory_list(std::size_t node_size, void* mem, std::size_t size) noexcept : ordered_free_memory_list(node_size)
                {
                    insert(mem, size);
                }

                ordered_free_memory_list(ordered_free_memory_list&& other) noexcept;

                ~ordered_free_memory_list() noexcept = default;

                ordered_free_memory_list& operator=(ordered_free_memory_list&& other) noexcept
                {
                    ordered_free_memory_list tmp(detail::move(other));
                    swap(*this, tmp);
                    return *this;
                }

                friend void swap(ordered_free_memory_list& a, ordered_free_memory_list& b) noexcept;

                //=== insert/allocation/deallocation ===//
                // inserts a new memory block, by splitting it up and setting the links
                // does not own memory!
                // mem must be aligned for alignment()
                // pre: size != 0
                void insert(void* mem, std::size_t size) noexcept;

                // returns the usable size
                // i.e. how many memory will be actually inserted and usable on a call to insert()
                std::size_t usable_size(std::size_t size) const noexcept
                {
                    return size;
                }

                // returns a single block from the list
                // pre: !empty()
                void* allocate() noexcept;

                // returns a memory block big enough for n bytes (!, not nodes)
                // might fail even if capacity is sufficient
                void* allocate(std::size_t n) noexcept;

                // deallocates a single block
                void deallocate(void* ptr) noexcept;

                // deallocates multiple blocks with n bytes total
                void deallocate(void* ptr, std::size_t n) noexcept;

                //=== getter ===//
                std::size_t node_size() const noexcept
                {
                    return node_size_;
                }

                // alignment of all nodes
                std::size_t alignment() const noexcept;

                // number of nodes remaining
                std::size_t capacity() const noexcept
                {
                    return capacity_;
                }

                bool empty() const noexcept
                {
                    return capacity_ == 0u;
                }

            private:
                std::size_t fence_size() const noexcept;

                // returns previous pointer
                char* insert_impl(void* mem, std::size_t size) noexcept;

                char* begin_node() noexcept;
                char* end_node() noexcept;

                std::uintptr_t begin_proxy_, end_proxy_;
                std::size_t    node_size_, capacity_;
                char *         last_dealloc_, *last_dealloc_prev_;
            };

            void swap(ordered_free_memory_list& a, ordered_free_memory_list& b) noexcept;

#if TRTLAB_MEMORY_DEBUG_DOUBLE_DEALLOC_CHECk
            // use ordered version to allow pointer check
            using node_free_memory_list  = ordered_free_memory_list;
            using array_free_memory_list = ordered_free_memory_list;
#else
            using node_free_memory_list = free_memory_list;
            using array_free_memory_list = ordered_free_memory_list;
#endif
        } // namespace detail
    }     // namespace memory
} // namespace trtlab

#endif // TRTLAB_MEMORY_DETAILL_FREE_LIST_H_INCLUDED
