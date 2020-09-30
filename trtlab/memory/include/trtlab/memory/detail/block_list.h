#ifndef TRTLAB_MEMORY_DETAILL_BLOCK_LIST_H_INCLUDED
#define TRTLAB_MEMORY_DETAILL_BLOCK_LIST_H_INCLUDED

#include <cstddef>
#include <cstdint>
#include <deque>

#include "../align.h"
#include "utility.h"
#include "../config.h"
#include "../memory_block.h"

namespace trtlab
{
    namespace memory
    {
        namespace detail
        {
            // stores free blocks for a memory pool
            // memory blocks are fragmented and stored in a list
            // only accessible one block at a time
            // designed for large block which themselves will be uses as backing stores
            class block_list
            {
            protected:
                struct node
                {
                    std::size_t size;
                    node*       next;
                };

            public:
                // minimum element size
                static constexpr auto min_element_size = sizeof(node);
                // alignment
                static constexpr auto min_element_alignment = alignof(node);

                //=== constructor ===//
                block_list() noexcept;

                block_list(block_list&& other) noexcept;
                ~block_list() noexcept = default;

                block_list& operator=(block_list&& other) noexcept;

                friend void swap(block_list& a, block_list& b) noexcept;

                //=== insert/allocation/deallocation ===//
                // inserts a new memory block, by splitting it up and setting the links
                // does not own memory!
                // mem must be aligned for alignment()
                // pre: size != 0
                void insert(memory_block&&) noexcept;

                // returns a single block from the list
                // pre: !empty()
                memory_block allocate() noexcept;

                // deallocates a single block
                void deallocate(memory_block&&) noexcept;

                std::size_t next_block_size() const noexcept
                {
                    return (empty() ? 0u : first_->size);
                }

                // number of nodes remaining
                std::size_t size() const noexcept
                {
                    return capacity_;
                }

                bool empty() const noexcept
                {
                    return first_ == nullptr;
                }

            private:
                node*       first_;
                std::size_t capacity_;
            };

            // unlike block_list; block_list_oob stores the nodes in a std::deque
            // instead of as part of the allocations
            // use block_list_oob when tracking blocks whose memory the cpu can not
            // directly read/write, e.g. gpu memory
            class block_list_oob
            {
            public:
                // minimum element size
                static constexpr auto min_element_size = sizeof(long);
                // alignment
                static constexpr auto min_element_alignment = alignof(long);

                //=== constructor ===//
                block_list_oob() noexcept;

                block_list_oob(block_list_oob&& other) noexcept;
                ~block_list_oob() noexcept = default;

                block_list_oob& operator=(block_list_oob&& other) noexcept;

                friend void swap(block_list_oob& a, block_list_oob& b) noexcept;

                //=== insert/allocation/deallocation ===//
                // inserts a new memory block, by splitting it up and setting the links
                // does not own memory!
                // mem must be aligned for alignment()
                // pre: size != 0
                void insert(memory_block&&) noexcept;

                // returns a single block from the list
                // pre: !empty()
                memory_block allocate() noexcept;

                // deallocates a single block
                void deallocate(memory_block&&) noexcept;

                std::size_t next_block_size() const noexcept
                {
                    return (empty() ? 0u : nodes.front().size);
                }

                // number of nodes remaining
                std::size_t size() const noexcept
                {
                    return nodes.size();
                }

                bool empty() const noexcept
                {
                    return nodes.empty();
                }

            private:
                std::deque<memory_block> nodes;
            };

            void swap(block_list& a, block_list& b) noexcept;

            void swap(block_list_oob& a, block_list_oob& b) noexcept;
            /*
            // stores free blocks for a memory pool
            // memory blocks are fragmented and stored in a list
            // only accessible one block at a time
            // designed for large block which themselves will be uses as backing stores
            class sorted_block_list
            {
                struct block_node : public memory_block
                {
                    block_node* next;
                };

            public:
                // minimum element size
                static constexpr auto min_element_size = sizeof(char*);
                // alignment
                static constexpr auto min_element_alignment = alignof(char*);

                //=== constructor ===//
                sorted_block_list() noexcept;

                // calls other constructor plus insert
                sorted_block_list(const memory_block& block) noexcept;
                sorted_block_list(void* mem, std::size_t size) noexcept;

                sorted_block_list(sorted_block_list&& other) noexcept;
                ~sorted_block_list() noexcept = default;

                sorted_block_list& operator=(sorted_block_list&& other) noexcept;

                friend void swap(sorted_block_list& a, sorted_block_list& b) noexcept;

                //=== insert/allocation/deallocation ===//
                // inserts a new memory block, by splitting it up and setting the links
                // does not own memory!
                // mem must be aligned for alignment()
                // pre: size != 0
                void insert(const memory_block& block) noexcept;
                void insert(void* mem, std::size_t size) noexcept;

                // returns a single block from the list
                // pre: !empty()
                memory_block allocate() noexcept;

                // deallocates a single block
                void deallocate(const memory_block&) noexcept;

                // size of next block in the list
                // will be the large available block
                std::size_t next_block_size() noexcept;

                // alignment of all nodes
                std::size_t alignment() const noexcept;

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
                void insert_impl(void* mem, std::size_t size) noexcept;

                block_node* first_;
                std::size_t capacity_;
            };

            void swap(sorted_block_list& a, sorted_block_list& b) noexcept;
*/
        } // namespace detail
    }     // namespace memory
} // namespace trtlab

#endif