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
#include <cstddef>
#include <map>
#include <queue>
#include <utility>

#include <glog/logging.h>

#include "memory_block.h"

namespace trtlab
{
    namespace memory
    {
        template <typename BlockType>
        class block_manager final
        {
            static_assert(std::is_base_of<memory_block, BlockType>::value, "should be derived from memory_block");

        public:
            using block_type = BlockType;

            block_manager()  = default;
            ~block_manager() = default;

            block_manager(block_manager&& other) noexcept : m_block_map(std::move(other.m_block_map)) {}

            block_manager& operator=(block_manager&& other)
            {
                m_block_map = std::move(other.m_block_map);
                return *this;
            }

            block_manager(const block_manager&) = delete;
            block_manager& operator=(const block_manager&) = delete;

            const block_type& add_block(block_type&& block)
            {
                auto key = reinterpret_cast<std::uintptr_t>(block.memory) + block.size;
                // TODO: check if an overlapping block exists
                // this would be a failure condition
                DVLOG(1) << "adding block: " << key << " - " << block.memory << "; " << block.size;
                m_block_map[key] = std::move(block);
                return m_block_map[key];
            }

            block_type* find_block(const void* ptr)
            {
                auto search = find_entry(ptr);
                if (search != m_block_map.end() && search->second.contains(ptr))
                {
                    DVLOG(3) << this << ": block found";
                    return &search->second;
                }
                DVLOG(3) << this << ": no block found for " << ptr;
                return nullptr;
            }

            void drop_block(void* ptr)
            {
                DVLOG(1) << "dropping block: " << ptr;
                auto search = find_entry(ptr);
                if (search != m_block_map.end())
                {
                    DVLOG(3) << "found block; dropping block: " << search->first << "; " << search->second.memory;
                    m_block_map.erase(search);
                }
            }

            auto size() const noexcept
            {
                return m_block_map.size();
            }

            void clear() noexcept
            {
                DVLOG(2) << "clearing block map";
                m_block_map.clear();
            }

            std::vector<void*> blocks() const noexcept
            {
                DVLOG(2) << "getting a vector of blocks - " << m_block_map.size();
                std::vector<void*> v;
                v.reserve(m_block_map.size());
                for (const auto& it : m_block_map)
                {
                    v.push_back(it.second.memory);
                }
                return v;
            }

            bool owns(void* addr)
            {
                auto block = find_block(addr);
                return (block && block->contains(addr));
            }

        private:
            inline auto find_entry(const void* ptr)
            {
                DVLOG(3) << "looking for block containing: " << ptr;
                auto key = reinterpret_cast<std::uintptr_t>(ptr);
                return m_block_map.upper_bound(key);
            }

            // todo: used a static block allocator here to avoid allocation issues
            std::map<std::uintptr_t, block_type> m_block_map;
        };
    } // namespace memory
} // namespace trtlab