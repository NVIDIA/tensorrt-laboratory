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
#include "trtlab/core/memory/sysv_allocator.h"

#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>

#include <map>
#include <mutex>

#include <glog/logging.h>

#include "trtlab/core/memory/block_arena.h"
#include "trtlab/core/memory/block_manager.h"

namespace foonathan {
namespace memory {
namespace trtlab {
namespace sysv_detail {

    struct sysv_allocation final : memory_block
    {
        sysv_allocation() : memory_block(), shm_id(-1), release(true) {}
        sysv_allocation(int id, void* mem, size_t size, bool rel = true)
            : shm_id(id), memory_block(mem, size), release(rel) {}

        sysv_allocation(const sysv_allocation&) = default;
        sysv_allocation& operator=(const sysv_allocation&) = default;

        sysv_allocation(sysv_allocation&& other)
            : shm_id(std::exchange(other.shm_id, -1)),
              release(other.shm_id),
              memory_block(std::move(other)) {}

        sysv_allocation& operator=(sysv_allocation&& other)
        {
            shm_id = std::exchange(other.shm_id, -1);
            release = other.release;
            memory_block::operator=(std::move(other));
            return *this;
        }

        int shm_id;
        bool release;
    };

    class sysv_manager final
    {
      public:
        ~sysv_manager()
        {
            // warn or remove registered sysv segments depending on settings
            DVLOG(1) << "sysv_manager shutting down";
            if(m_manager.size())
            {
                LOG(WARNING) << "Detected SysV allocations that were not deallocated";
                m_manager.for_each_block([](sysv_allocation& block) {
                    DVLOG(3) << "detaching block - ptr: " << block.memory << "; shm_id: " << block.shm_id;
                    sysv_manager::detach(block.memory);
                });
            }
        }

        static const sysv_allocation& allocate(std::size_t size)
        {
            auto shm_id = shmget(IPC_PRIVATE, size, IPC_CREAT | 0666);
            DVLOG(2) << "creating shmid: " << shm_id << "; size=" << size;
            if(shm_id == -1) { throw std::bad_alloc(); }
            return sysv_manager::attach_impl(shm_id, true);
        }

        static const sysv_allocation& attach(int shm_id)
        {
            return sysv_manager::attach_impl(shm_id, false);
        }

        static int detach(void* addr)
        {
            auto& manager = sysv_manager::global_manager();
            auto shm_id = manager.drop_allocation(addr);

            if(shm_id > 0)
            {
                auto stats = sysv_manager::get_stats(shm_id);
                if(!(stats.shm_perm.mode & SHM_DEST)) { sysv_manager::release(shm_id); }
            }

            auto rc = shmdt(addr);
            DCHECK_EQ(rc, 0) << "errno: "  << errno;
            if(rc != 0) { throw std::runtime_error("shmdt failed"); }
        }

        static void release(int shm_id)
        {
            // check if the segment has been marked for release/destruction
            auto stats = sysv_manager::get_stats(shm_id); 
            if(stats.shm_perm.mode & SHM_DEST) 
            {
                DVLOG(3) << "shm_id: " << shm_id << " already marked for removal";
                return; 
            }

            // otherwise mark it for release/destruction
            auto rc = shmctl(shm_id, IPC_RMID, &stats);
            DCHECK_EQ(rc, 0) << "shmctl failed; errno: " << errno;
            if(rc != 0) { throw std::runtime_error("unable to release shm_id"); }
            DVLOG(3) << "shm_id: " << shm_id << " marked as removed";
        }

        static std::size_t size(int shm_id)
        {
            auto stats = sysv_manager::get_stats(shm_id);
            return stats.shm_segsz;
        }

        static bool release_on_deallocate()
        {
            auto& manager = sysv_manager::global_manager();
            std::lock_guard<std::mutex> lock(manager.m_mutex);
            return manager.m_release_on_deallocate;
        }

        static bool release_on_deallocate(bool val)
        {
            auto& manager = sysv_manager::global_manager();
            std::lock_guard<std::mutex> lock(manager.m_mutex);
            manager.m_release_on_deallocate = val;
            return manager.m_release_on_deallocate;
        }

        static sysv_info sysv_info_for_pointer(void* ptr)
        {
            auto& manager = sysv_manager::global_manager();
            std::lock_guard<std::mutex> lock(manager.m_mutex);
            auto block = manager.m_manager.find_block(ptr);
            if(block && block->contains(ptr))
            {
                DCHECK_NE(block->shm_id, -1);
                auto stats = sysv_manager::get_stats(block->shm_id);
                bool released = stats.shm_perm.mode & SHM_DEST;
                return {block->shm_id, block->size, block->distance(ptr), stats.shm_nattch, !released};
            }
            throw std::runtime_error("no sysv info found for pointer");
        }

     private:
        sysv_manager() : m_release_on_deallocate(true) {}

        sysv_manager(const sysv_manager&) = delete;
        sysv_manager& operator=(const sysv_manager&) = delete;

        sysv_manager(sysv_manager&&) = delete;
        sysv_manager& operator=(sysv_manager&&) = delete;

        static sysv_manager& global_manager()
        {
            static sysv_manager manager;
            return manager;
        }

        static const sysv_allocation& attach_impl(int shm_id, bool has_ownership)
        {
            DVLOG(2) << "attaching to shm_id: " << shm_id;
            auto stats = sysv_manager::get_stats(shm_id);
            if(stats.shm_perm.mode & SHM_DEST) { throw std::bad_alloc(); }
            auto ptr = shmat(shm_id, nullptr, 0);
            if(ptr == (void*)-1) { throw std::bad_alloc(); }
            auto& manager = sysv_manager::global_manager();
            return manager.register_allocation(shm_id, ptr, sysv_manager::size(shm_id), has_ownership);
        }

        static struct shmid_ds  get_stats(int shm_id)
        {
            struct shmid_ds stats;
            auto rc = shmctl(shm_id, IPC_STAT, &stats);
            DCHECK_EQ(rc, 0);
            if(rc != 0) throw std::runtime_error("no sysv info found for pointer");
            return stats;
        }

        const sysv_allocation& register_allocation(int shm_id, void* addr, std::size_t size, bool has_ownership)
        {
            // ownership and release on deallocate need to be true for a sysv segment to be removed
            // allocate -> has_ownership == true
            // attach   -> has_ownership == false
            std::lock_guard<std::mutex> lock(m_mutex);
            DVLOG(3) << this << ": registering shmd_id: " << shm_id << "; addr: " << addr << "; size: " << size;
            return m_manager.add_block({shm_id, addr, size, m_release_on_deallocate && has_ownership});
        }

        int drop_allocation(void* addr)
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            int shm_id = -1;
            auto block = m_manager.find_block(addr);
            DCHECK(block);
            if(block)
            {
                DVLOG(3) << this << ": dropping sysv allocation containing " << addr;
                m_manager.drop_block(addr);
                if(block->release) { shm_id = block->shm_id; }
            }
            DVLOG(3) << this << ": " << addr << " maps to shm_id " << shm_id;
            return shm_id;
        }

        std::mutex m_mutex;
        block_manager<sysv_allocation> m_manager;
        bool m_release_on_deallocate;
    };

} // namesapce sysv_detail

    void* sysv_allocator::allocate_node(std::size_t size, std::size_t)
    {
        DVLOG(1) << "sysv::allocate_node - " << size;
        const auto& allocation = sysv_detail::sysv_manager::allocate(size);
        return allocation.memory;
    }

    void sysv_allocator::deallocate_node(void* ptr, std::size_t, std::size_t)
    {
        DVLOG(1) << "sysv::deallocate_node - " << ptr;
        sysv_detail::sysv_manager::detach(ptr);
    }

    void* sysv_allocator::attach(int shm_id)
    {
        DVLOG(1) << "sysv::attach - " << shm_id;
        const auto& allocation = sysv_detail::sysv_manager::attach(shm_id);
        return allocation.memory;
    }

    void sysv_allocator::release(int shm_id)
    {
        DVLOG(1) << "sysv::release - " << shm_id;
        sysv_detail::sysv_manager::release(shm_id);

    }

    sysv_info sysv_allocator::sysv_info_for_pointer(void* ptr)
    {
        return sysv_detail::sysv_manager::sysv_info_for_pointer(ptr);
    }

    bool sysv_allocator::release_on_deallocate()
    {
        return sysv_detail::sysv_manager::release_on_deallocate();
    }

    bool sysv_allocator::release_on_deallocate(bool val)
    {
        return sysv_detail::sysv_manager::release_on_deallocate(val);
    }

} // namespace trtlab
} // namespace memory
} // namespace foonathan