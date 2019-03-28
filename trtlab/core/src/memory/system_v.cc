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
#include "trtlab/core/memory/system_v.h"

#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>

#include <map>
#include <mutex>

#include <glog/logging.h>

static std::mutex g_Mutex;
static std::map<int, std::weak_ptr<int>> g_Segments;

namespace {

void* ShmAt(int shm_id)
{
    auto ptr = shmat(shm_id, nullptr, 0);
    CHECK(ptr);
    DLOG(INFO) << "Attaching SystemV shm_id: " << shm_id;
    return ptr;
}

size_t SegSize(int shm_id)
{
    struct shmid_ds stats;
    CHECK_EQ(shmctl(shm_id, IPC_STAT, &stats), 0);
    return stats.shm_segsz;
}

bool ShmExists(int shm_id)
{
    struct shmid_ds buff;
    auto ret = shmctl(shm_id, IPC_STAT, &buff);
    return (bool)(ret == 0);
}

} // namespace

namespace trtlab {

// SystemV

SystemV::SystemV() : HostMemory(), m_ShmID(nullptr) {}

SystemV::SystemV(int shm_id) : HostMemory(ShmAt(shm_id), SegSize(shm_id))
{
    RegisterAttachment(shm_id);
    CHECK_GE(ShmID(), 0);
}

SystemV::SystemV(void* ptr, mem_size_t size) : HostMemory(ptr, size) {}

SystemV::SystemV(void* ptr, mem_size_t size, const SystemV& parent)
    : HostMemory(ptr, size, parent), m_ShmID(parent.m_ShmID)
{
}

SystemV::SystemV(SystemV&& other) noexcept
    : HostMemory(std::move(other)), m_ShmID(std::move(other.m_ShmID))
{
}

SystemV::~SystemV()
{
    m_ShmID.reset();
}

const char* SystemV::TypeName() const { return "SystemV"; }

void* SystemV::Allocate(size_t size)
{
    int shm_id;
    shm_id = shmget(IPC_PRIVATE, size, IPC_CREAT | 0666);
    CHECK_NE(shm_id, -1);
    DLOG(INFO) << "Created SystemV shm_id: " << shm_id;
    m_ShmID = std::shared_ptr<int>(new int(shm_id), [shm_id](auto p) mutable {
        std::lock_guard<std::mutex> lock(g_Mutex);
        DLOG(INFO) << "Removing SystemV shm_id: " << shm_id;
        struct shmid_ds buff;
        CHECK_EQ(shmctl(shm_id, IPC_RMID, &buff), 0);
        g_Segments.erase(shm_id);
        delete p;
    });
    CHECK_EQ(*m_ShmID, shm_id);

    // While g_Segments weak_ptr is still valid, other can attach to the segment.
    // Once expired, the memory is still allocated until all attatched processes
    // detatch or expire.  No new attachments are allowed.
    std::lock_guard<std::mutex> lock(g_Mutex);
    g_Segments[shm_id] = m_ShmID;
    // maintain the lock so no one can release it before attaching
    return ShmAt(*m_ShmID);
}

std::function<void()> SystemV::Free()
{
    CHECK(Data() && Capacity());
    return [ptr = Data(), size = Capacity()] {
        DLOG(INFO) << "Detaching SystemV (allocator) ptr=" << ptr << "; size=" << size;
        shmdt(ptr);
    };
}

void SystemV::RegisterAttachment(int shm_id)
{
    std::lock_guard<std::mutex> lock(g_Mutex);
    auto search = g_Segments.find(shm_id);

    if(search == g_Segments.end())
    {
        if(ShmExists(shm_id))
        {
            // A different process created the shared segment,
            // it is responsible for managing/removing it
            m_ShmID = std::make_shared<int>(shm_id);
        }
        else
        {
            throw std::runtime_error("The shmid was not found on the system");
        }
    }
    else
    {
        if(!ShmExists(shm_id))
        {
            throw std::runtime_error(
                "The shmid was removed from the system; likely from outside the application");
        }
    }

    if(search->second.expired())
    {
        throw std::runtime_error("The shm_id expired before it could be attached");
    }

    // Grab a copy of the shared_ptr from the weak_ptr's control block
    m_ShmID = search->second.lock();
}

DescriptorHandle<SystemV> SystemV::Attach(int shm_id)
{
    class SystemVDescriptor final : public Descriptor<SystemV>
    {
      public:
        SystemVDescriptor(SystemV&& sysv, std::function<void()> deleter)
            : Descriptor<SystemV>(std::move(sysv), deleter, "SystemV Attach")
        {
        }
    };
    auto sysv = SystemV(shm_id);
    CHECK_GE(sysv.ShmID(), 0);
    auto deleter = [ptr = sysv.Data(), size = sysv.Capacity()] {
        DLOG(INFO) << "Detaching SystemV (descriptor) ptr=" << ptr << "; size=" << size;
        shmdt(ptr);
    };
    return std::make_unique<SystemVDescriptor>(std::move(sysv), deleter);
}

void SystemV::DisableAttachment() { m_ShmID.reset(); }

int SystemV::ShmID() const
{
    if(m_ShmID)
    {
        return *m_ShmID;
    }
    return -1;
}

} // namespace trtlab
