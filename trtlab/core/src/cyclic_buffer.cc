
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

#include "trtlab/core/cyclic_buffer.h"

#include <cstring>
#include <glog/logging.h>

using namespace trtlab;

CyclicBuffer::CyclicBuffer(memory::descriptor&& md, std::size_t window_size, std::size_t overlap_size, CallbackFn callback)
: m_Descriptor(std::move(md)),
  m_Data(reinterpret_cast<std::uintptr_t>(m_Descriptor.data())),
  m_Size(m_Descriptor.size()),
  m_WindowSize(window_size),
  m_ShiftSize(window_size - overlap_size),
  m_CurrentLocation(m_Data),
  m_WindowStart(m_Data),
  m_Callback(callback),
  m_CurrentWindow(0),
  m_PreferFullWindows(false)
{
    CHECK_LE(m_WindowSize, m_Size) << "Allocated memory must be >= the window_size";
    CHECK_LE(m_ShiftSize, m_WindowSize) << "CyclicBuffer shift_size must be <= window_size";

    m_WindowCount = (m_Size - m_WindowSize) / m_ShiftSize + 1;
    CHECK_GT(m_WindowCount, 1) << "CyclicBuffer data segment must be at least 2x the window size";
    VLOG(1) << "Allocation of " << m_Size << " bytes allows for " << m_WindowCount << " cyclic buffer windows";

    m_Size = m_WindowSize + (m_WindowCount - 1) * m_ShiftSize;
    VLOG(2) << "Effective allocation size " << m_Size << " bytes";

    // determine the location in the contiguous buffer where we need to replicate any bits
    // past that point to the head of the buffer
    auto replication_size = m_WindowSize - m_ShiftSize;
    m_NextPassStart       = m_Data;
    m_ReplicationStart    = m_Data + m_Size - replication_size;
    VLOG(2) << "Replication window begins at " << Offset(m_ReplicationStart);

    // set sync location to be the end of the buffer
    // no sync is needed on the first pass, when we reset the pointers to the head,
    // then the sync pointers move to the head and must shift forward before current
    // position is allowed to move forward
    m_SyncWindow   = m_WindowCount;
    m_SyncLocation = m_Data + m_Size;

    // initialze state trackers
    // note: if we shift by the shift size, then there is always one sync location
    // m_ShiftSize away from the end of the buffer that requires syncing.  this
    // sync location is tied to the last window and has the same state as the prior
    // sync location; however to make tracking simplier, we initialize this last
    // window state as always ready
    m_State.resize(m_WindowCount + 1);
    for (std::size_t i = 0; i < m_WindowCount + 1; i++)
    {
        ResetWindow(i);
    }
}

CyclicBuffer::~CyclicBuffer()
{
    Sync();
}

CyclicBuffer::SyncFn CyclicBuffer::AppendData(void* ptr, std::size_t size)
{
    // copy incoming data to the buffer
    // data will be copied in chunks upto the size of each window
    // windows will be computed when completed
    // this maps well to a cuda architecture of where the copy/compute
    // of one window can be overlapped with the copy of the next window
    // this call blocks until all the data has been copied to the buffer
    // the Free() method drives forward progress to reclaim buffer space
    // from windows that have completed their callbacks
    VLOG(1) << "AppendData: " << size << " bytes to buffer at offset " << Offset(m_CurrentLocation);
    std::uintptr_t data      = reinterpret_cast<std::uintptr_t>(ptr);
    auto           remaining = size;
    while (remaining)
    {
        auto chunk_size = std::min(remaining, Free());
        CopyToHead(data, chunk_size);
        data += chunk_size;
        remaining -= chunk_size;
    }
    return FinishedAppendingData();
}

std::size_t CyclicBuffer::Free()
{
    // compute the available free space remainining in the current window
    DCHECK_LE(m_WindowStart, m_CurrentLocation);
    DCHECK_LE(m_CurrentLocation, m_SyncLocation);

    if (m_CurrentLocation == m_SyncLocation)
    {
        SyncWindow(m_SyncWindow, true);
        m_SyncLocation += m_ShiftSize;
        m_SyncWindow++;
    }

    if (m_PreferFullWindows) // prefer full windows
    {
        auto next_window_start = m_WindowStart + m_WindowSize;
        while (m_SyncLocation < next_window_start)
        {
            SyncWindow(m_SyncWindow, true);
            m_SyncLocation += m_ShiftSize;
            m_SyncWindow++;
        }
    }

    auto free_current_window = m_WindowSize - (m_CurrentLocation - m_WindowStart);
    auto free_synced         = m_SyncLocation - m_CurrentLocation;
    auto free                = std::min(free_current_window, free_synced);
    VLOG(2) << free << " free bytes available in current window; " << free_current_window - free << " bytes require syncing";
    return free;
}

void CyclicBuffer::ReplicateData()
{
    VLOG(2) << "Replicating data from offset " << Offset(m_ReplicationStart) << "-" << Offset(m_Data + m_Size) << " to head";
    DCHECK_EQ(m_CurrentLocation, m_Data); // ensure we are at the start
    std::uintptr_t data      = m_ReplicationStart;
    auto           remaining = m_WindowSize - m_ShiftSize;
    while (remaining)
    {
        auto chunk_size = std::min(remaining, Free());
        ReplicateToHead(data, chunk_size);
        data += chunk_size;
        remaining -= chunk_size;
    }
    DCHECK_EQ(m_CurrentLocation, m_NextPassStart);
}

void CyclicBuffer::ReplicateToHead(std::uintptr_t data, std::size_t size)
{
    auto dst = reinterpret_cast<void*>(m_CurrentLocation);
    auto src = reinterpret_cast<void*>(data);
    VLOG(3) << "Replicating " << size << " bytes from " << Offset(data) << " to cyclic buffer @ " << Offset(m_CurrentLocation);
    Replicate(dst, src, size);
    m_CurrentLocation += size;
    DCHECK_LE(m_CurrentLocation, m_SyncLocation);
}

void CyclicBuffer::CopyToHead(std::uintptr_t data, std::size_t size)
{
    // copy bytes to current location
    // if a window is complete, kick off the execution callback
    auto dst = reinterpret_cast<void*>(m_CurrentLocation);
    auto src = reinterpret_cast<void*>(data);
    VLOG(2) << "Copying " << size << " bytes to offset " << Offset(m_CurrentLocation);
    Copy(dst, src, size);
    m_CurrentLocation += size;
    DCHECK_LE(m_CurrentLocation, m_SyncLocation);

    ExecuteWindows();
}

void CyclicBuffer::ExecuteWindows()
{
    while (m_CurrentLocation - m_WindowStart >= m_WindowSize)
    {
        VLOG(1) << "Run Callback on Window " << m_CurrentWindow << "; offset: " << Offset(m_WindowStart);
        auto data = reinterpret_cast<void*>(m_WindowStart);
        RunCallback(data, m_WindowSize);
        m_WindowStart += m_ShiftSize;
    }

    if (m_CurrentLocation - m_Data == m_Size)
    {
        VLOG(2) << "CyclicBuffer hit end of stack; resetting stack";
        m_CurrentLocation = m_Data;
        m_WindowStart     = m_Data;
        m_SyncLocation    = m_Data;
        m_SyncWindow      = 0;
        m_NextPassStart   = m_Data + (m_WindowSize - m_ShiftSize);
        // we haven't actually replicated the data from the last window
        // the first time we append data to window0, we first must sync on it
        // that ensures we can overwrite the data in window0 with the replicated
        // data first, then the new data to append

        // before we can reuse the buffer, we must replicate the last bits to the head
        if (m_CurrentLocation < m_NextPassStart)
        {
            DCHECK_EQ(m_ReplicationStart + m_ShiftSize, m_Data + m_Size);
            ReplicateData();
        }
    }
}

void CyclicBuffer::RunCallback(const void* data, std::size_t size)
{
    auto window            = m_CurrentWindow % m_WindowCount;
    m_State[window].status = WindowStatus::Running;
    m_State[window].syncfn = m_Callback(m_CurrentWindow, data, size);
    DCHECK(m_State[window].syncfn);
    m_CurrentWindow++;
}

bool CyclicBuffer::SyncWindow(std::size_t window, bool wait)
{
    VLOG(3) << "SyncWindow: " << window << "; wait=" << (wait ? "TRUE" : "FALSE");
    const auto& state = m_State[window];
    switch (state.status)
    {
    case WindowStatus::Ready:
    case WindowStatus::Finished:
        VLOG(2) << "SyncWindow " << window << " Completed: Synced == TRUE";
        return true;
    case WindowStatus::Running:
        bool finished = m_State[window].syncfn(wait);
        VLOG(2) << "SyncWindow " << window << " Completed: Synced == " << (finished ? "TRUE" : "FALSE");
        if (finished)
        {
            ResetWindow(window);
        }
        return finished;
    };

    LOG(FATAL) << "should not be reachable " << window;
    return false;
}

void CyclicBuffer::ResetWindow(std::size_t window)
{
    auto& state  = m_State[window];
    state.status = WindowStatus::Finished;
    state.syncfn = nullptr;
}

void CyclicBuffer::Sync()
{
    for (std::size_t i = 0; i < m_State.size(); i++)
    {
        SyncWindow(i, true);
    }
}

void CyclicBuffer::PreferFullWindows(bool val)
{
    m_PreferFullWindows = val;
}