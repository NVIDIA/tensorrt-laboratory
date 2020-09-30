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
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <experimental/propagate_const>

#include <trtlab/memory/allocator.h>
#include <trtlab/memory/descriptor.h>

namespace trtlab
{
    class CyclicBuffer
    {
    public:
        using SyncFn     = std::function<bool(bool)>;
        using CallbackFn = std::function<SyncFn(std::size_t, const void*, std::size_t)>;

        CyclicBuffer(memory::descriptor&&, std::size_t window_size, std::size_t overlap_size, CallbackFn);
        virtual ~CyclicBuffer();

        // returns the number of bytes needed for count windows of size overlapping by overlap bytes
        static std::size_t SizeFor(std::size_t count, std::size_t size, std::size_t overlap)
        {
            CHECK_GT(count, 0);
            CHECK_GT(size, overlap);
            return size + (count - 1) * (size - overlap);
        }

        // as data is appended, the callback will be triggered when a window is completed.
        // after a window is completed, the window is shifted by ShiftSize. if the
        // subsequent window is also complete, callbacks will be triggered.
        // AppendData will block until all data the buffer has allocated memory for all the data.
        // AppendData could issue async copies.  The only guarantee is that all memory copies have
        // been issued, but may not have finished when AppendData returns.  This means that the
        // memory passed to the AppendData method is volatile and should not be changed until the
        // SyncFn has returns true. AppendData will use the thread to drive forward progress
        // of the buffers internal state.  before new data is appended, a window must be available
        // for use.  Under the worst case scenario, no windows are free, so we must wait on the
        // next window in the ring to become available.  data is copied into available windows.
        // when a window fills, the callback method is applied to that data region and the state
        // of the copy and callback are tracked by the manager.  this continues until all data
        // has been consumed.

        // blocks until all copies are in-flight
        // one must wait on the CopyStatus method to check for completeness
        // when CopyStatus() returns true, then the data buffer passed to AppendData can be reused
        SyncFn AppendData(void*, std::size_t);

        // tell the buffer to shutdown, this will attempt to push all windows into a "running"
        // state.  the final window may not be data complete, but a method in the manager
        // will handle that case.
        void Shutdown(std::function<void(std::size_t, const void* data, std::size_t)> fill);

        // waits for all windows to become available
        void Sync();

        void PreferFullWindows(bool);

    protected:
        std::size_t Size() const
        {
            return m_Size;
        }
        std::size_t WindowSize() const
        {
            return m_WindowSize;
        }
        std::size_t ShiftSize() const
        {
            return m_ShiftSize;
        }

        std::size_t Offset(const void* addr) const
        {
            auto mem = reinterpret_cast<std::uintptr_t>(addr);
            DCHECK_GE(mem, m_Data);
            return mem - m_Data;
        }

        std::size_t Offset(std::uintptr_t addr) const
        {
            DCHECK_GE(addr, m_Data);
            return addr - m_Data;
        }

    private:
        enum class WindowStatus
        {
            Ready,
            Running,
            Finished
        };

        struct WindowState
        {
            WindowStatus status;
            SyncFn       syncfn;
        };

        // used to copy data from AppendData into the buffer
        virtual void Copy(void* dst, const void* src, std::size_t size) = 0;

        // used to copy data to be replicated data internally within the buffer
        virtual void Replicate(void* dst, const void* src, std::size_t size) = 0;

        // if data can be async copied into the buffer, then AppendData
        // may return before the entirely the data in its source buffers
        // has be copied to the buffer; however, all copies are in-flight
        // the return value of SyncFn signifies when the source data can be reused
        virtual SyncFn FinishedAppendingData() = 0;

        // wrapper to set state and capture the sync function of the callback
        void RunCallback(const void* data, std::size_t size);

        // determine the amount of free bytes availabe in the current window
        std::size_t Free();

        // repllicate overlapping data from end of the buffer to the start
        // this method is called automatically by Free when the buffer resets
        void ReplicateData();

        // wrapper to copy external data to the buffer and shift internal pointers
        void CopyToHead(std::uintptr_t data, std::size_t size);

        // wrapper to copy/move internal data within the buffer
        void ReplicateToHead(std::uintptr_t data, std::size_t size);

        // called to launch callback on all available windows
        void ExecuteWindows();

        // wrapper to set state and capture the sync function of a callback
        void RunCallbackOnWindow(std::uintptr_t data, std::size_t size);

        // returns true of the window is ready for use; false if not
        bool SyncWindow(std::size_t window, bool wait);

        // resets the state of a window
        void ResetWindow(std::size_t window);

        // the following are directly set by constructor calling arguments
        memory::descriptor m_Descriptor;
        std::uintptr_t     m_Data;
        std::size_t        m_Size;
        std::size_t        m_WindowSize;
        std::size_t        m_ShiftSize;
        std::uintptr_t     m_CurrentLocation; // location of next data append
        std::uintptr_t     m_WindowStart;     // start of current window
        std::size_t        m_CurrentWindow;
        bool               m_PreferFullWindows;
        CallbackFn         m_Callback;

        // the remaining variables are set in the constructor's body
        std::uintptr_t           m_ReplicationStart; // location of data which will be copied to head
        std::uintptr_t           m_NextPassStart;    // location of first new data after replicated data
        std::uintptr_t           m_SyncLocation;     // location of the last byte of sync'ed memory
        std::size_t              m_SyncWindow;       // window number of
        std::size_t              m_WindowCount;
        std::vector<WindowState> m_State;
    };

    class HostCyclicBuffer : public CyclicBuffer
    {
    public:
        using CyclicBuffer::CyclicBuffer;

    private:
        void Copy(void* dst, const void* src, std::size_t size) final override
        {
            std::memcpy(dst, src, size);
        }

        void Replicate(void* dst, const void* src, std::size_t size) final override
        {
            std::memcpy(dst, src, size);
        }

        SyncFn FinishedAppendingData() final override
        {
            return [](bool) { return true; };
        }
    };

    template <typename T, typename WindowedBuffer>
    class TypedWindowedBuffer : private WindowedBuffer
    {
    public:
        using SyncFn     = std::function<bool(bool)>;
        using CallbackFn = std::function<SyncFn(const T*, std::size_t)>;

        TypedWindowedBuffer(std::size_t window_size, std::size_t shift_size, memory::descriptor&& md, CallbackFn callback)
        : WindowedBuffer(window_size * sizeof(T), shift_size * sizeof(T), std::move(md),
                         [callback](const void* data, std::size_t size) -> SyncFn {
                             DCHECK_EQ(size % sizeof(T), 0);
                             return callback(static_cast<const T*>(data), size / sizeof(T));
                         })
        {
        }

        virtual ~TypedWindowedBuffer() {}

        SyncFn AppendData(const T* data, std::size_t count)
        {
            return WindowedBuffer::AppendData(static_cast<void*>(data), count * sizeof(T));
        }

        void Sync()
        {
            WindowedBuffer::Sync();
        }
    };

} // namespace trtlab