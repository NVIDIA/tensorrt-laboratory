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
#include <future>
#include <memory>

#include <trtlab/memory/align.h>
#include <trtlab/memory/descriptor.h>
#include <trtlab/memory/memory_type.h>

namespace trtlab
{
    // forward declare the classes in this header
    // buffer hold the data and core properties
    class cyclic_windowed_buffer;

    // stack manages window lifecycle and syncing
    //template<typename MemoryType = memory::host_memory>
    //class cyclic_windowed_stack;

    // launches a task as each window is filled
    template <typename MemoryType, typename ThreadType>
    class cyclic_windowed_task_executor;

    // provides a reserved window with the requested amount of overlap
    // only one window can be reserved at a time
    class cyclic_windowed_reservation;

    // cyclic windowed buffer

    class cyclic_windowed_buffer
    {
    public:
        cyclic_windowed_buffer();

        cyclic_windowed_buffer(memory::descriptor md, std::size_t window_size, std::size_t overlap_size);

        virtual ~cyclic_windowed_buffer() {}

        cyclic_windowed_buffer(cyclic_windowed_buffer&&) noexcept;
        cyclic_windowed_buffer& operator=(cyclic_windowed_buffer&&) noexcept;

        cyclic_windowed_buffer(const cyclic_windowed_buffer&) = delete;
        cyclic_windowed_buffer& operator=(const cyclic_windowed_buffer&) = delete;

        // returns the number of bytes needed for count windows of size overlapping by overlap bytes
        static std::size_t min_allocation_size(std::size_t window_count, std::size_t window_size, std::size_t overlap);

        // number of windows in the buffer
        std::size_t window_count() const noexcept
        {
            return m_window_count;
        }

        // size of each window in bytes
        std::size_t window_size() const noexcept
        {
            return m_window_size;
        }

        // the stride of the shift in bytes
        std::size_t shift_size() const noexcept
        {
            return m_shift_size;
        }

        // amount of overlap between windows in bytes
        std::size_t overlap_size() const noexcept
        {
            return m_window_size - m_shift_size;
        }

        // effective number of the allocated bytes used to back the windows
        std::size_t capacity() const noexcept
        {
            return m_capacity;
        };

        DLContext device_context() const noexcept
        {
            return m_descriptor.device_context();
        }

    protected:
        memory::addr_t data() const noexcept
        {
            return static_cast<memory::addr_t>(const_cast<void*>(m_descriptor.data()));
        }

        std::ptrdiff_t offset(memory::addr_t p) const noexcept
        {
            return p - data();
        }

    private:
        // initializer list
        memory::descriptor m_descriptor; /* move-only */
        std::size_t        m_window_size;

        // computed in constructor
        std::size_t m_window_count;
        std::size_t m_shift_size;
        std::size_t m_capacity;
    };

    namespace detail
    {
        class cyclic_windowed_stack_impl : private cyclic_windowed_buffer
        {
        public:
            cyclic_windowed_stack_impl();
            cyclic_windowed_stack_impl(memory::descriptor md, std::size_t window_size, std::size_t overlap_size);

            cyclic_windowed_stack_impl(cyclic_windowed_stack_impl&&) noexcept;
            cyclic_windowed_stack_impl& operator=(cyclic_windowed_stack_impl&&) noexcept;

            cyclic_windowed_stack_impl(const cyclic_windowed_stack_impl&) = delete;
            cyclic_windowed_stack_impl& operator=(const cyclic_windowed_stack_impl&) = delete;

            ~cyclic_windowed_stack_impl() override
            {
                reset();
            }

            // access the cyclic buffer
            const cyclic_windowed_buffer& buffer() const
            {
                return *this;
            }

        protected:
            virtual void on_window_complete_event(std::size_t, const void*, std::size_t) {}

            // bytes available in current window
            std::size_t available() const noexcept;

            // push the data pointer forward
            // cannot advance past the end of the current window
            std::size_t push_data(const void*, std::size_t);

            // shift to the next window, recording a sync function for the current window
            // this method will block if the next window is not available
            // if the buffer wraps, the last overlap_size bytes of data is replicated to the
            // front of the stack.
            void push_window(std::function<void()>);

            // sync all outstanding windows and resets the stack to the start of the buffer
            void reset();

            // access the top of the data stack
            void* data_top()
            {
                return m_data_top;
            }

            // access the start of the current window
            void* window_start()
            {
                return m_win_top;
            }

            // convenience function
            using cyclic_windowed_buffer::window_size;

        private:
            // todo: figure out a way to relax the constrain of blocking copy and replicate

            // function used to copy external data to the buffer
            // required to be a blocking call
            virtual void copy(void*, const void*, size_t) = 0;

            // internal buffer copy function: defaults to std::memcpy; override for cuda
            // required to be a blocking call
            virtual void replicate(void*, const void*, std::size_t) = 0;

            // access the top of the stack
            memory::addr_t top()
            {
                return m_data_top;
            }

            // return the unique window id for the current window
            // this is monotonically increasing even though the buffer wraps
            std::size_t window_id() const noexcept
            {
                return m_win_counter;
            }

            // recycle buffer - reset, sync and replicate
            void recycle_buffer();

            // sync the next window and push the sync stack
            void sync_and_shift();

            // end of the buffer
            memory::addr_t m_end;

            // location of first free byte in the stack
            // allowed range [m_win_top, m_win_top + window_size_in_bytes()]
            memory::addr_t m_data_top;

            // location of first byte that needs synchronization
            // allowed range [data() + shift_size(), data() + capacity() , shift_size()]
            memory::addr_t m_sync_top;

            // start of the current window
            // allowed range [data(), data() + capacity_in_bytes() - window_size_in_bytes(), shift_size()]
            memory::addr_t m_win_top;

            // count the number of windows that have been triggered
            // effectively a unqiue id for a window
            std::size_t m_win_counter;

            // synchronization functions
            std::queue<std::pair<memory::addr_t, std::function<void()>>> m_sync;
        };

    } // namespace detail

    template <typename MemoryType, typename ThreadType>
    class cyclic_windowed_stack;

    template <typename ThreadType>
    class cyclic_windowed_stack<memory::host_memory, ThreadType> : public detail::cyclic_windowed_stack_impl
    {
    public:
        using memory_type = memory::host_memory;

        cyclic_windowed_stack(memory::descriptor md, std::size_t window_size, std::size_t overlap_size)
        : cyclic_windowed_stack_impl(std::move(md), window_size, overlap_size)
        {
        }

        cyclic_windowed_stack(cyclic_windowed_stack&& other) noexcept
        : cyclic_windowed_stack_impl(std::move(other)) {}
        cyclic_windowed_stack& operator=(cyclic_windowed_stack&& other) noexcept
        {
            cyclic_windowed_stack_impl::operator=(std::move(other));
            return *this;
        }

        ~cyclic_windowed_stack() override {}

        using cyclic_windowed_stack_impl::buffer;

    private:
        void copy(void* dst, const void* src, std::size_t size) final override
        {
            std::memcpy(dst, src, size);
        }

        void replicate(void* dst, const void* src, std::size_t size) final override
        {
            std::memcpy(dst, src, size);
        }
    };

    template <typename MemoryType, typename ThreadType>
    class cyclic_windowed_reserved_stack : private cyclic_windowed_stack<MemoryType, ThreadType>
    {
        using stack = cyclic_windowed_stack<MemoryType, ThreadType>;

        using promise_t = typename ThreadType::template promise<void>;
        using future_t  = typename ThreadType::template future<void>;

    public:
        cyclic_windowed_reserved_stack() {}

        cyclic_windowed_reserved_stack(stack&& s) : stack(std::move(s)) {}

        ~cyclic_windowed_reserved_stack() override {}

        // allow access back to the underlying buffer
        using stack::buffer;

        struct reservation;

        const reservation reserve_window()
        {
            CHECK(stack::window_size());
            if (m_future.valid())
            {
                m_future.get();
                stack::push_window([] {} /* empty sync fn */);
            }

            // prepare promise/future combo
            promise_t promise;
            m_future = promise.get_future();

            // build reservation
            reservation r(stack::window_start(), stack::window_size(), stack::data_top(), stack::available(), std::move(promise));

            return r;
        }

        void reset()
        {
            if(m_future.valid())
            {
                m_future.get();
            }
            stack::reset();
        }

        struct reservation
        {
            reservation() {}
            reservation(void* wstart, std::size_t wsize, void* dstart, std::size_t dsize, promise_t&& promise)
            : window_start(wstart), window_size(wsize), data_start(dstart), data_size(dsize), m_promise(std::move(promise))
            {
            }

            virtual ~reservation() {}

            reservation(reservation&&) noexcept = default;
            reservation& operator=(reservation&&) noexcept = default;

            void*       window_start;
            std::size_t window_size;
            void*       data_start;
            std::size_t data_size;

            void release() const
            {
                DCHECK(window_start);
                m_promise.set_value();
                // nullify pointers and zero out sizes?
            }

        private:
            mutable promise_t m_promise;
        };

    private:
        future_t m_future;
    };

    // task executor

    template <typename MemoryType, typename ThreadType>
    class cyclic_windowed_task_executor : private cyclic_windowed_stack<MemoryType, ThreadType>
    {
        using stack           = cyclic_windowed_stack<MemoryType, ThreadType>;
        using shared_future_t = typename ThreadType::template shared_future<void>;

    public:
        //cyclic_windowed_task_executor();
        cyclic_windowed_task_executor(stack&& s) : stack(std::move(s)) {}

        cyclic_windowed_task_executor(const cyclic_windowed_task_executor&) = delete;
        cyclic_windowed_task_executor& operator=(const cyclic_windowed_task_executor&) = delete;

        cyclic_windowed_task_executor(cyclic_windowed_task_executor&& other) noexcept : stack(std::move(other)) {}
        cyclic_windowed_task_executor& operator=(cyclic_windowed_task_executor&& other) noexcept
        {
            stack::operator=(std::move(other));
            return *this;
        }

        ~cyclic_windowed_task_executor() override {}

        // allow access back to the underlying buffer
        using stack::buffer;

        // write data to the buffer
        // as windows are filled the task function is run on the window data
        // the task function must return a shared_future that be used to sync the window
        // the completion of the future signifies that the data in the window can be reused
        void append_data(const void* data, std::size_t size);

        // TODO: requires some memory_utilities<MemoryType>
        // writes val the the remaining entries in the current window
        // and triggers a task execution
        // void flush(T val) {}

        // TODO: requires some memory_utilities<MemoryType>
        // pre populate some portion the first window
        // only allowed before data is pushed/appended
        // count must be less than the size of the first window
        // void pre_populate(T val, std::size_t count) {}

        // sync and reset the stack
        void reset()
        {
            stack::reset();
        }

    private:
        virtual shared_future_t on_compute_window(std::size_t, const void*, std::size_t) = 0;
        void                    on_window_complete_event(std::size_t, const void*, std::size_t) final override;
    };

    template <typename MemoryType, typename ThreadType>
    void cyclic_windowed_task_executor<MemoryType, ThreadType>::append_data(const void* data, std::size_t size)
    {
        memory::addr_t src = reinterpret_cast<memory::addr_t>(const_cast<void*>(data));

        while (size)
        {
            auto bytes = std::min(size, stack::available());
            size -= stack::push_data(src, bytes);
            src += bytes;
        }
    }

    template <typename MemoryType, typename ThreadType>
    void cyclic_windowed_task_executor<MemoryType, ThreadType>::on_window_complete_event(std::size_t id, const void* data, std::size_t size)
    {
        auto f = on_compute_window(id, data, size);
        stack::push_window([f] { f.get(); });
    }

} // namespace trtlab