#pragma once

#include <trtlab/core/cyclic_windowed_buffer.h>

#include "memory/device_memory.h"
#include "sync.h"

namespace trtlab
{
    template <typename ThreadType>
    class cyclic_windowed_stack<memory::device_memory, ThreadType> : public detail::cyclic_windowed_stack_impl
    {
    public:
        using memory_type = memory::device_memory;

        cyclic_windowed_stack() : cyclic_windowed_stack_impl(), m_Stream(nullptr) {}

        cyclic_windowed_stack(memory::descriptor md, std::size_t window_size, std::size_t overlap_size, cudaStream_t stream)
        : cyclic_windowed_stack_impl(std::move(md), window_size, overlap_size), m_Stream(stream)
        {
        }

        cyclic_windowed_stack(cyclic_windowed_stack&&) noexcept = default;
        cyclic_windowed_stack& operator=(cyclic_windowed_stack&&) noexcept = default;

        ~cyclic_windowed_stack() override {}

    private:
        void copy(void* dst, const void* src, std::size_t size) final override
        {
            CHECK_EQ(cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, m_Stream), cudaSuccess);
            cuda_sync<ThreadType>::stream_sync(m_Stream);
        }

        void replicate(void* dst, const void* src, std::size_t size) final override
        {
            CHECK_EQ(cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, m_Stream), cudaSuccess);
            cuda_sync<ThreadType>::stream_sync(m_Stream);
        }

        cudaStream_t m_Stream;
    };

} // namespace trtlab