
#include <algorithm>
#include <queue>

#include "cyclic_windowed_buffer.h"

#include <glog/logging.h>

using namespace trtlab;
using namespace detail;

cyclic_windowed_buffer::cyclic_windowed_buffer() : m_window_size(0), m_window_count(0), m_shift_size(0), m_capacity(0) {}

cyclic_windowed_buffer::cyclic_windowed_buffer(memory::descriptor md, std::size_t window_size, std::size_t overlap)
: m_descriptor(std::move(md)), m_window_size(window_size), m_shift_size(window_size - overlap)
{
    CHECK_LE(window_size, m_descriptor.size()) << "allocated memory must be >= the window_size";
    CHECK_LT(overlap, window_size) << "requested overlap=" << overlap << "; must be [0 <= overlap < " << window_size << ")";

    m_window_count = (m_descriptor.size() - m_window_size) / m_shift_size + 1;
    DVLOG(1) << "Allocation of " << m_descriptor.size() << " bytes allows for " << m_window_count << " cyclic buffer windows";

    m_capacity = m_window_size + (m_window_count - 1) * m_shift_size;
    DVLOG(2) << "Effective allocation size " << m_capacity << " usable bytes";

    DVLOG(2) << "window_size: " << m_window_size << "; shift_size: " << m_shift_size;
}

cyclic_windowed_buffer::cyclic_windowed_buffer(cyclic_windowed_buffer&& other) noexcept
: m_descriptor(std::move(other.m_descriptor)),
  m_window_size(std::exchange(other.m_window_size, 0UL)),
  m_shift_size(std::exchange(other.m_shift_size, 0UL)),
  m_window_count(std::exchange(other.m_window_count, 0UL)),
  m_capacity(std::exchange(other.m_capacity, 0UL))
{
}

cyclic_windowed_buffer& cyclic_windowed_buffer::operator=(cyclic_windowed_buffer&& other) noexcept
{
    m_descriptor   = std::move(other.m_descriptor);
    m_window_size  = std::exchange(other.m_window_size, 0UL);
    m_shift_size   = std::exchange(other.m_shift_size, 0UL);
    m_window_count = std::exchange(other.m_window_count, 0UL);
    m_capacity     = std::exchange(other.m_capacity, 0UL);
    return *this;
}


std::size_t cyclic_windowed_buffer::min_allocation_size(std::size_t window_count, std::size_t window_size, std::size_t overlap)
{
    CHECK_GT(window_count, 0);
    CHECK_GT(window_size, overlap);
    return window_size + (window_count - 1) * (window_size - overlap);
}

// stack impl

cyclic_windowed_stack_impl::cyclic_windowed_stack_impl() : m_end(nullptr), m_data_top(nullptr), m_sync_top(nullptr), m_win_top(nullptr), m_win_counter(0) {}

cyclic_windowed_stack_impl::cyclic_windowed_stack_impl(memory::descriptor md, std::size_t window_size, std::size_t overlap)
: cyclic_windowed_buffer(std::move(md), window_size, overlap),
  m_end(data() + capacity()),
  m_data_top(data()),
  m_sync_top(m_end),
  m_win_top(data()),
  m_win_counter(0)
{
}

/*
cyclic_windowed_stack_impl::cyclic_windowed_stack_impl(cyclic_windowed_buffer&& buffer)
: cyclic_windowed_buffer(std::move(buffer)),
  m_end(data() + capacity()),
  m_data_top(data()),
  m_sync_top(m_end),
  m_win_top(data()),
  m_win_counter(0)
{
}
*/


cyclic_windowed_stack_impl::cyclic_windowed_stack_impl(cyclic_windowed_stack_impl&& other) noexcept
: cyclic_windowed_buffer(std::move(other)),
  m_end(std::exchange(other.m_end, nullptr)),
  m_data_top(std::exchange(other.m_data_top, nullptr)),
  m_sync_top(std::exchange(other.m_sync_top, nullptr)),
  m_win_top(std::exchange(other.m_win_top, nullptr)),
  m_win_counter(std::exchange(other.m_win_counter, 0UL))
{
}

cyclic_windowed_stack_impl& cyclic_windowed_stack_impl::operator=(cyclic_windowed_stack_impl&& other) noexcept
{
    cyclic_windowed_buffer::operator=(std::move(other));
    m_end                           = std::exchange(other.m_end, nullptr);
    m_data_top                      = std::exchange(other.m_data_top, nullptr);
    m_sync_top                      = std::exchange(other.m_sync_top, nullptr);
    m_win_top                       = std::exchange(other.m_win_top, nullptr);
    m_win_counter                   = std::exchange(other.m_win_counter, 0UL);
    return *this;
}


std::size_t cyclic_windowed_stack_impl::available() const noexcept
{
    DCHECK_GE(offset(m_data_top), 0);
    DCHECK_LE(offset(m_data_top), capacity());
    DCHECK_GE(offset(m_sync_top), shift_size());
    DCHECK_LE(offset(m_sync_top), capacity());
    //DCHECK_EQ(offset(m_sync_top) % overlap_size(), 0);
    DCHECK_GE(offset(m_win_top), 0);
    DCHECK_LE(offset(m_win_top), capacity() - window_size());
    //DCHECK_EQ(offset(m_win_top) % overlap_size(), 0);

    DCHECK_LE(offset(m_data_top), offset(m_sync_top));

    std::ptrdiff_t avail     = m_sync_top - m_data_top;
    std::ptrdiff_t in_use    = m_data_top - m_win_top;
    std::ptrdiff_t remaining = window_size() - in_use;

    return std::min(avail, remaining);
}

void cyclic_windowed_stack_impl::push_window(std::function<void()> sync)
{
    DVLOG(1) << "push window: " << window_id() << " - start offset = " << offset(m_win_top);
    DCHECK_GE(offset(m_data_top), 0);
    DCHECK_LE(offset(m_data_top), capacity());
    DCHECK_GE(offset(m_sync_top), shift_size());
    DCHECK_LE(offset(m_sync_top), capacity());
    //DCHECK_EQ(offset(m_sync_top) % overlap_size(), 0);
    DCHECK_GE(offset(m_win_top), 0);
    DCHECK_LE(offset(m_win_top), capacity() - window_size());
    //DCHECK_EQ(offset(m_win_top) % overlap_size(), 0);

    DVLOG(1) << "push window: " << window_id() << " - start offset = " << offset(m_win_top);

    // record win_start and sync function for the current window
    m_sync.push(std::make_pair(m_win_top, sync));

    // move data pointer to end of current window
    m_data_top = m_win_top + window_size();

    // push window pointer
    m_win_top += shift_size();

    // if we are at the end, then recycle
    if (m_data_top == m_end)
    {
        recycle_buffer();
    }

    // ensure the next window is free
    while (m_sync_top < m_win_top + window_size())
    {
        sync_and_shift();
    }

    DVLOG(3) << "push window complete";

    // increrement win_count;
    m_win_counter++;
}

std::size_t cyclic_windowed_stack_impl::push_data(const void* src, std::size_t bytes)
{
    DCHECK_LE(m_data_top + bytes, m_sync_top);
    DCHECK_LE(m_data_top + bytes, m_win_top + window_size());

    // copy data to buffer
    copy(m_data_top, src, bytes);

    // push data stack
    m_data_top += bytes;

    // if we completed a window, trigger an event
    if (m_data_top == m_win_top + window_size())
    {
        on_window_complete_event(window_id(), m_win_top, window_size());
    }

    return bytes;
}

void cyclic_windowed_stack_impl::sync_and_shift()
{
    DCHECK_GE(offset(m_sync_top), 0);
    DCHECK_LE(offset(m_sync_top), capacity());
    //DCHECK_EQ(offset(m_sync_top) % shift_size(), 0);

    auto [start, syncfn] = m_sync.front();
    m_sync.pop();

    DCHECK_EQ(offset(start), offset(m_sync_top));
    DVLOG(2) << "sync next window - starting window offset = " << offset(start);

    // call sync function
    if (syncfn)
    {
        syncfn();
    }

    // push sync stack
    m_sync_top += shift_size();

    // the replication window does not need to be sync'd
    // no window starts at end - shift_size
    // that winwdow would overflow the buffer
    if (m_sync_top == m_end - overlap_size())
    {
        m_sync_top = m_end;
    }
}

void cyclic_windowed_stack_impl::recycle_buffer()
{
    DVLOG(2) << "end of buffer - recycle";
    DCHECK_EQ(m_data_top, m_end);

    // to keep the logic clean, we are going to push a no-op sync

    // reset stack pointers
    m_data_top = m_sync_top = m_win_top = data();

    if (overlap_size())
    {
        DVLOG(2) << "replicating " << overlap_size() << " bytes from the end to the beginning";

        // make space for the replicated data by syncing on windows from the last pass
        while (m_sync_top <= data() + overlap_size())
        {
            sync_and_shift();
        }

        // copy overlap_size bytes from the end to the beginning
        replicate(m_data_top, m_end - overlap_size(), overlap_size());

        // move data top to after the replicated data
        m_data_top += overlap_size();
    }
    DVLOG(3) << "recycle complete";
}

void cyclic_windowed_stack_impl::reset()
{
    while (!m_sync.empty())
    {
        auto [start, syncfn] = m_sync.front();
        if (syncfn)
        {
            syncfn();
        }
        m_sync.pop();
    }

    m_data_top    = data();
    m_sync_top    = m_end;
    m_win_top     = data();
    m_win_counter = 0;
}

// reservation
