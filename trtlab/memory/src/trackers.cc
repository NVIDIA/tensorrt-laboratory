
#include "trackers.h"
#include <atomic>

using namespace trtlab;
using namespace memory;

struct size_tracker::impl
{
    std::atomic<std::size_t> bytes = 0;
};

size_tracker::size_tracker() : pimpl(std::make_unique<size_tracker::impl>()) {}
size_tracker::~size_tracker() = default;

size_tracker::size_tracker(size_tracker&&) noexcept = default;
size_tracker& size_tracker::operator=(size_tracker&&) noexcept = default;

std::size_t size_tracker::bytes() const noexcept
{
    return pimpl->bytes.load();
}

void size_tracker::on_node_allocation(void* ptr, std::size_t size, std::size_t alignment) noexcept
{
    pimpl->bytes += size;
}

void size_tracker::on_node_deallocation(void* ptr, std::size_t size, std::size_t alignment) noexcept
{
    pimpl->bytes -= size;
}

void size_tracker::on_array_allocation(void* ptr, std::size_t count, std::size_t size, std::size_t alignment) noexcept
{
    pimpl->bytes += (count * size);
}

void size_tracker::on_array_deallocation(void* ptr, std::size_t count, std::size_t size, std::size_t alignment) noexcept
{
    pimpl->bytes -= (count * size);
}