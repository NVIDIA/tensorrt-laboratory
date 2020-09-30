
#include "memory_type.h"
#include "ilog2.h"

using namespace trtlab::memory;

std::size_t detail::any_memory::ilog2(std::size_t size) noexcept
{
    return ::trtlab::memory::detail::ilog2(size);
}
