#include "detail/block_list.h"

#include "align.h"
#include "detail/assert.h"
#include "error.h"

#include "free_list_utils.h"

using namespace trtlab::memory;
using namespace detail;


constexpr std::size_t block_list::min_element_size;
constexpr std::size_t block_list::min_element_alignment;

block_list::block_list() noexcept
: first_(nullptr), capacity_(0u)
{
}

block_list::block_list(block_list&& other) noexcept
: first_(std::exchange(other.first_, nullptr)), capacity_(std::exchange(other.capacity_, 0u))
{
}

block_list& block_list::operator=(block_list&& other) noexcept
{
    block_list tmp(detail::move(other));
    swap(*this, tmp);
    return *this;
}

void trtlab::memory::detail::swap(block_list& a, block_list& b) noexcept
{
    detail::adl_swap(a.first_, b.first_);
    detail::adl_swap(a.capacity_, b.capacity_);
}

void block_list::insert(memory_block&& block) noexcept
{
    TRTLAB_MEMORY_ASSERT(block.memory);
    TRTLAB_MEMORY_ASSERT(block.size > sizeof(node));
    DVLOG(4) << "block_list::insert " << block.memory;
    
    auto n = static_cast<node*>(block.memory);
    n->size = block.size;
    n->next = first_;
    first_ = n;
    capacity_++;
}

memory_block block_list::allocate() noexcept
{
    TRTLAB_MEMORY_ASSERT(!empty());
    --capacity_;

    memory_block block;
    block.memory = static_cast<void*>(first_);
    block.size = first_->size;
    first_ = first_->next;
    return block;
}

void block_list::deallocate(memory_block&& block) noexcept
{
    insert(std::move(block));
}


block_list_oob::block_list_oob() noexcept
{
}

block_list_oob::block_list_oob(block_list_oob&& other) noexcept
: nodes(std::move(other.nodes))
{
}

block_list_oob& block_list_oob::operator=(block_list_oob&& other) noexcept
{
    block_list_oob tmp(detail::move(other));
    swap(*this, tmp);
    return *this;
}

void trtlab::memory::detail::swap(block_list_oob& a, block_list_oob& b) noexcept
{
    detail::adl_swap(a.nodes, b.nodes);
}

void block_list_oob::insert(memory_block&& block) noexcept
{
    TRTLAB_MEMORY_ASSERT(block.memory);
    TRTLAB_MEMORY_ASSERT(block.size > sizeof(node));
    DVLOG(4) << "block_list_oob::insert " << block.memory;
    nodes.push_front(std::move(block));
}

memory_block block_list_oob::allocate() noexcept
{
    TRTLAB_MEMORY_ASSERT(!empty());
    memory_block block = nodes.front();
    nodes.pop_front();
    return block;
}

void block_list_oob::deallocate(memory_block&& block) noexcept
{
    insert(std::move(block));
}