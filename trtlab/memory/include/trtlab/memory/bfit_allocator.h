#pragma once
#include <set>
#include <utility>
#include <experimental/propagate_const>

#include <glog/logging.h>

#include "align.h"
#include "memory_block.h"
#include "utils.h"

namespace trtlab
{
    namespace memory
    {
        namespace bfit_detail
        {
            // node for double linked list holding all memory blocks - both allocated and free
            // offset allows for aligned allocation if the block start is not aligned as requested
            struct memory_node : public memory_block
            {
                memory_node() : memory_block(), is_allocated(false), prev_node(nullptr), next_node(nullptr), offset(0) {}
                memory_node(const memory_block& block)
                : memory_block(block), is_allocated(false), prev_node(nullptr), next_node(nullptr), offset(0)
                {
                }

                memory_node(const memory_node&) = delete;
                memory_node& operator=(const memory_node&) = delete;

                memory_node(memory_node&&) = delete;
                memory_node& operator=(memory_node&&) = delete;

                // memory_block::memory will is the start of the data segment
                // i.e. the value of the pointer returned to the user by allocate

                // memory_block::size is the entire size of the block including any
                // left and right padding

                bool         is_allocated;
                memory_node* prev_node;
                memory_node* next_node;
                std::size_t  offset;
            };

            template <typename Compare = std::less<>>
            struct memory_node_compare_size : public memory_block_compare_size<Compare>
            {
                constexpr bool operator()(std::size_t size, const memory_node* node) const
                {
                    return memory_block_compare_size<Compare>::operator()(size, *node);
                }

                constexpr bool operator()(const memory_node* node, std::size_t size) const
                {
                    return memory_block_compare_size<Compare>::operator()(*node, size);
                }

                constexpr bool operator()(const memory_node* lhs, const memory_node* rhs) const
                {
                    return memory_block_compare_size<Compare>::operator()(*lhs, *rhs);
                }
            };

            template <typename Compare = std::less<>>
            struct memory_node_compare_addr : public memory_block_compare_addr<Compare>
            {
                constexpr bool operator()(void* addr, const memory_node* node) const
                {
                    return memory_block_compare_addr<Compare>::operator()(addr, *node);
                }

                constexpr bool operator()(const memory_node* node, void* addr) const
                {
                    return memory_block_compare_addr<Compare>::operator()(*node, addr);
                }

                constexpr bool operator()(const memory_node* lhs, const memory_node* rhs) const
                {
                    return memory_block_compare_addr<Compare>::operator()(*lhs, *rhs);
                }
            };

            class node_in_place
            {
            };

            class node_external
            {
            };

        } // namespace bfit_detail

        class bfit_options
        {
        public:
            bfit_options();

            bfit_options(const bfit_options&) = delete;
            bfit_options& operator=(const bfit_options&) = delete;

            bfit_options(bfit_options&&) noexcept;
            bfit_options& operator=(bfit_options&&) noexcept;

            void set_initial_size(std::size_t);
            void set_max_size(std::size_t);
            void set_allow_growth(bool);
            void set_growth_factor(float);

            std::size_t initial_size() const;
            std::size_t max_size() const;
            bool        allow_growth() const;
            float       growth_factor() const;

        private:
            class impl;
            std::experimental::propagate_const<std::unique_ptr<impl>> p_impl;
        };

        template <typename RawAllocator>
        class bfit_allocator : TRTLAB_EBO(RawAllocator)
        {
            using node_t = bfit_detail::memory_node;
            using traits = allocator_traits<RawAllocator>;

            struct stats;

        public:
            using is_stateful    = std::true_type;
            using memory_type    = typename traits::memory_type;
            using allocator_type = typename traits::allocator_type;

            bfit_allocator(std::size_t initial_size, RawAllocator&& alloc) : allocator_type(std::move(alloc))
            {
                m_min_alignment = traits::min_alignment(get_allocator());
                auto memory     = traits::allocate_node(get_allocator(), initial_size, m_min_alignment);
                auto node       = priv_create_node({memory, initial_size});
                auto root       = priv_create_node({memory, initial_size});

                // special case: mark as allocated to avoid being merged during a free
                root->is_allocated = true;
                root->next_node    = node;
                node->prev_node    = root;

                m_heads.push_back(root);
                m_free_nodes.insert(node);
            }

            ~bfit_allocator()
            {
                if (!m_heads.empty())
                {
                    auto& alloc = get_allocator();
                    VLOG(3) << "deallocating " << m_heads.size() << " root allocations";
                    for (auto& block_ptr : m_heads)
                    {
                        DCHECK(block_ptr->next_node);
                        DCHECK(!block_ptr->next_node->next_node);
                        traits::deallocate_node(alloc, block_ptr->next_node->memory, block_ptr->next_node->size, m_min_alignment);
                    }
                }
            }

            bfit_allocator(const bfit_allocator&) = delete;
            bfit_allocator& operator=(const bfit_allocator&) = delete;

            bfit_allocator(bfit_allocator&&) = default;
            bfit_allocator& operator=(bfit_allocator&&) = default;

            void* allocate_node(std::size_t size, std::size_t alignment)
            {
                auto search = m_free_nodes.lower_bound(size);
                if (search == m_free_nodes.end())
                {
                    // no memory block available to handle request
                    // trigger fallback -- possible fallbacks:
                    // 1) request caches to be depopulated
                    // 2) attempt to grow the allocation pool
                    // for now, just fail
                    LOG(ERROR) << "unable to find block to fullfil request of " << size << " bytes";
                    throw std::bad_alloc();
                }

                // partition the best fit memory block
                auto [alloc_node, free_node] = priv_split_node(*search, size, alignment);
                DCHECK(alloc_node) << "internal allocation failure";

                // remove from free nodes
                m_free_nodes.erase(search);

                // insert new allocation in to allocated nodes
                m_alloc_nodes.insert(alloc_node);

                // if sufficient remaining free space, add to free nodes
                if (free_node)
                {
                    m_free_nodes.insert(free_node);
                }

                DVLOG(3) << "allocate_node succeeded: " << alloc_node->memory << " - " << bytes_to_string(alloc_node->size) << " bytes";
                return alloc_node->memory;
            }

            void deallocate_node(void* memory, std::size_t size, std::size_t alignment) noexcept
            {
                DVLOG(5) << "deallocating node at " << memory;

                // todo: this lookup could be skipped entirely if we allocate a descriptor that holds
                // the linked list node info as part of the descriptor
                auto search = m_alloc_nodes.find(memory);
                if (search == m_alloc_nodes.end())
                {
                    LOG(ERROR) << "failed to find allocation for " << memory;
                    auto node = debug_scan_for_node(memory);
                    if (node)
                    {
                        LOG(ERROR) << "found node containing " << memory << "; marked as " << (node->is_allocated ? "ALLOCATED" : "FREED");
                    }
                    LOG(FATAL) << "this is a fatal allocation error: aborting...";
                }

                // note: erasing the pointer from alloc_nodes does not free the node's allocation
                node_t* free_node = *search;
                m_alloc_nodes.erase(search);

                // mark as deallocated
                free_node->is_allocated = false;

                // remove any offset - the memory_block starting addres and size make it contiguous
                // with its left and right neighbors
                priv_collapse_alignment(free_node);

                // merge with neighboring free/deallocated blocks
                if (free_node->next_node && free_node->next_node->is_allocated == false)
                {
                    DVLOG(5) << "merge current node (" << free_node << ") with right neighbor (" << free_node->next_node << ")";
                    // erasing the node only removes the pointer from the set; it does not deallocate it
                    CHECK_EQ(m_free_nodes.erase(free_node->next_node), 1);
                    auto destroy_node = free_node->next_node;
                    priv_merge_node(free_node, free_node->next_node);
                    priv_destroy_node(destroy_node);
                }
                if (free_node->prev_node && free_node->prev_node->is_allocated == false)
                {
                    DVLOG(5) << "merge left node (" << free_node->prev_node << ") with current node (" << free_node << ")";
                    // erasing the node only removes the pointer from the set; it does not deallocate it
                    CHECK_EQ(m_free_nodes.erase(free_node->prev_node), 1);
                    free_node         = free_node->prev_node;
                    auto destroy_node = free_node->next_node;
                    priv_merge_node(free_node, free_node->next_node);
                    priv_destroy_node(destroy_node);
                }

                m_free_nodes.insert(free_node);
            }

            allocator_type& get_allocator()
            {
                return *this;
            }

            const allocator_type& get_allocator() const
            {
                return *this;
            }

            std::string debug_print_allocate_blocks() const
            {
                return priv_list_of_nodes(m_alloc_nodes);
            }

            std::size_t free_nodes() const
            {
                return m_free_nodes.size();
            }

            std::size_t used_nodes() const
            {
                return m_alloc_nodes.size();
            }

            const node_t* debug_scan_for_node(void* memory)
            {
                for (auto& head : m_heads)
                {
                    auto node = head;
                    while (true)
                    {
                        node = node->next_node;
                        if (!node)
                            break;
                        if (node->contains(memory))
                            return node;
                    }
                }
                return nullptr;
            }

            bool allocation_found(void* memory)
            {
                auto search = m_alloc_nodes.find(memory);
                return search != m_alloc_nodes.end();
            }

        private:
            std::pair<node_t*, node_t*> priv_split_node(const node_t* bfit, std::size_t size, std::size_t alignment)
            {
                DVLOG(5) << "spliting node with " << bfit->size << " bytes into allocation of " << size << " bytes";

                // create one or two new nodes in the double linked list
                // when complete, the bfit node is removed from the list
                auto alloc_node            = priv_create_node(*bfit);
                auto [data_start, loffset] = align_shift(alloc_node->memory, alignment);

                // the loffset + size was larger than the block (unlikely)
                if (size + loffset > bfit->size)
                    return std::pair<node_t*, node_t*>(nullptr, nullptr);

                auto data_end              = memory_shift(data_start, size);
                auto [free_start, roffset] = align_shift(data_end, m_min_alignment);

                alloc_node->memory       = data_start;
                alloc_node->offset       = loffset;
                alloc_node->size         = loffset + size + roffset;
                alloc_node->is_allocated = true;

                DVLOG(10) << "new used_node; memory=" << data_start << "; size=" << alloc_node->size << "; offset=" << loffset;

                // partial linked_list update; insert alloc_node after bfit->prev_node
                // [bfit->prev_node] [alloc_node]
                alloc_node->prev_node = bfit->prev_node;
                if (bfit->prev_node)
                    bfit->prev_node->next_node = alloc_node;

                // compute the remaining free space
                std::size_t free_size = bfit->size - alloc_node->size;

                if (free_size < 1024)
                {
                    DVLOG(5) << "remaining free space is not sufficent for a free_node: " << free_size;
                    // complete linked list update
                    // [bfit->prev_node] [alloc_node] [bfit->next_node]
                    alloc_node->next_node = bfit->next_node;
                    if (bfit->next_node)
                        bfit->next_node->prev_node = alloc_node;
                    return std::pair<node_t*, node_t*>(alloc_node, nullptr);
                }

                // there is enough free memory in the allocated block to warrant a new free node
                auto free_node = priv_create_node({free_start, free_size});

                // insert free_node into linked list after alloc_node
                // [bfit->prev_node] [alloc_node] [free_node]
                alloc_node->next_node = free_node;
                free_node->prev_node  = alloc_node;

                // complete final linked list update
                // [bfit->prev_node] [alloc_node] [free_node] [bfit->next_node]
                free_node->next_node = bfit->next_node;
                if (bfit->next_node)
                    bfit->next_node->prev_node = free_node;

                return std::make_pair(alloc_node, free_node);
            }

            void priv_merge_node(node_t* dst, node_t* src)
            {
                CHECK(!src->is_allocated);
                DCHECK_EQ(memory_shift(dst->memory, dst->size), src->memory);
                DCHECK_EQ(src->prev_node, dst);
                DCHECK_EQ(src->offset, 0);
                DCHECK(!dst->is_allocated);

                // extend the dst node to include the src node
                dst->size += src->size;

                // remove src from the linked list
                dst->next_node = src->next_node;
                if (dst->next_node)
                    dst->next_node->prev_node = dst;
            }

            void priv_collapse_alignment(node_t* node)
            {
                if (node->offset)
                {
                    node->memory = memory_shift(node->memory, -1 * node->offset);
                    node->offset = 0;
                }
            }

            node_t* priv_create_node(const memory_block& block)
            {
                // this is where you call your templated node allocator
                // whether the node is allocated "in-block", i.e. as part of the actuall backing memory
                // or if th node is allocated "out-of-block", i.e. the node data is stored apart from the backing memory
                // the latter is needed for gpu memory

                auto node = new node_t(block);

                // move to constructor
                node->memory       = block.memory;
                node->size         = block.size;
                node->offset       = 0;
                node->is_allocated = false;
                node->next_node    = nullptr;
                node->prev_node    = nullptr;

                return node;
            }

            void priv_destroy_node(node_t* node) noexcept
            {
                // this is where you call your templated node deallocator
                delete node;
            }

            template <typename Container>
            std::string priv_list_of_nodes(const Container& container) const
            {
                std::stringstream os;
                os << std::endl;
                for (auto& item : container)
                {
                    os << item->memory << " - " << item->size << " - " << item->offset << std::endl;
                }
                return os.str();
            }

            std::size_t                                                m_min_alignment;
            std::vector<const node_t*>                                 m_heads;
            std::set<node_t*, bfit_detail::memory_node_compare_size<>> m_free_nodes;
            std::set<node_t*, bfit_detail::memory_node_compare_addr<>> m_alloc_nodes;
        };

        template <typename RawAllocator>
        auto make_bfit_allocator(std::size_t initial_size, RawAllocator&& alloc)
        {
            return bfit_allocator<RawAllocator>(initial_size, std::move(alloc));
        }

    } // namespace memory
} // namespace trtlab
