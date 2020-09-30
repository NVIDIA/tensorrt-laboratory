#pragma once
#include <cstdlib>
#include <memory>
#include <experimental/propagate_const>

namespace trtlab
{
    namespace memory
    {
        struct size_tracker
        {
            size_tracker();
            ~size_tracker();

            size_tracker(const size_tracker&) = delete;
            size_tracker& operator=(const size_tracker&) = delete;

            size_tracker(size_tracker&&) noexcept;
            size_tracker& operator=(size_tracker&&) noexcept;

            void on_node_allocation(void* ptr, std::size_t size, std::size_t alignment) noexcept;
            void on_node_deallocation(void* ptr, std::size_t size, std::size_t alignment) noexcept;
            void on_array_allocation(void* ptr, std::size_t count, std::size_t size, std::size_t alignment) noexcept;
            void on_array_deallocation(void* ptr, std::size_t count, std::size_t size, std::size_t alignment) noexcept;

            std::size_t bytes() const noexcept;

        private:
            struct impl;
            std::experimental::propagate_const<std::unique_ptr<impl>> pimpl;
        };

        /*

// todo

struct histogram_tracker
{
    histogram_tracker();
    ~histogram_tracker();

    histogram_tracker(const histogram_tracker&) = delete;
    histogram_tracker& operator=(const histogram_tracker&) = delete;

    histogram_tracker(histogram_tracker&&) noexcept;
    histogram_tracker& operator=(histogram_tracker&&) noexcept;

    void on_node_allocation(void* ptr, std::size_t size, std::size_t alignment) noexcept;
    void on_node_deallocation(void* ptr, std::size_t size, std::size_t alignment) noexcept;
    void on_array_allocation(void* ptr, std::size_t count, std::size_t size, std::size_t alignment) noexcept;
    void on_array_deallocation(void* ptr, std::size_t count, std::size_t size, std::size_t alignment) noexcept;

    std::size_t bytes() const noexcept;

private:
    struct impl;
    std::experimental::propagate_const<std::unique_ptr<impl>> pimpl;
};

*/
    } // namespace memory
} // namespace trtlab