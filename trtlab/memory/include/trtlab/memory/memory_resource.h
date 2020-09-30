
#pragma once

#include <cstddef>
#include <utility>

#include "descriptor.h"

namespace trtlab {
namespace memory {
/**---------------------------------------------------------------------------*
 * @brief Base class for host memory allocation.
 *
 * This is based on `std::pmr::memory_resource`:
 * https://en.cppreference.com/w/cpp/memory/memory_resource
 *
 * When C++17 is available for use in RMM, `rmm::memory_resource` should
 * inherit from `std::pmr::memory_resource`.
 *
 * This class serves as the interface that all host memory resource
 * implementations must satisfy.
 *
 * There are two private, pure virtual functions that all derived classes must
 * implement: `do_allocate` and `do_deallocate`. Optionally, derived classes may
 * also override `is_equal`. By default, `is_equal` simply performs an identity
 * comparison.
 *
 * The public, non-virtual functions `allocate`, `deallocate`, and `is_equal`
 * simply call the private virtual functions. The reason for this is to allow
 * implementing shared, default behavior in the base class. For example, the
 * base class' `allocate` function may log every allocation, no matter what
 * derived class implementation is used.
 *
 *---------------------------------------------------------------------------**/
class memory_resource {
 public:
  memory_resource() = default;
  virtual ~memory_resource() = default;

  /**---------------------------------------------------------------------------*
   * @brief Allocates memory on the host of size at least `bytes` bytes.
   *
   * The returned storage is aligned to the specified `alignment` if supported,
   * and to `alignof(std::max_align_t)` otherwise.
   *
   * @throws std::bad_alloc When the requested `bytes` and `alignment` cannot be
   * allocated.
   *
   * @param bytes The size of the allocation
   * @param alignment Alignment of the allocation
   * @return void* Pointer to the newly allocated memory
   *---------------------------------------------------------------------------**/
  void* allocate(std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t)) {
    return do_allocate(bytes, alignment);
  }

  /**---------------------------------------------------------------------------*
   * @brief Deallocate memory pointed to by `p`.
   *
   * `p` must have been returned by a prior call to `allocate(bytes,alignment)`
   * on a `memory_resource` that compares equal to `*this`, and the storage
   * it points to must not yet have been deallocated, otherwise behavior is
   * undefined.
   *
   * @throws Nothing.
   *
   * @param p Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param alignment Alignment of the allocation. This must be equal to the
   *value of `alignment` that was passed to the `allocate` call that returned
   *`p`.
   * @param stream Stream on which to perform deallocation
   *---------------------------------------------------------------------------**/
  void deallocate(void* p, std::size_t bytes,
                  std::size_t alignment = alignof(std::max_align_t)) {
    do_deallocate(p, bytes, alignment);
  }

  /**---------------------------------------------------------------------------*
   * @brief Compare this resource to another.
   *
   * Two `memory_resource`s compare equal if and only if memory allocated
   * from one `memory_resource` can be deallocated from the other and vice
   * versa.
   *
   * By default, simply checks if \p *this and \p other refer to the same
   * object, i.e., does not check if they are two objects of the same class.
   *
   * @param other The other resource to compare to
   * @returns If the two resources are equivalent
   *---------------------------------------------------------------------------**/
  bool is_equal(memory_resource const& other) const noexcept {
    return do_is_equal(other);
  }

 private:
  /**---------------------------------------------------------------------------*
   * @brief Allocates memory on the host of size at least `bytes` bytes.
   *
   * The returned storage is aligned to the specified `alignment` if supported,
   * and to `alignof(std::max_align_t)` otherwise.
   *
   * @throws std::bad_alloc When the requested `bytes` and `alignment` cannot be
   * allocated.
   *
   * @param bytes The size of the allocation
   * @param alignment Alignment of the allocation
   * @return void* Pointer to the newly allocated memory
   *---------------------------------------------------------------------------**/
  virtual void* do_allocate(
      std::size_t bytes, std::size_t alignment = alignof(std::max_align_t)) = 0;

  /**---------------------------------------------------------------------------*
   * @brief Deallocate memory pointed to by `p`.
   *
   * `p` must have been returned by a prior call to `allocate(bytes,alignment)`
   * on a `memory_resource` that compares equal to `*this`, and the storage
   * it points to must not yet have been deallocated, otherwise behavior is
   * undefined.
   *
   * @throws Nothing.
   *
   * @param p Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param alignment Alignment of the allocation. This must be equal to the
   *value of `alignment` that was passed to the `allocate` call that returned
   *`p`.
   * @param stream Stream on which to perform deallocation
   *---------------------------------------------------------------------------**/
  virtual void do_deallocate(
      void* p, std::size_t bytes,
      std::size_t alignment = alignof(std::max_align_t)) = 0;

  /**---------------------------------------------------------------------------*
   * @brief Compare this resource to another.
   *
   * Two host_memory_resources compare equal if and only if memory allocated
   * from one memory_resource can be deallocated from the other and vice
   * versa.
   *
   * By default, simply checks if \p *this and \p other refer to the same
   * object, i.e., does not check if they are two objects of the same class.
   *
   * @param other The other resource to compare to
   * @return true If the two resources are equivalent
   * @return false If the two resources are not equal
   *---------------------------------------------------------------------------**/
  virtual bool do_is_equal(memory_resource const& other) const noexcept {
    return this == &other;
  }
};

class raii_memory_resource
{
public:
  raii_memory_resource() = default;
  virtual ~raii_memory_resource() = default;

  virtual descriptor allocate_descriptor(std::size_t size) = 0;
  virtual descriptor allocate_descriptor(std::size_t size, std::size_t alignment) = 0;
};

class device_context_interface
{
public:
  device_context_interface() = default;
  virtual ~device_context_interface() = default;

  virtual DLContext  device_context() const = 0;
};

}  // namespace memory
}  // namespace trtlab