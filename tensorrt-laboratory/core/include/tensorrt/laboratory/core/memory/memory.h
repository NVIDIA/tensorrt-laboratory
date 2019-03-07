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
#include <string>

namespace trtlab {

/**
 * @brief Abstract base Memory class
 *
 * Abstract base class that tracks the base pointer and size of a memory allocation.
 *
 * Methods exposed by IMemory shall be implemented and available to all memory types.
 * Methods that do not depend on the specific type of memory should be implemented in
 * CoreMemory, while those that depend on the specialization, e.g. CUDA, should be
 * implemented in a class derived from BaseMemory.
 *
 * Two BaseMemory types are defined:
 * - HostMemory
 * - DeviceMemory (available in the cuda extension)
 *
 * Specific memory types are derived from these base classes and must implement the
 * IAllocatable interface and are expected to implement protected constructor, move
 * construstors and the move assignment operator.
 *
 * In order to allocate specific memory one should use the templated Allocator class.
 *
 * ```
 * auto allocated = std::make_unique<Allocator<Malloc>>(one_mb);
 * ```
 *
 * To expose memory own by another object, the object should create a descriptor object
 * derived from the templated Descriptor class. It is the responsibility of the object
 * providing the memory descriptor to ensure the lifecycle of the descriptor and
 * the object owning the memory is enforced.  See SmartStack for details on how
 * the SmartStack::Allocate methods returns a StackDescriptor which own a reference
 * (shared_ptr) to the stack ensuring the stack cannot deallocated be until all
 * StackDescriptors are released.
 */
class CoreMemory
{
  protected:
    CoreMemory(void* ptr, size_t size, bool allocated);

    CoreMemory(CoreMemory&& other) noexcept;
    CoreMemory& operator=(CoreMemory&&) noexcept;

    CoreMemory(const CoreMemory&) = delete;
    CoreMemory& operator=(const CoreMemory&) = delete;

  public:
    virtual ~CoreMemory();

    inline void* Data() { return m_MemoryAddress; }

    inline const void* Data() const { return m_MemoryAddress; }

    inline size_t Size() const { return m_BytesAllocated; }

    inline bool Allocated() const { return m_Allocated; }

    void* operator[](size_t offset);
    const void* operator[](size_t offset) const;

    template<typename T>
    T* CastToArray()
    {
        return static_cast<T*>(Data());
    }

    template<typename T>
    const T* CastToArray() const
    {
        return static_cast<const T*>(Data());
    }

    // Implemented by every unique derived class
    virtual const std::string& Type() const = 0;

    // Implemented by BaseMemory classes, e.g. HostMemory or DeviceMemory
    virtual void Fill(char) = 0;

  private:
    void* m_MemoryAddress;
    size_t m_BytesAllocated;
    bool m_Allocated;
};

template<class MemoryType>
class BaseMemory : public CoreMemory
{
  public:
    using CoreMemory::CoreMemory;
    using BaseType = MemoryType;

    static size_t AllocationSizeWithAlignment(size_t);

    /*
        // This does not work :-/
        template<typename T>
        static size_t AllocationSizeWithAlignment(size_t count_of_type)
        {
            size_t size = count_of_type * sizeof(T);
            return MemoryType::AllocationSizeWithAlignment(size);
        }
    */
};

template<class MemoryType>
size_t BaseMemory<MemoryType>::AllocationSizeWithAlignment(size_t size_in_bytes)
{
    size_t alignment = MemoryType::DefaultAlignment();
    size_t remainder = size_in_bytes % alignment;
    return (remainder == 0) ? size_in_bytes : size_in_bytes + alignment - remainder;
}

} // namespace trtlab
