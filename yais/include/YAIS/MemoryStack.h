/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#ifndef _YAIS_MEMORY_STACK_H_
#define _YAIS_MEMORY_STACK_H_

#include <vector>

#include "YAIS/Memory.h"
#include <glog/logging.h>

namespace yais
{

class IMemoryStack
{
  public:
    virtual ~IMemoryStack() {}
    virtual void *Allocate(size_t) = 0;
};

/**
 * @brief General MemoryStack
 * 
 * Memory stack using a single memory allocation of AllocatorType.  The stack pointer is advanced
 * by the Allocate function.  Each new stack pointer is aligned to either the DefaultAlignment of
 * the AllocatorType or customized when the MemoryStack is instantiated.
 * 
 * @tparam AllocatorType 
 */
template <class AllocatorType>
class MemoryStack : public IMemoryStack
{
  protected:
    /**
     * @brief Construct a new MemoryStack object
     * 
     * A stack using a single allocation of AllocatorType.  The stack can only be advanced or reset.
     * Popping is not supported.
     * 
     * @param size Size of the memory allocation
     * @param alignment Byte alignment for all pointer pushed on the stack
     */
    MemoryStack(size_t size)
        : m_Allocator(AllocatorType::make_unique(size)), m_Alignment(m_Allocator->DefaultAlignment()),
          m_CurrentSize(0), m_CurrentPointer(m_Allocator->Data()) {}

  public:
    virtual ~MemoryStack() override {}

    using shared_ptr = std::shared_ptr<MemoryStack<AllocatorType>>;

    /**
     * @brief Factory function to create a std::shared_ptr<MemoryStack<AllocatorType>>
     * 
     * @param size 
     * @return shared_ptr 
     */
    static shared_ptr make_shared(size_t size)
    {
        return shared_ptr(new MemoryStack<AllocatorType>(size));
    }

    /**
     * @brief Advances the stack pointer
     * 
     * Allocate advances the stack pointer by `size` plus some remainder to ensure subsequent
     * calles return aligned pointers
     * 
     * @param size Number of bytes to reserve on the stack
     * @return void* Starting address of the stack reservation
     */
    void *Allocate(size_t size) final override;

    /**
     * @brief Reset the memory stack
     * 
     * This operation resets the stack pointer to the base pointer of the memory allocation.
     */
    void Reset(bool writeZeros = false);

    /**
     * @brief Get Size of the Memory Stack
     */
    size_t Size() { return m_Allocator->Size(); }

    /**
     * @brief Get number of bytes currently allocated
     * @return size_t 
     */
    size_t Allocated() { return m_CurrentSize; }

    /**
     * @brief Get the number of free bytes that are not allocated
     * @return size_t 
     */
    size_t Available() { return Size() - Allocated(); }

    /**
     * @brief Byte alignment for stack allocations
     */
    size_t Alignment() { return m_Alignment; }

  private:
    std::unique_ptr<AllocatorType> m_Allocator;
    size_t m_Alignment;
    size_t m_CurrentSize;
    void *m_CurrentPointer;
};

// TODO: add a MemoryStack that allows internal allocations to grow.

class MemoryStackTracker : public IMemoryStack
{
  public:
    explicit MemoryStackTracker(std::shared_ptr<IMemoryStack> stack);
    ~MemoryStackTracker() override {}

    /**
     * @brief Advances the stack pointer
     * 
     * Allocate advances the stack pointer by `size` plus some remainder to ensure subsequent
     * calles return aligned pointers
     * 
     * @param size Number of bytes to reserve on the stack
     * @return void* Starting address of the stack reservation
     */
    void *Allocate(size_t size) final override;

    /**
     * @brief Get the Pointer to stack resversion `id`
     * 
     * @param id 0-indexed ID of stack reservation
     * @return void* Pointer the id-th stack reservation
     */
    void *GetPointer(uint32_t id) const;

    /**
     * @brief Get the Size of stack reservation `id`
     * 
     * @param id 0-indexed ID of stack reservation
     * @return size_t Size of the id-th reservation
     */
    size_t GetSize(uint32_t id) const;

    /**
     * @brief Get the number of reservations pushed to the stack.
     * 
     * @return size_t 
     */
    size_t Count() const;

    /**
     * @brief Get a list of all stack pointers
     * 
     * @return void** list of void* stack pointers
     */
    void **GetPointers() const;

  private:
    std::shared_ptr<IMemoryStack> m_Stack;
    std::vector<size_t> m_StackSize;
    std::vector<void *> m_StackPointers;
};

// Template Implementations

template <class AllocatorType>
void *MemoryStack<AllocatorType>::Allocate(size_t size)
{
    CHECK_LE(m_CurrentSize + size, m_Allocator->Size())
        << "Allocation too large.  Memory Total: " << m_Allocator->Size() / (1024 * 1024) << "MB. "
        << "Used: " << m_CurrentSize / (1024 * 1024) << "MB. "
        << "Requested: " << size / (1024 * 1024) << "MB.";

    void *return_ptr = m_CurrentPointer;
    size_t remainder = size % m_Alignment;
    size = (remainder == 0) ? size : size + m_Alignment - remainder;
    m_CurrentPointer = (unsigned char *)m_CurrentPointer + size;
    m_CurrentSize += size;
    return return_ptr;
}

template <class AllocatorType>
void MemoryStack<AllocatorType>::Reset(bool writeZeros)
{
    m_CurrentPointer = m_Allocator->Data();
    m_CurrentSize = 0;
    if (writeZeros)
    {
        m_Allocator->WriteZeros();
    }
}

} // end namespace yais

#endif // _YAIS_MEMORY_STACK_H_
