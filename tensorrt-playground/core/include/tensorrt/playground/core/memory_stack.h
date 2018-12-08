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
#pragma once
#include <vector>

#include <glog/logging.h>

#include "tensorrt/playground/core/allocator.h"
#include "tensorrt/playground/core/memory.h"

namespace yais
{
/**
 * @brief General MemoryStack
 *
 * Memory stack using a single memory allocation of MemoryType.  The stack pointer is advanced
 * by the Allocate function.  Each new stack pointer is aligned to either the DefaultAlignment of
 * the MemoryType or customized when the MemoryStack is instantiated.
 *
 * @tparam MemoryType
 */
template<class MemoryType>
class MemoryStack
{
  public:
    /**
     * @brief Construct a new MemoryStack object
     *
     * A stack using a single allocation of MemoryType.  The stack can only be advanced or reset.
     * Popping is not supported.
     *
     * @param size Size of the memory allocation
     * @param alignment Byte alignment for all pointer pushed on the stack
     */

    MemoryStack(std::unique_ptr<MemoryType> memory)
        : m_Memory(std::move(memory)), m_CurrentPointer(m_Memory->Data()), m_CurrentSize(0),
          m_Alignment(m_Memory->DefaultAlignment())
    {
        CHECK(m_Memory);
    }

    MemoryStack(size_t size)
      : MemoryStack(std::move(Allocator<MemoryType>::make_unique(size)))
    {
    }

    virtual ~MemoryStack() {}

    using BaseType = typename MemoryType::BaseType;

    /**
     * @brief Advances the stack pointer
     *
     * Allocate advances the stack pointer by `size` plus some remainder to ensure subsequent
     * calles return aligned pointers
     *
     * @param size Number of bytes to reserve on the stack
     * @return void* Starting address of the stack reservation
     */
    void* Allocate(size_t size);

    /**
     * @brief Reset the memory stack
     *
     * This operation resets the stack pointer to the base pointer of the memory allocation.
     */
    void Reset(bool writeZeros = false);

    /**
     * @brief Get Size of the Memory Stack
     */
    size_t Size() const
    {
        return m_Memory->Size();
    }

    /**
     * @brief Get number of bytes currently allocated
     * @return size_t
     */
    size_t Allocated() const
    {
        return m_CurrentSize;
    }

    /**
     * @brief Get the number of free bytes that are not allocated
     * @return size_t
     */
    size_t Available() const
    {
        return Size() - Allocated();
    }

    /**
     * @brief Byte alignment for stack allocations
     */
    size_t Alignment() const
    {
        return m_Alignment;
    }

  private:
    std::unique_ptr<MemoryType> m_Memory;
    void* m_CurrentPointer;
    size_t m_CurrentSize;
    size_t m_Alignment;
};

template<typename MemoryType>
class MemoryDescriptorStack : public MemoryStack<MemoryType>,
                              public std::enable_shared_from_this<MemoryDescriptorStack<MemoryType>>
{
  protected:
    using MemoryStack<MemoryType>::Allocate;

  public:
    using MemoryStack<MemoryType>::MemoryStack;
    using Handle = std::shared_ptr<MemoryDescriptorStack<MemoryType>>;
    using BaseType = typename MemoryType::BaseType;
    using Descriptor = std::shared_ptr<BaseType>;

    static Handle Create(size_t size)
    {
        return std::make_shared<MemoryDescriptorStack>(size);
    }
    static Handle Create(std::unique_ptr<MemoryType> memory)
    {
        return std::make_shared<MemoryDescriptorStack>(memory);
    }

    Descriptor Allocate(size_t size)
    {
        CHECK_LE(size, this->Available());

        auto ptr = MemoryStack<MemoryType>::Allocate(size);
        auto segment = this->shared_from_this();

        // Special smart pointer that hold a reference to the Segment
        // and who's destructor does not try to free any memory,
        // instead, it frees only the wrapper object
        auto ret = Allocator<BaseType>::UnsafeWrapRawPointer(ptr, size, [segment](BaseType* p) {});

        DLOG(INFO) << "Allocated " << ret->Size() << " starting at " << ret->Data()
                   << " on segment " << segment.get();

        return ret;
    }
};

// Template Implementations

template<class MemoryType>
void* MemoryStack<MemoryType>::Allocate(size_t size)
{
    DLOG(INFO) << "Allocate pushes MemoryStack Pointer by : " << size;

    CHECK_LE(m_CurrentSize + size, m_Memory->Size())
        << "Allocation too large.  Memory Total: " << m_Memory->Size() / (1024 * 1024) << "MB. "
        << "Used: " << m_CurrentSize / (1024 * 1024) << "MB. "
        << "Requested: " << size / (1024 * 1024) << "MB.";

    void* return_ptr = m_CurrentPointer;
    size_t remainder = size % m_Alignment;
    size = (remainder == 0) ? size : size + m_Alignment - remainder;
    m_CurrentPointer = static_cast<unsigned char*>(m_CurrentPointer) + size;
    m_CurrentSize += size;
    return return_ptr;
}

template<class MemoryType>
void MemoryStack<MemoryType>::Reset(bool writeZeros)
{
    m_CurrentPointer = m_Memory->Data();
    m_CurrentSize = 0;
    if(writeZeros)
    {
        m_Memory->Fill(0);
    }
}

} // end namespace yais
