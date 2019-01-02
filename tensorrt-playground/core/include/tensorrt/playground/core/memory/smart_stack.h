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
#include <memory>

#include <glog/logging.h>

#include "tensorrt/playground/core/memory/descriptor.h"
#include "tensorrt/playground/core/memory/memory.h"
#include "tensorrt/playground/core/memory/memory_stack.h"

namespace yais {
namespace Memory {

template<typename MemoryType>
class SmartStack : public MemoryStack<MemoryType>,
                   public std::enable_shared_from_this<SmartStack<MemoryType>>
{
  protected:
    using MemoryStack<MemoryType>::MemoryStack;
    using MemoryStack<MemoryType>::Allocate;

    class StackDescriptorImpl : public Descriptor<MemoryType>
    {
      public:
        StackDescriptorImpl(std::shared_ptr<const SmartStack<MemoryType>> stack, void* ptr,
                            size_t size)
            : Descriptor<MemoryType>(ptr, size, "SmartStack"), m_Stack(stack),
              m_Offset(Stack().Offset(this->Data()))
        {
        }

        StackDescriptorImpl(StackDescriptorImpl&& other)
            : MemoryType(std::move(other)), m_Offset{std::exchange(other.m_Offset, 0)},
              m_Stack{std::exchange(other.m_Stack, nullptr)}
        {
        }

        virtual ~StackDescriptorImpl() override {}

        size_t Offset() const
        {
            return m_Offset;
        }

        const MemoryStack<MemoryType>& Stack() const
        {
            return *m_Stack;
        }

      private:
        std::shared_ptr<const SmartStack<MemoryType>> m_Stack;
        size_t m_Offset;

        friend class SmartStack<MemoryType>;
    };

  public:
    using StackType = std::shared_ptr<SmartStack<MemoryType>>;
    using StackDescriptor = std::unique_ptr<StackDescriptorImpl>;

    static StackType Create(size_t size)
    {
        return StackType(new SmartStack(size));
    }
    static StackType Create(std::unique_ptr<MemoryType> memory)
    {
        return StackType(new SmartStack(memory));
    }

    StackDescriptor Allocate(size_t size)
    {
        CHECK_LE(size, this->Available());

        auto ptr = MemoryStack<MemoryType>::Allocate(size);
        auto stack = this->shared_from_this();

        // Special Descriptor derived from MemoryType that hold a reference to the MemoryStack,
        // and who's destructor does not try to free the MemoryType memory.
        auto ret = std::make_unique<StackDescriptorImpl>(stack, ptr, size);

        DLOG(INFO) << "Allocated " << ret->Size() << " starting at " << ret->Data()
                   << " on SmartStack " << stack.get();

        return std::move(ret);
    }
};

} // end namespace Memory
} // end namespace yais
