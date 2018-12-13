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

#include <glog/logging.h>
#include <memory>

#include "tensorrt/playground/core/utils.h"

namespace yais
{
/**
 * @brief Abstract base Memory class
 *
 * Abstract base class that tracks the base pointer and size of a memory allocation.
 * Derived classes are expected to allocate memory and initialize the base pointer
 * ink their protected/private constructor.  Derived classes are also expected to
 * implement a static Deleter method responsible for freeing the memory allocation.
 *
 * Derived classes are expected to use protected or private constructors.  Memory
 * objects will only be created by Allocator factory functions which return either a
 * `shared_ptr` or `unique_ptr` of an Allocator object of MemoryType.
 *
 * Recommended alignment values are not directly used by the Memory or the Allocator
 * object; however, subsequent specializations like MemoryStacks can use the DefaultAlignment
 * to specialize their function specific to the MemoryType used.
 */
struct IMemory
{
    virtual ~IMemory() = default;

    virtual const std::string& Type() const = 0;

    virtual void* Data() const = 0;
    virtual size_t Size() const = 0;

    virtual bool Allocated() const = 0;

    virtual void Fill(char) = 0;
    virtual size_t DefaultAlignment() const = 0;
};

struct IAllocatableMemory
{
    virtual void* Allocate(size_t) = 0;
    virtual void Free() = 0;
};

template<class MemoryType>
class BaseMemory : public IMemory
{
  protected:
    BaseMemory(void* ptr, size_t size, bool allocated); 
    BaseMemory(BaseMemory&& other) noexcept;
    BaseMemory& operator=(BaseMemory&&) noexcept = delete;
    DELETE_COPYABILITY(BaseMemory);

  public:
    virtual ~BaseMemory() {}

    using BaseType = MemoryType;

    inline void* Data() const final override;
    inline size_t Size() const final override;

    inline bool Allocated() const final override;

    void* operator[](size_t offset) const;

    template <typename T>
    T* cast_to_array() { return static_cast<T*>(Data()); }
    // auto array_cast() { return dynamic_cast<T[Size()/sizeof(T)]>(Data()); }

  private:
    void* m_MemoryAddress;
    size_t m_BytesAllocated;
    bool m_Allocated;
};

class HostMemory : public BaseMemory<HostMemory>
{
  public:
    using BaseMemory<HostMemory>::BaseMemory;
    const std::string& Type() const override;

    void Fill(char) override;
    size_t DefaultAlignment() const override;
};

class SystemMallocMemory : public HostMemory, public IAllocatableMemory
{
  public:
    using HostMemory::HostMemory;
    const std::string& Type() const final override;

  protected:
    void* Allocate(size_t) final override;
    void Free() final override;
};

class SystemV : public HostMemory, public IAllocatableMemory
{
  protected:
    SystemV(void* ptr, size_t size, bool allocated);

  public:
    SystemV(int shm_id);
    SystemV(SystemV&& other) noexcept;

  public:
    virtual ~SystemV() override;

    const std::string& Type() const final override;

    int ShmID() const;
    void DisableAttachment();

  protected:
    void* Allocate(size_t) final override;
    void Free() final override;

  private:
    void* Attach(int shm_id);
    static size_t SegSize(int shm_id);

    int m_ShmID;
};

} // end namespace yais

#include "tensorrt/playground/core/impl/memory.h"
