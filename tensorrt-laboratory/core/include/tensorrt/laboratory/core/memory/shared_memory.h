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

#include <glog/logging.h>
#include <memory>

#include "tensorrt/laboratory/core/utils.h"

namespace trtlab {

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

    // Implemented by CoreMemory
    virtual void* Data() const = 0;
    virtual size_t Size() const = 0;
    virtual bool Allocated() const = 0;

    // Implemented by a BaseMemory derivative, e.g. HostMemory
    virtual void Fill(char) = 0;
    virtual size_t DefaultAlignment() const = 0;

    // Implemented by every derivative class
    virtual const std::string& Type() const = 0;
};

struct IAllocatable
{
    virtual void* Allocate(size_t) = 0;
    virtual void Free() = 0;
};

class CoreMemory : public virtual IMemory
{
  protected:
    CoreMemory(void* ptr, size_t size, bool allocated);
    CoreMemory(CoreMemory&& other) noexcept;
    CoreMemory& operator=(CoreMemory&&) noexcept;

    DELETE_COPYABILITY(CoreMemory);

  public:
    virtual ~CoreMemory() override;

    inline void* Data() const final override
    {
        return m_MemoryAddress;
    }
    inline size_t Size() const final override
    {
        return m_BytesAllocated;
    }
    inline bool Allocated() const final override
    {
        return m_Allocated;
    }

    void* operator[](size_t offset) const
    {
        CHECK_LE(offset, Size());
        return static_cast<void*>(static_cast<char*>(Data()) + offset);
    }

    template<typename T>
    T* CastToArray()
    {
        return static_cast<T*>(Data());
    }

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
};

template<class MemoryType>
class Allocator final : public MemoryType
{
  public:
    Allocator(size_t size);
    Allocator(Allocator&& other) noexcept;
    Allocator& operator=(Allocator&& other) noexcept;
    ~Allocator() override;

    DELETE_COPYABILITY(Allocator);
};

template<typename MemoryType>
class Descriptor : public MemoryType
{
  protected:
    Descriptor(MemoryType&&);
    Descriptor(void* ptr, size_t size);
    Descriptor(Descriptor&& other) noexcept;
    Descriptor& operator=(Descriptor&& other) noexcept;

    DELETE_COPYABILITY(Descriptor);

  public:
    virtual ~Descriptor() override;
};

template<typename MemoryType>
using MemoryDescriptor = std::unique_ptr<Descriptor<MemoryType>>;

class HostMemory : public BaseMemory<HostMemory>
{
  public:
    using BaseMemory<HostMemory>::BaseMemory;
    const std::string& Type() const override;

    void Fill(char) override;
    size_t DefaultAlignment() const override;
};

class Malloc : public HostMemory, public IAllocatable
{
  public:
    using HostMemory::HostMemory;
    const std::string& Type() const final override;

  protected:
    void* Allocate(size_t) final override;
    void Free() final override;
};

class SystemV : public HostMemory, public IAllocatable
{
  protected:
    SystemV(int shm_id);
    SystemV(void* ptr, size_t size, bool allocated);
    SystemV(SystemV&& other) noexcept;
    SystemV& operator=(SystemV&& other) noexcept;

    DELETE_COPYABILITY(SystemV);

  public:
    virtual ~SystemV() override;
    const std::string& Type() const final override;

    static MemoryDescriptor<SystemV> Attach(int shm_id);

    int ShmID() const;
    void DisableAttachment();

  protected:
    void* Allocate(size_t) final override;
    void Free() final override;

  private:
    int m_ShmID;
};


} // namespace trtlab

#include "tensorrt/laboratory/core/impl/memory.h"
