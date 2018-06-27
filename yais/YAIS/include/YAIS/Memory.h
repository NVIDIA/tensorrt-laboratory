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
#ifndef NVIS_MEMORY_H
#define NVIS_MEMORY_H
#pragma once

#include <memory>
#include <cstring>

#include <cuda.h>
#include <cuda_runtime.h>
#include <glog/logging.h>

namespace yais
{

/**
 * @brief Abstract base Memory class
 * 
 * Abstract base class that tracks the base pointer and size of a memory allocation.
 * Derived classes are expected to allocate memory and initialize the base pointer
 * in their protected/private constructor.  Derived classes are also expected to 
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
class Memory
{
  public:
    /**
     * @brief Get the Address to the Memory Allocation
     * @return void* 
     */
    inline void *Data() { return m_BasePointer; }

    /**
     * @brief Get the Size of the Memory Allocation
     * @return size_t 
     */
    size_t Size() { return m_BytesAllocated; }

    /**
     * @brief Write Zeros to Memory Allocation
     * 
     * Pure virtual function that is overrided to provide the details on how to write
     * zeros to all elements in the memory segment.
     */
    virtual void WriteZeros() = 0;

    /**
     * @brief Get the default alignment that should be used with this type of memory allocation.
     * 
     * @return size_t 
     */
    static size_t DefaultAlignment() { return 64; }

  protected:
    /**
     * @brief Construct a new Memory object
     *
     * The constructor of derived classes are responsible for performing the memory allocation and
     * setting the value of the base pointer.  All memory classes are expected to have protected or
     * private constructors.
     * 
     * @see m_BasePointer
     *  
     * @param size Size of memory allocation to be performed
     */
    Memory(size_t size) : m_BytesAllocated(size), m_BasePointer(nullptr) {}

    /**
     * @brief Size of Memory allocation in bytes
     */
    size_t m_BytesAllocated;

    /**
     * @brief Pointer to starting address of the memory allocation
     */
    void *m_BasePointer;
};

/**
 * @brief GPU memory management and properties
 * 
 * Derived Memory class for GPU memory management using cudaMalloc and cudaFree.
 */
class CudaDeviceMemory : public Memory
{
  protected:
    CudaDeviceMemory(size_t size) : Memory(size)
    {
        CHECK_EQ(cudaMalloc((void **)&m_BasePointer, size), CUDA_SUCCESS) << "cudaMalloc " << size << " bytes failed";
        DLOG(INFO) << "Allocated Cuda Device Memory (" << m_BasePointer << ", " << m_BytesAllocated << ")";
    }

    ~CudaDeviceMemory()
    {
        CHECK_EQ(cudaFree(m_BasePointer), CUDA_SUCCESS) << "cudaFree(" << m_BasePointer << ") failed";
        DLOG(INFO) << "Deleted Cuda Device Memory (" << m_BasePointer << ", " << m_BytesAllocated << ")";
    }

  public:
    static size_t DefaultAlignment() { return 256; }

    void WriteZeros() final override
    {
        CHECK_EQ(cudaMemset(Data(), 0, Size()), CUDA_SUCCESS) << "WriteZeros failed on Device Allocation";
    }
};

/**
 * @brief Allocates Page Locked (Pinned) Memory on the Host
 * 
 * Allocated page locked host memory using cudaMallocHost.  Pinned memory can provided better performance, but should
 * be used sparingly for staging areas for H2D and D2H transfers.
 */
class CudaHostMemory : public Memory
{
  protected:
    CudaHostMemory(size_t size) : Memory(size)
    {
        CHECK_EQ(cudaMallocHost((void **)&m_BasePointer, size), CUDA_SUCCESS) << "cudaMalloc " << size << " bytes failed";
        DLOG(INFO) << "Allocated Cuda Host Memory (" << m_BasePointer << ", " << m_BytesAllocated << ")";
    }

    ~CudaHostMemory()
    {
        CHECK_EQ(cudaFreeHost(m_BasePointer), CUDA_SUCCESS) << "cudaFree(" << m_BasePointer << ") failed";
        DLOG(INFO) << "Deleted Cuda Host Memory (" << m_BasePointer << ", " << m_BytesAllocated << ")";
    }

  public:
    void WriteZeros() final override
    {
        std::memset(Data(), 0, Size());
    }
};

/**
 * @brief Allocates CUDA Managed Memory
 * 
 * Allocates memory that will be automatically managed by the Unified Memory system.
 */
class CudaManagedMemory : public Memory
{
  protected:
    CudaManagedMemory(size_t size) : Memory(size)
    {
        CHECK_EQ(cudaMallocManaged((void **)&m_BasePointer, size, cudaMemAttachGlobal), CUDA_SUCCESS) << "cudaMallocManaged " << size << " bytes failed";
        DLOG(INFO) << "Allocated Cuda Managed Memory (" << m_BasePointer << ", " << m_BytesAllocated << ")";
    }

    ~CudaManagedMemory()
    {
        CHECK_EQ(cudaFree(m_BasePointer), CUDA_SUCCESS) << "cudaFree(" << m_BasePointer << ") failed";
        DLOG(INFO) << "Deleted Cuda Manged Memory (" << m_BasePointer << ", " << m_BytesAllocated << ")";
    }

  public:
    static size_t DefaultAlignment() { return 256; }

    void WriteZeros() final override
    {
        CHECK_EQ(cudaMemset(Data(), 0, Size()), CUDA_SUCCESS) << "WriteZeros failed on Device Allocation";
    }
};

/**
 * @brief Allocates memory using malloc
 */
class MallocMemory : public Memory
{
  protected:
    MallocMemory(size_t size) : Memory(size)
    {
        m_BasePointer = malloc(size);
        CHECK(m_BasePointer) << "malloc(" << size << ") failed";
        DLOG(INFO) << "malloc(" << m_BytesAllocated << ") returned " << m_BasePointer;
    }

    ~MallocMemory()
    {
        free(m_BasePointer);
        DLOG(INFO) << "Deleted Malloc'ed Memory (" << m_BasePointer << ", " << m_BytesAllocated << ")";
    }

  public:
    void WriteZeros() final override
    {
        std::memset(Data(), 0, Size());
    }
};

/**
 * @brief Allocator Factory
 * 
 * This factory class provides the factory functions to create shared_ptr and unique_ptr to an
 * Allocator of a given MemoryType.  
 * 
 * @tparam MemoryType MemoryType should be derived from the base Memory class.
 */
template <class MemoryType>
class Allocator : public MemoryType
{
  protected:
    // Inherit Constructors from Base Class
    using MemoryType::MemoryType;

  public:
    /**
     * @brief Unique Pointer to an Allocator of MemoryType
     * 
     * Unique pointers must specify their Deleter as part of the type definition.
     */
    // using unique_ptr = std::unique_ptr<Allocator, decltype(&Allocator::Deleter)>;
    using unique_ptr = std::unique_ptr<Allocator>;

    /**
     * @brief Create a shared_ptr to an Allocator of MemoryType
     * 
     * @param size Size in bytes to be allocated
     * @return std::shared_ptr<Allocator> 
     */
    static std::shared_ptr<Allocator> make_shared(size_t size)
    {
        return std::shared_ptr<Allocator>(new Allocator(size));
    }

    /**
     * @brief Create a unique_ptr to an Allocator of MemoryType
     * 
     * @param size Size in bytes to be allocated
     * @return unique_ptr 
     */
    static unique_ptr make_unique(size_t size)
    {
        return unique_ptr(new Allocator(size));
    }
};

/**
 * @brief CudaHostAllocator
 * Allocator using CudaHostMemory
 */
using CudaHostAllocator = Allocator<CudaHostMemory>;

/**
 * @brief CudaDeviceAllocator
 * Allocator using CudaDeviceMemory
 */
using CudaDeviceAllocator = Allocator<CudaDeviceMemory>;

/**
 * @brief CudaManagedAllocator
 * Allocator using CudaManagedMemory
 */
using CudaManagedAllocator = Allocator<CudaManagedMemory>;

/**
 * @brief MallocAllocator
 * Allocator using malloc
 */
using MallocAllocator = Allocator<MallocMemory>;

/**
 * @brief Interface Class for MemoryStack
 */
class IMemoryStack
{
    /**
     * @brief Advances the stack pointer
     * 
     * Allocate advances the stack pointer by `size` plus some remainder to ensure subsequent
     * calles return aligned pointers
     * 
     * @param size Number of bytes to reserve on the stack
     * @return void* Starting address of the stack reservation
     */
    virtual void *Allocate(size_t size) = 0;

    /**
     * @brief Reset the memory stack
     * 
     * This operation resets the stack pointer to the base pointer of the memory allocation.
     */
    virtual void ResetAllocations(bool writeZeros = false) = 0;
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
  public:
    /**
     * @brief Construct a new MemoryStack object
     * 
     * A stack using a single allocation of AllocatorType.  The stack can only be advanced or reset.
     * Popping is not supported.
     * 
     * @param size Size of the memory allocation
     * @param alignment Byte alignment for all pointer pushed on the stack
     */
    MemoryStack(size_t size, size_t alignment = AllocatorType::DefaultAlignment())
        : m_Allocator(AllocatorType::make_unique(size)), m_Alignment(alignment), m_CurrentSize(0),
          m_CurrentPointer(m_Allocator->Data()) {}

    /**
     * @brief Advances the stack pointer
     * 
     * Allocate advances the stack pointer by `size` plus some remainder to ensure subsequent
     * calles return aligned pointers
     * 
     * @param size Number of bytes to reserve on the stack
     * @return void* Starting address of the stack reservation
     */
    void *Allocate(size_t size) override;

    /**
     * @brief Reset the memory stack
     * 
     * This operation resets the stack pointer to the base pointer of the memory allocation.
     */
    void ResetAllocations(bool writeZeros = false) override;

    /**
     * @brief Get Size of the Memory Stack
     */
    size_t Size() { return m_Allocator->Size(); }

    /**
     * @brief 
     * 
     */
    size_t AllocatedBytes();
    size_t AvailableBytes();

  private:
    typename AllocatorType::unique_ptr m_Allocator;
    size_t m_Alignment;
    size_t m_CurrentSize;
    void *m_CurrentPointer;
};

// TODO: add a MemoryStack that allows internal allocations to grow.

class MemoryStackTracker : public IMemoryStack
{
  public:
    MemoryStackTracker(std::shared_ptr<IMemoryStack> stack);

    /**
     * @brief Advances the stack pointer
     * 
     * Allocate advances the stack pointer by `size` plus some remainder to ensure subsequent
     * calles return aligned pointers
     * 
     * @param size Number of bytes to reserve on the stack
     * @return void* Starting address of the stack reservation
     */
    void *Allocate(size_t size);

    /**
     * @brief Reset the memory stack
     * 
     * This operation resets the stack pointer to the base pointer of the memory allocation.
     * TODO: Optionally, provide a feature to zero the entire stack on reset.
     */
    void ResetAllocations(bool writeZeros = false);

    /**
     * @brief Get the Pointer to stack resversion `id`
     * 
     * @param id 0-indexed ID of stack reservation
     * @return void* Pointer the id-th stack reservation
     */
    void *GetAllocationPointer(uint32_t id);

    /**
     * @brief Get the Size of stack reservation `id`
     * 
     * @param id 0-indexed ID of stack reservation
     * @return size_t Size of the id-th reservation
     */
    size_t GetAllocationSize(uint32_t id);

    /**
     * @brief Get the number of reservations pushed to the stack.
     * 
     * @return size_t 
     */
    size_t GetAllocationCount();

    /**
     * @brief Get a list of all stack pointers
     * 
     * @return void** list of void* stack pointers
     */
    void **GetPointers();

  private:
    std::shared_ptr<IMemoryStack> m_Stack;
    std::vector<size_t> m_StackSize;
    std::vector<void *> m_StackPointers;
};

/**
 * @brief Extends the base MemoryStack to provide tracking.
 * 
 * Standard MemoryStack extended to track each allocation pushed to stack.  Details of each
 * allocaton can be looked up by integer index of the allocation.  This object is useful for
 * tracking TensorRT input/output bindings as they are pushed to the stack.  You probably 
 * want to avoid using this object if you plan to push a large number of Allocates to the
 * stack.
 * 
 * @tparam AllocatorType 
 */
template <class AllocatorType>
class MemoryStackWithTracking : public MemoryStack<AllocatorType>
{
  public:
    MemoryStackWithTracking(size_t size, size_t alignment = AllocatorType::DefaultAlignment())
        : MemoryStack<AllocatorType>(size, alignment) {}

    /**
     * @brief Advances the stack pointer
     * 
     * Allocate advances the stack pointer by `size` plus some remainder to ensure subsequent
     * calles return aligned pointers
     * 
     * @param size Number of bytes to reserve on the stack
     * @return void* Starting address of the stack reservation
     */
    void *Allocate(size_t size);

    /**
     * @brief Reset the memory stack
     * 
     * This operation resets the stack pointer to the base pointer of the memory allocation.
     * TODO: Optionally, provide a feature to zero the entire stack on reset.
     */
    void ResetAllocations(bool writeZeros = false);

    /**
     * @brief Get the Pointer to stack resversion `id`
     * 
     * @param id 0-indexed ID of stack reservation
     * @return void* Pointer the id-th stack reservation
     */
    void *GetPointer(uint32_t id);

    /**
     * @brief Get a list of all stack pointers
     * 
     * @return void** list of void* stack pointers
     */
    void **GetPointers();

    /**
     * @brief Get the Size of stack reservation `id`
     * 
     * @param id 0-indexed ID of stack reservation
     * @return size_t Size of the id-th reservation
     */
    size_t GetSize(uint32_t id);

    /**
     * @brief Get the number of reservations pushed to the stack.
     * 
     * @return size_t 
     */
    size_t GetCount();

  private:
    std::vector<size_t> m_StackSize;
    std::vector<void *> m_StackPointers;
};

//
// Template Implementations
//

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
void *MemoryStackWithTracking<AllocatorType>::Allocate(size_t size)
{
    m_StackSize.push_back(size);
    m_StackPointers.push_back(MemoryStack<AllocatorType>::Allocate(size));
    return m_StackPointers.back();
}

template <class AllocatorType>
void MemoryStack<AllocatorType>::ResetAllocations(bool writeZeros)
{
    m_CurrentPointer = m_Allocator->Data();
    m_CurrentSize = 0;
    if (writeZeros)
    {
        m_Allocator->WriteZeros();
    }
}

template <class AllocatorType>
void MemoryStackWithTracking<AllocatorType>::ResetAllocations(bool writeZeros)
{
    MemoryStack<AllocatorType>::ResetAllocations(writeZeros);
    m_StackSize.clear();
    m_StackPointers.clear();
}

template <class AllocatorType>
void *MemoryStackWithTracking<AllocatorType>::GetPointer(uint32_t id)
{
    CHECK_LT(id, m_StackPointers.size()) << "Invalid Stack Pointer ID";
    return m_StackPointers[id];
}

template <class AllocatorType>
size_t MemoryStackWithTracking<AllocatorType>::GetSize(uint32_t id)
{
    CHECK_LT(id, m_StackSize.size()) << "Invalid Stack Pointer ID";
    return m_StackSize[id];
}

template <class AllocatorType>
void **MemoryStackWithTracking<AllocatorType>::GetPointers()
{
    return (void **)m_StackPointers.data();
}

template <class AllocatorType>
size_t MemoryStackWithTracking<AllocatorType>::GetCount()
{
    return m_StackPointers.size();
}

} // end namespace yais

#endif // NVIS_GPU_H_
