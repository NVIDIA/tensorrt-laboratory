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
#ifndef _YAIS_MEMORY_H_
#define _YAIS_MEMORY_H_

#include <memory>

namespace yais
{

std::string BytesToString(size_t bytes);

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
class IMemory
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
  virtual size_t DefaultAlignment() = 0;

  /**
     * @brief Destroy the Memory object
     */
  virtual ~IMemory() {}

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
  IMemory(size_t size) : m_BytesAllocated(size), m_BasePointer(nullptr) {}

  void SetBasePointer(void *ptr) { m_BasePointer = ptr; }

  /**
     * @brief Size of Memory allocation in bytes
     */
  size_t m_BytesAllocated;

  /**
     * @brief Pointer to starting address of the memory allocation
     */
  void *m_BasePointer;
};

class HostMemory : public IMemory
{
public:
  using IMemory::IMemory;
  void WriteZeros() override;
  size_t DefaultAlignment() override;
};

class DeviceMemory : public IMemory
{
public:
  using IMemory::IMemory;
  void WriteZeros() override;
  size_t DefaultAlignment() override;
};

/**
 * @brief Allocates CUDA Managed Memory
 * 
 * Allocates memory that will be automatically managed by the Unified Memory system.
 */
class CudaManagedMemory : public DeviceMemory
{
protected:
  CudaManagedMemory(size_t size);
  virtual ~CudaManagedMemory() override;
};

/**
 * @brief Allocates CUDA Device Memory
 * 
 * Derived Memory class for GPU memory management using cudaMalloc and cudaFree.
 */
class CudaDeviceMemory : public DeviceMemory
{
protected:
  CudaDeviceMemory(size_t size);
  virtual ~CudaDeviceMemory() override;
};

/**
 * @brief Allocates Page Locked (Pinned) Memory on the Host
 * 
 * Allocated page locked host memory using cudaMallocHost.  Pinned memory can provided better performance, but should
 * be used sparingly for staging areas for H2D and D2H transfers.
 */
class CudaHostMemory : public HostMemory
{
protected:
  CudaHostMemory(size_t size);
  virtual ~CudaHostMemory() override;
};

/**
 * @brief Allocates Memory using the System's malloc
 */
class SystemMallocMemory : public HostMemory
{
protected:
  SystemMallocMemory(size_t size);
  virtual ~SystemMallocMemory() override;
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
class Allocator final : public MemoryType
{
protected:
  // Inherit Constructors from Base Class
  using MemoryType::MemoryType;

public:
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
  static std::unique_ptr<Allocator> make_unique(size_t size)
  {
    return std::unique_ptr<Allocator>(new Allocator(size));
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
using SystemMallocAllocator = Allocator<SystemMallocMemory>;

} // end namespace yais

#endif // _YAIS_MEMORY_H_
