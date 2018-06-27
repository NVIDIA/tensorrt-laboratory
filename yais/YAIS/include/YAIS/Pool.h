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
#ifndef NVIS_RESOURCEPOOL_H_
#define NVIS_RESOURCEPOOL_H_
#pragma once

#include <memory>
#include <condition_variable>
#include <mutex>
#include <queue>

namespace yais 
{

/**
 * @brief Templated Thread-safe Queue
 * 
 * A simple thread-safe queue using a mutex and a condition variable.  This
 * class is derived from `std::enabled_shared_from_this` which requires it 
 * to be create using `std::make_shared`.
 * 
 * @tparam T 
 */
template <typename T>
class Queue : public std::enable_shared_from_this<Queue<T>>
{
  protected:
    Queue() = default;

  public:
    /**
     * @brief Factory function to properly create a Queue.
     * 
     * @return std::shared_ptr<Queue<T>> 
     */
    static std::shared_ptr<Queue<T>> Create() {
        return std::shared_ptr<Queue<T>>(new Queue<T>());
    }

    Queue(Queue &&other)
    {
        std::unique_lock<std::mutex> lock(other.mutex_);
        queue_ = std::move(other.queue_);
    }

    /**
     * @brief Push a new value of T to the Queue
     * 
     * @param value 
     */
    void Push(T value)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        queue_.push(std::move(value));
        lock.unlock();
        cond_.notify_one();
    }

    /**
     * @brief Pop the Front object from the Queue and return.
     * 
     * @return T 
     */
    T Pop()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_.wait(lock, [this] { return !queue_.empty(); });
        T value = std::move(queue_.front());
        queue_.pop();
        return value;
    }

    /**
     * @brief Numbe of items in the Queue
     * 
     * @return std::size_t 
     */
    std::size_t Size()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.size();
    }

  private:
    mutable std::mutex mutex_;
    std::queue<T> queue_;
    std::condition_variable cond_;
};


/**
 * @brief Pool of ResourceType
 * 
 * Pool of ResourceTypes implemented as a Queue.
 * 
 * A unique aspect of this Pool object is the return type of the Pop method.  While the
 * Pool consists of shared_ptr's to objects of ResourceType typically created with the
 * the std::default_deleter.  The Pop method of this Pool clsss returned a different 
 * type of shared_ptr<ResourceType> than the object pushed to the Pool.  The difference
 * is that the returned shared_ptr uses a custom deleter thereby creating a new logic
 * block for tracking this shared_ptr.  Rather than freeing the object in question, 
 * the custom deleter captures both the original shared_ptr from the pool and the
 * shared_ptr of the Pool (shared_from_this), and uses those captured values to return
 * the original shared_ptr (with the default_deleter) to the Pool.
 * 
 * By holding a reference to the pool in the custom deleter of the returned shared_ptr,
 * we ensure that the pool can not be deallocated while resources have been checked out.
 * 
 * The custom shared_ptr also helps ensure resources are returned to the pool even if the
 * thread using the resources throws an exception.
 * 
 * @tparam T 
 */
template <typename ResourceType>
class Pool : public Queue<std::shared_ptr<ResourceType>>
{
  protected:
    Pool() = default;
    
  public:
    /**
     * @brief Factory function for creating a Pool.
     * 
     * @return std::shared_ptr<Pool<ResourceType>> 
     */
    static std::shared_ptr<Pool<ResourceType>> Create() {
        return std::shared_ptr<Pool<ResourceType>>(new Pool<ResourceType>());
    }

    /**
     * @brief Acquire a shared pointer to a ResourceType held by the Pool.
     *
     * Returns a shared_ptr<ResourceType> with a custom deleter that return the 
     * Resource object by to the pool when the reference count of the shared_ptr
     * goes to zero.
     *  
     * @return std::shared_ptr<ResourceType> 
     */
    std::shared_ptr<ResourceType> Pop()
    {
        auto pool_ptr = this->shared_from_this();
        auto from_pool = Queue<std::shared_ptr<ResourceType>>::Pop();
        std::shared_ptr<ResourceType> ptr(from_pool.get(), [from_pool, pool_ptr](auto p) {
            pool_ptr->Push(from_pool);
        });
        return ptr;
    }

    /**
     * @brief Instantiates and Pushes a new Resource object.
     * 
     * @param newObj Raw pointer to an object of ResourceType
     */
    void EmplacePush(ResourceType* newObj)
    {
        this->Push(std::shared_ptr<ResourceType>(newObj));
    }

    /**
     * @brief Instantiates and Pushes a new Resource object.
     * 
     * Forwards the arguments passed to this method to the ResourceType constructor.
     * 
     * @tparam Args 
     * @param args
     */
    template <typename ... Args>
    void EmplacePush(Args&& ... args)
    {
        EmplacePush(new ResourceType(std::forward<Args>(args)...));
    }
};


#if 0
// A pool of available resources is simply implemented as a thread-safe queue
template <typename T>
using UniquePool = Queue<std::unique_ptr<T>>;


// A RAII class for acquiring a scoped resource from a pool. 
template <typename T>
class ScopedPointer
{
  public:
    explicit ScopedPointer(std::shared_ptr<UniquePool<T>> pool)
        : m_Pool(pool), m_Pointer(m_Pool->Pop()) {}

    ScopedPointer(ScopedPointer &&other)
        : m_Pool(other.m_Pool)
    {
        m_Pool = other.m_Pool;
        m_Pointer = std::move(other.m_Pointer);
    }

    ~ScopedPointer()
    {
        Clear();
    }

    void Clear()
    {
        if (m_Pointer) {
            m_Pool->Push(std::move(m_Pointer));
        }
    }

    T *operator->() const
    {
        return m_Pointer.get();
    }

  private:
    std::shared_ptr<UniquePool<T>> m_Pool;
    std::unique_ptr<T> m_Pointer;
};
#endif

}

#endif // NVIS_RESOURCEPOOL_H_