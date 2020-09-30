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

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>

#include "standard_threads.h"
#include "utils.h"

#include "glog/logging.h"

namespace trtlab
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
        static std::shared_ptr<Queue<T>> Create()
        {
            return std::shared_ptr<Queue<T>>(new Queue<T>());
        }

        Queue(Queue&& other)
        {
            std::lock_guard<std::mutex> lock(other.mutex_);
            queue_ = std::move(other.queue_);
        }

        virtual ~Queue() {}

        /**
     * @brief Push a new value of T to the Queue
     *
     * @param value
     */
        void Push(T value)
        {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                queue_.push(std::move(value));
            }
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
            std::lock_guard<std::mutex> lock(mutex_);
            return queue_.size();
        }

    private:
        mutable std::mutex      mutex_;
        std::queue<T>           queue_;
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

    namespace v1
    {
        template <typename ResourceType>
        class Pool : public Queue<std::shared_ptr<ResourceType>>
        {
        protected:
            using Queue<std::shared_ptr<ResourceType>>::Queue;

        public:
            /**
     * @brief Factory function for creating a Pool.
     *
     * @return std::shared_ptr<Pool<ResourceType>>
     */

            static std::shared_ptr<Pool<ResourceType>> Create()
            {
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
                return Pop([](ResourceType&) {});
            }

            /**
     * @brief Acquire a shared pointer to a ResourceType held by the Pool.
     *
     * Returns a shared_ptr<ResourceType> with a custom deleter that return the
     * Resource object by to the pool when the reference count of the shared_ptr
     * goes to zero.
     *
     * onReturn will be executed prior to the object being returned to the pool.
     * onReturn is passed the raw pointer to the ResourceType.
     *
     * @param onReturn
     * @return std::shared_ptr<ResourceType>
     */
            std::shared_ptr<ResourceType> Pop(std::function<void(ResourceType&)> onReturn)
            {
                // auto pool_ptr = this->shared_from_this();
                auto                          from_pool = Queue<std::shared_ptr<ResourceType>>::Pop();
                auto                          raw       = from_pool.get();
                std::shared_ptr<ResourceType> ptr(raw, [from_pool = std::move(from_pool), pool_ptr = std::move(this->shared_from_this()),
                                                        onReturn](auto p) mutable {
                    onReturn(*p);
                    pool_ptr->Push(std::move(from_pool));
                    pool_ptr.reset();
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
            template <typename... Args>
            void EmplacePush(Args&&... args)
            {
                EmplacePush(new ResourceType(std::forward<Args>(args)...));
            }

            /**
     * @brief Pop/Dequeue a shared pointer to a ResourceType object.
     *
     * Unlike the Pop() that provides a shared_ptr whose Deleter returns the object
     * to the Pool; this method permanently removes the object from the Pool.
     *
     * @return std::shared_ptr<ResourceType>
     * @see Pop()
     */
            std::shared_ptr<ResourceType> PopWithoutReturn()
            {
                return Queue<std::shared_ptr<ResourceType>>::Pop();
            }
        };

    } // namespace v1

    namespace v2
    {
        template <typename ResourceType, typename ThreadType = standard_threads>
        class Pool final : public std::enable_shared_from_this<Pool<ResourceType, ThreadType>>
        {
            struct internal_key
            {
            };
            using mutex_t = typename ThreadType::mutex;
            using cv_t    = typename ThreadType::cv;
            mutable mutex_t          m_Mutex;
            cv_t                     m_Condition;
            std::queue<ResourceType> m_Queue;

        public:
            using PoolItem   = std::shared_ptr<ResourceType>;
            using PoolType   = std::shared_ptr<Pool<ResourceType, ThreadType>>;
            using OnReturnFn = std::function<void(ResourceType&)>;

            Pool(internal_key) {}
            ~Pool() {}

            static PoolType Create()
            {
                return std::make_shared<Pool<ResourceType, ThreadType>>(internal_key());
            }

            void Push(ResourceType&& obj)
            {
                {
                    std::lock_guard<mutex_t> lock(m_Mutex);
                    m_Queue.push(std::move(obj));
                }
                m_Condition.notify_one();
            }

            template <typename... Args>
            void EmplacePush(Args&&... args)
            {
                ResourceType obj(std::forward<Args>(args)...);
                Push(std::move(obj));
            }

            PoolItem Pop(OnReturnFn onReturn)
            {
                std::unique_lock<mutex_t> lock(m_Mutex);
                m_Condition.wait(lock, [this] { return !m_Queue.empty(); });
                std::shared_ptr<ResourceType> ptr(new ResourceType(std::move(m_Queue.front())),
                                                  [pool = this->shared_from_this(), onReturn](ResourceType* p) {
                                                      onReturn(*p);
                                                      pool->Push(std::move(*p));
                                                      delete p;
                                                  });
                m_Queue.pop();
                return ptr;
            }

            PoolItem Pop()
            {
                return Pop([](ResourceType&) {});
            }

            PoolItem PopWithoutReturn()
            {
                std::unique_lock<mutex_t> lock(m_Mutex);
                m_Condition.wait(lock, [this] { return !m_Queue.empty(); });
                auto ptr = std::make_shared<ResourceType>(std::move(m_Queue.front()));
                m_Queue.pop();
                return ptr;
            }

            std::size_t Size() const
            {
                std::lock_guard<mutex_t> lock(m_Mutex);
                return m_Queue.size();
            }
        };

    } // namespace v2

    namespace v3
    {
        template <typename ResourceType, typename ThreadType = standard_threads>
        class Pool final : public std::enable_shared_from_this<Pool<ResourceType, ThreadType>>
        {
            struct key
            {
            };
            class Resource;

            using pool_t = std::shared_ptr<Pool<ResourceType, ThreadType>>;
            using item_t = std::unique_ptr<ResourceType>;

        public:
            using resource_type = ResourceType;
            using on_return_fn  = std::function<void(ResourceType&)>;

            Pool(key) {}
            virtual ~Pool() {}

            DELETE_COPYABILITY(Pool);
            DELETE_MOVEABILITY(Pool);

            static pool_t Create()
            {
                return std::make_shared<Pool<ResourceType, ThreadType>>(key());
            }

            void push(ResourceType&& resource)
            {
                internal_push(std::make_unique<ResourceType>(std::move(resource)));
            }

            void push(std::unique_ptr<ResourceType> resource)
            {
                internal_push(std::move(resource));
            }

            template <typename... Args>
            void emplace_push(Args&&... args)
            {
                push(std::make_unique<ResourceType>(std::forward<Args>(args)...));
            }

            Resource pop(on_return_fn on_return)
            {
                return Resource(internal_pop(), this->shared_from_this(), on_return);
            }

            Resource pop()
            {
                return pop([](ResourceType&) {});
            }

            Resource pop_without_return()
            {
                return Resource(internal_pop(), nullptr, [](ResourceType&) {});
            }

        private:
            void internal_push(item_t ptr)
            {
                {
                    std::lock_guard<mutex_t> lock(m_Mutex);
                    m_Queue.push(std::move(ptr));
                }
                m_Condition.notify_one();
            }

            item_t internal_pop()
            {
                std::unique_lock<mutex_t> lock(m_Mutex);
                m_Condition.wait(lock, [this] { return !m_Queue.empty(); });
                auto ptr = std::move(m_Queue.front());
                m_Queue.pop();
                return ptr;
            }

            class Resource
            {
            public:
                Resource(item_t resource, pool_t pool, on_return_fn on_return)
                : m_Resource(std::move(resource)), m_Pool(pool), m_OnReturnFn(on_return)
                {
                }

                virtual ~Resource()
                {
                    if (m_Resource)
                    {
                        m_OnReturnFn(*m_Resource);
                        if (m_Pool)
                        {
                            m_Pool->push(std::move(m_Resource));
                        }
                    }
                }

                DELETE_COPYABILITY(Resource);

                Resource(Resource&&) noexcept = default;
                Resource& operator=(Resource&&) noexcept = default;

                ResourceType* operator->()
                {
                    return m_Resource.get();
                }
                const ResourceType* operator->() const
                {
                    return m_Resource.get();
                }

            private:
                pool_t       m_Pool;
                item_t       m_Resource;
                on_return_fn m_OnReturnFn;
            };

            using mutex_t = typename ThreadType::mutex;
            using cv_t    = typename ThreadType::cv;
            mutable mutex_t    m_Mutex;
            cv_t               m_Condition;
            std::queue<item_t> m_Queue;
        };

    } // namespace v3

    inline namespace v4
    {
        template <typename ResourceType, typename ThreadType = standard_threads>
        class Pool final : public std::enable_shared_from_this<Pool<ResourceType, ThreadType>>
        {
            struct key
            {
            };
            class UniqueItem;
            using SharedItem = std::shared_ptr<ResourceType>;

            using pool_t = std::shared_ptr<Pool<ResourceType, ThreadType>>;
            using item_t = ResourceType;

        public:
            using resource_type = ResourceType;
            using on_return_fn  = std::function<void(ResourceType&)>;

            Pool(key) {}
            virtual ~Pool() {}

            DELETE_COPYABILITY(Pool);
            DELETE_MOVEABILITY(Pool);

            static pool_t Create()
            {
                return std::make_shared<Pool<ResourceType, ThreadType>>(key());
            }

            void Push(ResourceType&& resource)
            {
                internal_push(std::move(resource));
            }

            void push(ResourceType&& resource)
            {
                internal_push(std::move(resource));
            }

            template <typename... Args>
            void emplace_push(Args&&... args)
            {
                push(ResourceType(std::forward<Args>(args)...));
            }

            template <typename... Args>
            void EmplacePush(Args&&... args)
            {
                push(ResourceType(std::forward<Args>(args)...));
            }

            SharedItem Pop(on_return_fn on_return)
            {
                return pop_shared(on_return);
            }

            SharedItem Pop()
            {
                return pop_shared();
            }

            SharedItem PopWithoutReturn()
            {
                return pop_shared_without_return();
            }

            UniqueItem pop_unique(on_return_fn on_return)
            {
                return UniqueItem(internal_pop(), this->shared_from_this(), on_return);
            }

            UniqueItem pop_unique()
            {
                return pop_unique([](ResourceType&) {});
            }

            UniqueItem pop_unique_without_return()
            {
                return UniqueItem(internal_pop(), nullptr, [](ResourceType&) {});
            }

            SharedItem pop_shared(on_return_fn on_return)
            {
                return std::shared_ptr<item_t>(new item_t(std::move(internal_pop())), [pool = this->shared_from_this(), on_return](item_t* ptr) mutable {
                    on_return(*ptr);
                    pool->push(std::move(*ptr));
                    delete ptr;
                });
            }

            SharedItem pop_shared()
            {
                return pop_shared([](ResourceType&) {});
            }

            SharedItem pop_shared_without_return()
            {
                return std::make_shared<item_t>(std::move(internal_pop()));
            }

            std::size_t Size() const
            {
                std::lock_guard<mutex_t> lock(m_Mutex);
                return m_Queue.size();
            }

            std::size_t size() const
            {
                std::lock_guard<mutex_t> lock(m_Mutex);
                return m_Queue.size();
            }

        private:
            void internal_push(item_t&& ptr)
            {
                {
                    std::lock_guard<mutex_t> lock(m_Mutex);
                    m_Queue.push(std::move(ptr));
                }
                m_Condition.notify_one();
            }

            item_t internal_pop()
            {
                std::unique_lock<mutex_t> lock(m_Mutex);
                m_Condition.wait(lock, [this] { return !m_Queue.empty(); });
                auto ptr = std::move(m_Queue.front());
                m_Queue.pop();
                return std::move(ptr);
            }

            class UniqueItem
            {
            public:
                UniqueItem(item_t resource, pool_t pool, on_return_fn on_return)
                : m_Resource(std::move(resource)), m_Pool(pool), m_OnReturnFn(on_return)
                {
                }

                virtual ~UniqueItem()
                {
                    if (m_Pool)
                    {
                        m_OnReturnFn(m_Resource);
                        m_Pool->push(std::move(m_Resource));
                    }
                }

                DELETE_COPYABILITY(UniqueItem);
                //DELETE_MOVEABILITY(UniqueItem);

                UniqueItem(UniqueItem&& other) noexcept
                : m_Resource(std::move(other.m_Resource)), m_Pool(std::exchange(other.m_Pool, nullptr)), m_OnReturnFn(std::move(other.m_OnReturnFn))
                {
                }
                UniqueItem& operator=(UniqueItem&& other) noexcept
                {
                    m_Resource   = std::move(other.m_Resource);
                    m_Pool       = std::exchange(other.m_Pool, nullptr);
                    m_OnReturnFn = std::exchange(other.m_OnReturnFn, nullptr);
                    return *this;
                }

                ResourceType* operator->()
                {
                    return &m_Resource;
                }
                const ResourceType* operator->() const
                {
                    return &m_Resource;
                }

            private:
                item_t       m_Resource;
                pool_t       m_Pool;
                on_return_fn m_OnReturnFn;
            };


            using mutex_t = typename ThreadType::mutex;
            using cv_t    = typename ThreadType::cv;
            mutable mutex_t    m_Mutex;
            cv_t               m_Condition;
            std::queue<item_t> m_Queue;
        };

        template <typename ResourceType, typename ThreadType = standard_threads>
        class UniquePool final : public std::enable_shared_from_this<UniquePool<ResourceType, ThreadType>>
        {
            struct key
            {
            };

            using pool_t = std::shared_ptr<UniquePool<ResourceType, ThreadType>>;

        public:
            using item_t = std::unique_ptr<ResourceType>;
            using resource_type = ResourceType;
            using on_return_fn  = std::function<void(ResourceType&)>;

            class UniqueItem;
            
            UniquePool(key) {}
            virtual ~UniquePool() {}

            DELETE_COPYABILITY(UniquePool);
            DELETE_MOVEABILITY(UniquePool);

            static pool_t Create()
            {
                return std::make_shared<UniquePool<ResourceType, ThreadType>>(key());
            }

            void push(item_t item)
            {
                internal_push(std::move(item));
            }

            template <typename... Args>
            void emplace_push(Args&&... args)
            {
                internal_push(std::make_unique<ResourceType>(std::forward<Args>(args)...));
            }

            UniqueItem pop_unique(on_return_fn on_return)
            {
                return UniqueItem(internal_pop(), this->shared_from_this(), on_return);
            }

            UniqueItem pop_unique()
            {
                return pop_unique([](ResourceType&) {});
            }

            UniqueItem pop_unique_without_return()
            {
                return UniqueItem(internal_pop(), nullptr, [](ResourceType&) {});
            }

            std::size_t size() const
            {
                std::lock_guard<mutex_t> lock(m_Mutex);
                return m_Queue.size();
            }

        private:
            void internal_push(item_t&& ptr)
            {
                {
                    std::lock_guard<mutex_t> lock(m_Mutex);
                    m_Queue.push(std::move(ptr));
                }
                m_Condition.notify_one();
            }

            item_t internal_pop()
            {
                std::unique_lock<mutex_t> lock(m_Mutex);
                m_Condition.wait(lock, [this] { return !m_Queue.empty(); });
                auto ptr = std::move(m_Queue.front());
                m_Queue.pop();
                return std::move(ptr);
            }

        public:
            class UniqueItem
            {
            public:
                UniqueItem(item_t resource, pool_t pool, on_return_fn on_return)
                : m_Resource(std::move(resource)), m_Pool(pool), m_OnReturnFn(on_return)
                {
                }

                virtual ~UniqueItem()
                {
                    if (m_Pool)
                    {
                        m_OnReturnFn(*m_Resource);
                        m_Pool->push(std::move(m_Resource));
                    }
                }

                DELETE_COPYABILITY(UniqueItem);
                //DELETE_MOVEABILITY(UniqueItem);

                UniqueItem(UniqueItem&& other) noexcept
                : m_Resource(std::move(other.m_Resource)), m_Pool(std::exchange(other.m_Pool, nullptr)), m_OnReturnFn(std::move(other.m_OnReturnFn))
                {
                }
                UniqueItem& operator=(UniqueItem&& other) noexcept
                {
                    m_Resource   = std::move(other.m_Resource);
                    m_Pool       = std::exchange(other.m_Pool, nullptr);
                    m_OnReturnFn = std::exchange(other.m_OnReturnFn, nullptr);
                    return *this;
                }

                ResourceType* operator->()
                {
                    return m_Resource.get();
                }
                const ResourceType* operator->() const
                {
                    return m_Resource.get();
                }

            private:
                item_t       m_Resource;
                pool_t       m_Pool;
                on_return_fn m_OnReturnFn;
            };


            using mutex_t = typename ThreadType::mutex;
            using cv_t    = typename ThreadType::cv;
            mutable mutex_t    m_Mutex;
            cv_t               m_Condition;
            std::queue<item_t> m_Queue;

        };

    } // namespace v4

} // namespace trtlab
