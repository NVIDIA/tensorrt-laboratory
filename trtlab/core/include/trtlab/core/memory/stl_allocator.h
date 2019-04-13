
#include <cstring>

#include "trtlab/core/memory/bytes.h"
#include "trtlab/core/memory/cyclic_allocator.h"

#include <gtest/gtest.h>

namespace trtlab {
namespace stl {

template<typename T, typename MemoryType>
struct temporary_allocator;

template<typename MemoryType>
struct temporary_smart_pool
{
  private:
    static std::unique_ptr<CyclicAllocator<MemoryType>>& singleton()
    {
        static std::unique_ptr<CyclicAllocator<MemoryType>> _singleton;
        return _singleton;
    }

    template<typename T, typename MT>
    friend class temporary_allocator;
};

template<typename T, typename MemoryType>
struct temporary_allocator
{
    using value_type = T;

    temporary_allocator()
    {
        if(!temporary_smart_pool<MemoryType>::singleton())
        {
            DLOG(INFO) << "Initializing global cyclic allocator";
            auto provider = std::make_unique<CyclicAllocator<MemoryType>>(10, 128 * 1024 * 1024);
            temporary_smart_pool<MemoryType>::singleton() = std::move(provider);
        }
    }

    temporary_allocator(const temporary_allocator& other) { DLOG(INFO) << "copy_ctor: noop"; }

    template<typename U>
    temporary_allocator(const temporary_allocator<U, MemoryType>& other)
    {
        DLOG(INFO) << "templated copy_ctor: noop";
    }

    temporary_allocator(temporary_allocator&& other) : m_BytesMap(std::move(other.m_BytesMap))
    {
        DLOG(INFO) << "move_ctor: moves the bytes map";
    }

    T* allocate(std::size_t n)
    {
        DCHECK(temporary_smart_pool<MemoryType>::singleton());
        auto b =
            std::move(temporary_smart_pool<MemoryType>::singleton()->AllocateBytes(n * sizeof(T)));
        auto ptr = b.Data();
        m_BytesMap[ptr] = std::move(b);
        // m_Bytes = std::move(b);
        DLOG(INFO) << "Allocating: " << n * sizeof(T) << " bytes @ " << ptr;
        return reinterpret_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t n)
    {
        DLOG(INFO) << "Releasing " << n * sizeof(T) << " bytes @ " << (void*)p;
        auto search = m_BytesMap.find((void*)p);
        DCHECK(search != m_BytesMap.end());
        m_BytesMap.erase((void*)p);
        // m_Bytes.Release();
    }

  private:
    std::map<void*, Bytes<MemoryType>> m_BytesMap;
    Bytes<MemoryType> m_Bytes;
};

template<class T, class U, class MemoryTypeT, class MemoryTypeU>
bool operator==(const temporary_allocator<T, MemoryTypeT>&,
                const temporary_allocator<U, MemoryTypeU>&)
{
    return std::is_same<MemoryTypeT, MemoryTypeU>::value;
}

/*
template<class T, class U, class MemoryTypeT, class MemoryTypeU>
bool operator!=(const temporary_allocator<T, MemoryTypeT>&,
                const temporary_allocator<U, MemoryTypeU>&)
{
    return !std::is_same<MemoryTypeT, MemoryTypeU>::value;
}
*/



template<typename MemoryType>
struct temporary_raw_pool
{
  private:
    class cyclic_raw
    {
      public:
        using Stack = MemoryStack<MemoryType>;
        using StackPtr = std::unique_ptr<Stack>;
        using Descriptor = std::pair<void*, Stack*>;

        Descriptor Allocate(size_t size)
        {
            std::lock_guard<std::mutex> lock(m_Mutex);
            if(!m_CurrentStack || size > m_CurrentStack->Available())
            {
                if(m_AvailableStacks.size() == 0)
                {
                    throw std::runtime_error("allocator oom");
                }
                m_CurrentStack = m_AvailableStacks.front();
                m_AvailableStacks.pop();
                m_UseCount[m_CurrentStack] = 0;
            }
            void* ptr = m_CurrentStack->Allocate(size);
            m_UseCount[m_CurrentStack]++;
        }

      private:
        std::queue<Stack*> m_AvailableStacks;
        std::map<Stack*, uint64_t> m_UseCount;
        Stack* m_CurrentStack;
        mutable std::mutex m_Mutex;
    };

    static std::unique_ptr<CyclicAllocator<MemoryType>>& singleton()
    {
        static std::unique_ptr<CyclicAllocator<MemoryType>> _singleton;
        return _singleton;
    }

    template<typename T, typename MT>
    friend class temporary_allocator;
};


} // namespace stl
} // namespace trtlab