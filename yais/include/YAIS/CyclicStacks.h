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
#ifndef _YAIS_CYCLIC_STACK_H_
#define _YAIS_CYCLIC_STACK_H_

#include <vector>

#include "YAIS/Memory.h"
#include "YAIS/MemoryStack.h"
#include "YAIS/Pool.h"

#include <glog/logging.h>

namespace yais
{

/**
 * @brief CyclicStacks
 *
 * Ring of RotatingSegments which are specialized MemoryStack objects of AllocatorType.
 * This type of allocator is best used in situations where the memory allocated is temporary
 * and the lifespan of the allocation is reasonably consistent with its peers. One usecase
 * for this allocator is to provide the scratch space for input/output bindings and activation
 * storage needed by TensorRT for a given inference transaction.
 * 
 * The set of RotatingSegments that form the ring is managed by the Pool resource manager.
 * RotatingSegments that are Pop()'ed from the pool will automatically be returned to the 
 * Pool and have their stack pointers Reset when the reference count of the RotatingSegment
 * goes to zero.
 * 
 * RotatingSegment does not conform to the IMemoryStack inteface as the Allocate function
 * returns a different and unique object.
 * 
 * MemoryStack::Allocate returns a void* to the start of a continuous segment of a given
 * size reserved on the stack.
 * 
 * RotatingSegment::Allocate returns a specialized std::shared_ptr<IMemory> which is
 * instantiated from the same void* from the RotatingSegment's internal MemoryStack.
 * The returned shared_ptr is created with a custom deleter which holds a reference to 
 * the RotatingSegment that created it. This capture of shared_from_this ensures that
 * the RotatingSegment is only returned to the Pool after all allocations from the
 * segment have been released.
 * 
 * Allocations are performed on one segment until that segment is no longer capable of
 * meeting the allocation requested.  In that situation, a new/fresh segment is pulled
 * from the Pool and allocation continutes.
 * 
 * The ineffiency or memory lost to fragmentation is whatever is left on the stack when
 * the segment is rotated.  Say there is 1000mb stack with 50mb remaining and the next
 * request is 100mb. 50mb of memory for that segment goes unused.
 * 
 * To this end, segments should be large.  However, there is a competing force that 
 * advocates for smaller segments.  A segment can only be reused when all the allocations
 * that have been reserved on it have been released.
 * 
 * Specialized CyclicStacks could monitor the utilization of the Pool of RotatingSegments
 * and can grow or shrink the ring based on some defined critria.  Similarly, one could
 * manually grow or shrink the ring.
 * 
 * RotatingSegments do not need to be a fixed size; however, the default implementation used
 * constant sized segments.
 * 
 *  Common Guidelines:
 *  - Segments should be sized larger than the largest allowed allocation.
 *  - Depending on the mean and variance in your allocations, you want to size your segments
 *    to be at least twice as large as your largest segment.
 *  - The more common larger segments are allocated, the larger the base RotatingSgement
 *    should be
 *  - The number of segments should be large enough so that the queue consuming the prior
 *    allocations can finish one segment before the last segment is fully consumed.
 *  
 * @tparam AllocatorType 
 */
template <class AllocatorType>
class CyclicStacks
{
  public:
    CyclicStacks(size_t segments, size_t bytes_per_segment) 
        : m_MaximumAllocationSize(bytes_per_segment)
    {
        LOG(INFO) 
            << "Allocating " << segments << " rotating segments "
            << "with " << BytesToString(bytes_per_segment) << "/segment";

        m_Segments = Pool<RotatingSegment>::Create();

        for(int i=0; i<segments; i++) {
            auto stack = MemoryStack<AllocatorType>::make_shared(bytes_per_segment);
            auto segment = std::make_shared<RotatingSegment>(stack);
            m_Segments->Push(segment);
            LOG(INFO) << "RotatingSegment " << i << ": " << segment.get();
        }

        m_CurrentSegment = NextSegment();
    }

    virtual ~CyclicStacks() {}

    std::shared_ptr<IMemory> AllocateBuffer(size_t size) {
        return Allocate(size);
    }
    std::shared_ptr<MemoryStack<IMemory>> AllocateStack(size_t size) {
        return std::make_shared<MemoryStack<IMemory>>(Allocate(size));
    }

    auto AvailableSegments()
    {
        std::unique_lock<std::mutex> l(m_Mutex);
        return m_Segments->Size() + 1;
    }

    auto AvailableBytes()
    {
        std::unique_lock<std::mutex> l(m_Mutex);
        return m_Segments->Size() * m_MaximumAllocationSize + m_CurrentSegment->Available();
    }

    auto Alignment()
    {
        std::unique_lock<std::mutex> l(m_Mutex);
        return m_CurrentSegment->m_Stack->Alignment();
    }

  private:
    // The returned shared_ptr<IMemory> holds a reference to the RotatingSegment object
    // which ensures the RotatingSegment cannot be returned to the Pool until all its
    // reference count goes to zero
    std::shared_ptr<IMemory> Allocate(size_t size) {
        CHECK_LE(size, m_MaximumAllocationSize)
            << "Requested allocation of " << size << " bytes exceeds the maximum allocations "
            << "size " << m_MaximumAllocationSize << " for this CyclicStacks memory allocator.";
        std::unique_lock<std::mutex> l(m_Mutex);
        if(size > m_CurrentSegment->Available()) {
            LOG(INFO) << "Current Segment cannot fulfill the request; rotate segment";
            // Removing the CHECK if you want the program to block on memory being returned.
            // This is safe if you assure yourself that the program is continuing to make
            // forward progress and that eventually RotatingSegments will be returned to
            // the Pool
            CHECK(m_Segments->Size()) << "OOM";
            m_CurrentSegment = NextSegment();
            CHECK_LE(size, m_CurrentSegment->Available());
        }
        return m_CurrentSegment->Allocate(size);
    }

    auto NextSegment() {
        return m_Segments->Pop([](RotatingSegment *segment) {
            LOG(INFO) << "Returning RotatingSegment to Pool";
            segment->m_Stack->Reset();
        });
    }

    class RotatingSegment : public std::enable_shared_from_this<RotatingSegment>
    {
      public:
        RotatingSegment(std::shared_ptr<MemoryStack<AllocatorType>> stack)
            : m_Stack(stack) {}

        virtual ~RotatingSegment() {}

        std::shared_ptr<IMemory> Allocate(size_t size) {
            LOG(INFO) << "RotatingSegment::Allocate: " << size;
            CHECK(m_Stack);
            CHECK_LE(size, m_Stack->Available());
            auto segment = this->shared_from_this();
            auto ptr = m_Stack->Allocate(size);
            auto ret = AllocatorType::UnsafeWrapRawPointer(
                ptr, size, [segment](IMemory *p){ delete p; });
            LOG(INFO) 
                << "Allocated " << ret->Size() << " starting at " << ret->Data() 
                << " on segment " << segment.get();
            return ret;
        }

        size_t Available() {
            return m_Stack->Available();
        }

      private:
        std::shared_ptr<MemoryStack<AllocatorType>> m_Stack;
        friend class CyclicStacks;
    };

    std::shared_ptr<Pool<RotatingSegment>> m_Segments;
    std::shared_ptr<RotatingSegment> m_CurrentSegment;
    std::mutex m_Mutex;
    const size_t m_MaximumAllocationSize;
};

} // end namespace yais

#endif // _YAIS_CYCLIC_STACK_H_
