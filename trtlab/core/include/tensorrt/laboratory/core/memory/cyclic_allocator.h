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

#include <memory>

#include "tensorrt/laboratory/core/memory/smart_stack.h"
#include "tensorrt/laboratory/core/pool.h"
#include "tensorrt/laboratory/core/utils.h"

#include <glog/logging.h>

namespace trtlab {

/**
 * @brief CyclicAllocator
 *
 * Ring of RotatingSegments which are specialized MemoryStack objects of MemoryType.
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
 * RotatingSegment::Allocate returns a specialized std::shared_ptr<BaseType> which is
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
 * Specialized CyclicAllocator could monitor the utilization of the Pool of RotatingSegments
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
 * @tparam MemoryType
 */
template<class MemoryType>
class CyclicAllocator
{
  public:
    using RotatingSegment = SmartStack<MemoryType>;
    using Descriptor = typename RotatingSegment::StackDescriptor;

    CyclicAllocator(size_t segments, size_t bytes_per_segment)
        : m_Segments(Pool<RotatingSegment>::Create()), m_MaximumAllocationSize(bytes_per_segment)
    {
        DLOG(INFO) << "Allocating " << segments << " rotating segments "
                   << "with " << BytesToString(bytes_per_segment) << "/segment";

        for(int i = 0; i < segments; i++)
        {
            InternalPushSegment();
        }

        m_CurrentSegment = InternalPopSegment();
        m_Alignment = m_CurrentSegment->Alignment();
    }

    virtual ~CyclicAllocator() { m_CurrentSegment.reset(); }

    Descriptor Allocate(size_t size) { return InternalAllocate(size); }

    size_t MaxAllocationSize() const { return m_MaximumAllocationSize; }

    /*
        std::shared_ptr<MemoryStack<BaseType>> AllocateStack(size_t size)
        {
            return std::make_shared<MemoryStack<BaseType>>(InternalAllocate(size));
        }
    */
    void AddSegment() { InternalPushSegment(); }

    void DropSegment() { InternalDropSegment(); }

    auto AvailableSegments() { return m_Segments->Size() + (m_CurrentSegment ? 1 : 0); }

    auto AvailableBytes()
    {
        return m_Segments->Size() * m_MaximumAllocationSize +
               (m_CurrentSegment ? m_CurrentSegment->Available() : 0);
    }

    auto Alignment() { return m_Alignment; }

  private:
    // The returned shared_ptr<MemoryType> holds a reference to the RotatingSegment object
    // which ensures the RotatingSegment cannot be returned to the Pool until all its
    // reference count goes to zero
    Descriptor InternalAllocate(size_t size)
    {
        DLOG(INFO) << "Requested Allocation: " << size << " bytes";
        CHECK_LE(size, m_MaximumAllocationSize)
            << "Requested allocation of " << size << " bytes exceeds the maximum allocation "
            << "size of " << m_MaximumAllocationSize << " for this CyclicAllocator.";
        std::lock_guard<std::mutex> lock(m_Mutex);
        if(!m_CurrentSegment || size > m_CurrentSegment->Available())
        {
            DLOG(INFO) << "Current Segment cannot fulfill the request; rotate segment";
            m_CurrentSegment.reset(); // explicitily drop the current segment -> returns to pool
            m_CurrentSegment = InternalPopSegment(); // get the next segment from pool
        }
        auto retval = m_CurrentSegment->Allocate(size);
        // TODO: proactive release should be dependent the recent allocations statistics
        if(!m_CurrentSegment->Available())
        {
            DLOG(INFO) << "Proactively releasing the current segment as it is maxed";
            m_CurrentSegment.reset();
        }
        return retval;
    }

    void InternalPushSegment()
    {
        // auto stack = std::make_unique<MemoryStack<MemoryType>>(m_MaximumAllocationSize);
        // auto segment = RotatingSegment::make_shared(std::move(stack));
        auto segment = std::make_shared<RotatingSegment>(m_MaximumAllocationSize);
        m_Segments->Push(segment);
        DLOG(INFO) << "Pushed New Rotating Segment " << segment.get() << " to Pool";
    }

    auto InternalPopSegment()
    {
        auto val = m_Segments->Pop([](RotatingSegment* segment) {
            DLOG(INFO) << "Returning RotatingSegment " << segment << " to Pool";
            segment->Reset();
        });
        DLOG(INFO) << "Acquired RotatingSegment " << val.get() << " from Pool";
        return val;
    }

    auto InternalDropSegment()
    {
        // Remote a Segment from the Ring
        return m_Segments->PopWithoutReturn();
    }

    std::shared_ptr<Pool<RotatingSegment>> m_Segments;
    std::shared_ptr<RotatingSegment> m_CurrentSegment;
    std::mutex m_Mutex;
    const size_t m_MaximumAllocationSize;
    size_t m_Alignment;
};

} // namespace trtlab
