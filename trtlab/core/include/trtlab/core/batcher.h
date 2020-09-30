#pragma once

#include <chrono>
#include <future>
#include <memory>
#include <optional>
#include <queue>
#include <vector>

#include <glog/logging.h>

namespace trtlab
{
    // Defines the batching logic used by the Batcher class.
    // This class only provides the core logic for batching,
    // managed the state, and provides the synchronization
    // future-promise mapping.
    // There are no public methods besides the constructor
    // and deconstructor.  There is also no internal mutex
    // for synchronization.  This class is designed to be
    // privately inherited by the Batcher which will provide
    // the necessary threads, mutexes, etc. for use.
    template <typename T, typename ThreadType>
    class StandardBatcher
    {
        using promise_t       = typename ThreadType::template promise<void>;
        using shared_future_t = typename ThreadType::template shared_future<void>;

    public:
        StandardBatcher(std::size_t max_batch_size) : m_MaxBatchSize(max_batch_size), m_BatchCounter(0)
        {
        }
        virtual ~StandardBatcher(){};

        StandardBatcher(StandardBatcher&&) = default;

    public:
        using thread_type = ThreadType;
        using clock_type  = std::chrono::high_resolution_clock;

        struct Batch
        {
            std::vector<T>    items;
            mutable promise_t promise;
            std::size_t       batch_id;
        };

        using batch_item  = T;
        using batch_type  = std::optional<Batch>;
        using future_type = shared_future_t;
        using release_fn  = std::function<void(void)>;

        // Enqueue a batch item
        // This is the data-in interface to the batcher.
        // This method is intended to collect together multiple BatchItems.
        // For each BatchItem a shared_future (unique to the batch) is returned
        // to the caller.  The shared_future is not unique to the BatchItem.
        // This future is the caller portion of the Batcher's promise to
        // use the data and to fulfill the promise by setting some value
        // of ReturnType at a later time.  Until the batcher fulfills the
        // promise, the caller should not modify or delete any of the data
        // passed to the batcher.  This is a performance optimization to avoid
        // data copies.
        future_type enqueue(T);

        // Updates the state of the batcher and tests to determine if the
        // conditions for closing the batching window have been met.
        // If the batch is complete, the optional to the batch is fulfilled;
        // otherwise, the optional is empty or std::nullopt;
        batch_type update();

        // Closes the current batch and returns an optional batch_type
        batch_type close_batch();

        bool empty()
        {
            return !m_State.has_value();
        }

        clock_type::time_point start_time()
        {
            return (m_State ? m_State->start_time : clock_type::time_point{});
        }

    private:
        struct State
        {
            Batch                  batch;
            shared_future_t        future;
            clock_type::time_point start_time;
        };

        State create_state();

        std::size_t          m_MaxBatchSize;
        std::optional<State> m_State;
        std::size_t          m_BatchCounter;
    };

    template <typename T, typename ThreadType>
    typename StandardBatcher<T, ThreadType>::future_type StandardBatcher<T, ThreadType>::enqueue(T new_item)
    {
        // no state is ok - simiply create a state to start the timing window
        if (!m_State)
        {
            m_State = create_state();
        }

        // perform some checks
        DCHECK_LT(m_State->batch.items.size(), m_MaxBatchSize);

        // push back new item (memory has been reserved)
        m_State->batch.items.push_back(std::move(new_item));

        return m_State->future;
    }

    template <typename T, typename ThreadType>
    typename StandardBatcher<T, ThreadType>::batch_type StandardBatcher<T, ThreadType>::update()
    {
        if (m_State)
        {
            if (m_State->batch.items.size() == m_MaxBatchSize) /* || clock_type::now() > m_State->deadline)*/
            {
                batch_type batch(std::move(m_State->batch));
                m_State = std::nullopt;
                return batch;
            }
        }
        return std::nullopt;
    }

    template <typename T, typename ThreadType>
    typename StandardBatcher<T, ThreadType>::batch_type StandardBatcher<T, ThreadType>::close_batch()
    {
        if (m_State)
        {
            batch_type batch(std::move(m_State->batch));
            m_State = std::nullopt;
            return batch;
        }
        return std::nullopt;
    }

    template <typename T, typename ThreadType>
    typename StandardBatcher<T, ThreadType>::State StandardBatcher<T, ThreadType>::create_state()
    {
        State state;
        state.batch.items.reserve(m_MaxBatchSize);
        state.batch.batch_id = m_BatchCounter++;
        state.future         = state.batch.promise.get_future().share();
        state.start_time     = clock_type::now();
        return state;
    }

} // namespace trtlab