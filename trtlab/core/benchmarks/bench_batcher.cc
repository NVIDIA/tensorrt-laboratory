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
#include <benchmark/benchmark.h>

#include <trtlab/core/standard_threads.h>

#include <trtlab/core/batcher.h>
#include <trtlab/core/dispatcher.h>

using namespace trtlab;

static void batcher_standard_batcher_int(benchmark::State& state)
{
    StandardBatcher<int, standard_threads> batcher(state.range(0));
    std::size_t                            counter = 0;

    for (auto _ : state)
    {
        auto future = batcher.enqueue(++counter);
        auto batch  = batcher.update();

        if (batch)
        {
            batch->promise.set_value();
            future.wait();
        }
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

struct audio_state
{
    const std::uint16_t*  data;
    std::size_t           size;
    std::shared_ptr<long> state;
};

static void batcher_standard_batcher_audio(benchmark::State& state)
{
    StandardBatcher<audio_state, standard_threads> batcher(state.range(0));
    std::size_t                                    counter = 0;

    for (auto _ : state)
    {
        auto future = batcher.enqueue({nullptr, 0ul, nullptr});
        auto batch  = batcher.update();

        if (batch)
        {
            batch->promise.set_value();
            future.wait();
        }
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

static void batcher_engine(benchmark::State& state)
{
    const std::size_t batch_size = state.range(0);

    auto execute_on_batch = [](const std::vector<int>& batch, std::function<void()> free_inputs) { free_inputs(); };

    auto                                   thread_pool = std::make_shared<ThreadPool>(1);
    auto                                   task_pool   = std::make_shared<DeferredShortTaskPool>();
    StandardBatcher<int, standard_threads> batcher(batch_size);
    Dispatcher<decltype(batcher)> dispatcher(std::move(batcher), std::chrono::milliseconds(15), thread_pool, task_pool, execute_on_batch);

    std::queue<std::shared_future<void>> f;

    int pre_load = 3;

    for (int i = 0; i < pre_load; i++)
    {
        f.push(dispatcher.enqueue(0));
    }
    for (int i = 0; i < (batch_size - 1) * pre_load; i++)
    {
        dispatcher.enqueue(i);
    }

    for (auto _ : state)
    {
        f.push(dispatcher.enqueue(0));
        for (int i = 0; i < batch_size - 1; i++)
        {
            dispatcher.enqueue(i);
        }
        f.front().wait();
        f.pop();
    }

    while (!f.empty())
    {
        f.front().wait();
        f.pop();
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * state.range(0));
}

BENCHMARK(batcher_standard_batcher_int)->RangeMultiplier(2)->Range(1, 1 << 7);
BENCHMARK(batcher_standard_batcher_audio)->RangeMultiplier(2)->Range(1 << 6, 1 << 7);
BENCHMARK(batcher_engine)->RangeMultiplier(2)->Range(4, 1 << 6)->UseRealTime()->MinTime(3.0);
;