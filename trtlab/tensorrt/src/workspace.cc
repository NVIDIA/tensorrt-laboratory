#include <trtlab/tensorrt/workspace.h>
#include <trtlab/tensorrt/utils.h>

#include <trtlab/cuda/common.h>
#include <trtlab/cuda/memory/cuda_allocators.h>

using namespace trtlab;
using namespace TensorRT;

WorkspaceBase::WorkspaceBase()
{
    CHECK_CUDA(cudaStreamCreate(&m_Stream));
}

WorkspaceBase::~WorkspaceBase()
{
    CHECK_CUDA(cudaStreamSynchronize(m_Stream));
    CHECK_CUDA(cudaStreamDestroy(m_Stream));
}

StaticSingleModelGraphWorkspace::StaticSingleModelGraphWorkspace(std::shared_ptr<Model> model) : m_Context(model)
{
    auto& engine      = m_Context.engine();
    auto  cuda_malloc = memory::make_cuda_allocator();

    std::stringstream ss;
    ss << this;
    m_Name = ss.str();

    //m_Context.context().setOptimizationProfile(0);

    for (int i = 0; i < engine.getNbBindings(); i++)
    {
        auto bytes = m_Context.binding_size_in_bytes(i);
        VLOG(2) << "binding " << i << ": bytes = " << bytes;
        m_Bindings.emplace_back(cuda_malloc.allocate_descriptor(bytes));
        m_BindingPointers.push_back(m_Bindings[i].data());
    }

    m_DeviceMemory = cuda_malloc.allocate_descriptor(engine.getDeviceMemorySize());
    m_Context.context().setDeviceMemory(m_DeviceMemory.data());
    VLOG(2) << "execution context device memory: " << m_DeviceMemory.size();

    CHECK_CUDA(cudaStreamCreate(&m_Stream));

    // warm up and let mContext do cublas initialization
    m_Context.context().enqueueV2(m_BindingPointers.data(), m_Stream, nullptr);

    // create graph
    //CHECK_CUDA(cudaStreamBeginCapture(m_Stream, cudaStreamCaptureModeThreadLocal));
    CHECK_CUDA(cudaStreamBeginCapture(m_Stream, cudaStreamCaptureModeRelaxed));
    m_Context.context().enqueueV2(m_BindingPointers.data(), m_Stream, nullptr);
    CHECK_CUDA(cudaStreamEndCapture(m_Stream, &m_Graph));

    // create graph executor
    CHECK_CUDA(cudaGraphInstantiate(&m_GraphExecutor, m_Graph, NULL, NULL, 0));
}

StaticSingleModelGraphWorkspace::~StaticSingleModelGraphWorkspace()
{
    DVLOG(3) << "StaticSingleModelGraphWorkspace Deconstructor";
    CHECK_CUDA(cudaStreamSynchronize(m_Stream));

    DVLOG(4) << "Destroying GraphExecutors";
    CHECK_CUDA(cudaGraphExecDestroy(m_GraphExecutor));

    DVLOG(4) << "Destroying Graphs";
    CHECK_CUDA(cudaGraphDestroy(m_Graph));

    CHECK_CUDA(cudaStreamDestroy(m_Stream));
}

void StaticSingleModelGraphWorkspace::enqueue()
{
    CHECK_CUDA(cudaGraphLaunch(m_GraphExecutor, m_Stream));
}

memory::descriptor& StaticSingleModelGraphWorkspace::binding(std::uint32_t binding_id)
{
    DCHECK_LT(binding_id, m_Bindings.size());
    return m_Bindings[binding_id];
}

std::size_t StaticSingleModelGraphWorkspace::batch_size()
{
    auto dims = m_Context.context().getBindingDimensions(0);
    return dims.d[0];
}

BenchmarkWorkspace::BenchmarkWorkspace(std::shared_ptr<Model> model) : StaticSingleModelGraphWorkspace(model)
{
    auto pinned_alloc = memory::make_allocator(memory::cuda_malloc_host_allocator());

    for (int i = 0; i < exec_ctx().engine().getNbBindings(); i++)
    {
        auto bytes = binding(i).size();
        VLOG(2) << "binding " << i << ": bytes = " << bytes;
        m_HostBindings.emplace_back(pinned_alloc.allocate_descriptor(bytes));
    }
}

void BenchmarkWorkspace::async_h2d()
{
    for (int i = 0; i < m_HostBindings.size(); i++)
    {
        if (exec_ctx().engine().bindingIsInput(i))
        {
            CHECK_CUDA(
                cudaMemcpyAsync(binding(i).data(), m_HostBindings[i].data(), m_HostBindings[i].size(), cudaMemcpyHostToDevice, stream()));
        }
    }
}

void BenchmarkWorkspace::async_d2h()
{
    for (int i = 0; i < m_HostBindings.size(); i++)
    {
        if (!exec_ctx().engine().bindingIsInput(i))
        {
            CHECK_CUDA(
                cudaMemcpyAsync(m_HostBindings[i].data(), binding(i).data(), m_HostBindings[i].size(), cudaMemcpyDeviceToHost, stream()));
        }
    }
}

TimedBenchmarkWorkspace::TimedBenchmarkWorkspace(std::shared_ptr<Model> model) : BenchmarkWorkspace(model)
{
    CHECK_CUDA(cudaEventCreate(&m_Start));
    CHECK_CUDA(cudaEventCreate(&m_CompleteAsyncH2D));
    CHECK_CUDA(cudaEventCreate(&m_CompleteCompute));
    CHECK_CUDA(cudaEventCreate(&m_CompleteAsyncD2H));
}

void TimedBenchmarkWorkspace::enqueue_pipeline()
{
    CHECK_CUDA(cudaEventRecord(m_Start, stream()));
    async_h2d();
    CHECK_CUDA(cudaEventRecord(m_CompleteAsyncH2D, stream()));
    enqueue();
    CHECK_CUDA(cudaEventRecord(m_CompleteCompute, stream()));
    async_d2h();
    CHECK_CUDA(cudaEventRecord(m_CompleteAsyncD2H, stream()));
}

float TimedBenchmarkWorkspace::get_compute_time_ms()
{
    float ms = 0.0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, m_CompleteAsyncH2D, m_CompleteCompute));
    return ms;
}

float TimedBenchmarkWorkspace::get_h2d_time_ms()
{
    float ms = 0.0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, m_Start, m_CompleteAsyncH2D));
    return ms;
}

float TimedBenchmarkWorkspace::get_d2h_time_ms()
{
    float ms = 0.0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, m_CompleteCompute, m_CompleteAsyncD2H));
    return ms;
}
