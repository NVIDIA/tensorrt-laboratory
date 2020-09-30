#include "trtlab/tensorrt/execution_context.h"
#include "trtlab/tensorrt/utils.h"

#include <glog/logging.h>

using namespace trtlab;
using namespace TensorRT;

ExecutionContext::ExecutionContext(model_t model)
: m_Model(model), m_Context(nv_unique(model->engine().createExecutionContextWithoutDeviceMemory()))
{
}

ExecutionContext::~ExecutionContext()
{
    if (m_Context)
    {
        VLOG(2) << "Destroying IExecutionContext " << m_Context.get();
    }
}

std::size_t ExecutionContext::binding_size_in_bytes(std::uint32_t binding_id)
{
    auto dims = m_Context->getBindingDimensions(binding_id);
    auto dtype = m_Context->getEngine().getBindingDataType(binding_id);
    return dims_element_count(dims) * data_type_size(dtype);
}
