#include "trtlab/tensorrt/model.h"
#include "trtlab/tensorrt/utils.h"

#include <glog/logging.h>

using namespace trtlab;
using namespace TensorRT;

namespace
{
    std::string ProfileSelectorString(nvinfer1::OptProfileSelector selector)
    {
        if (selector == nvinfer1::OptProfileSelector::kMIN)
        {
            return "MIN";
        }
        else if (selector == nvinfer1::OptProfileSelector::kOPT)
        {
            return "OPT";
        }
        else if (selector == nvinfer1::OptProfileSelector::kMAX)
        {
            return "MAX";
        }
        else
        {
            LOG(FATAL) << "unknown profile selector";
        }
    }
} // namespace

Model::Model(engine_t engine, const weights_t& weights) : m_Engine(engine), m_Weights(weights) {}

Model::~Model()
{
    VLOG(2) << "Destroying ICudaEngine " << m_Engine.get();
}

std::string Model::profiles_info() const
{
    std::stringstream ss;
    ss << "Optimization Profiles" << std::endl;

    for (int profile_id = 0; profile_id < m_Engine->getNbOptimizationProfiles(); profile_id++)
    {
        for (auto selector : {nvinfer1::OptProfileSelector::kMIN, nvinfer1::OptProfileSelector::kOPT, nvinfer1::OptProfileSelector::kMAX})
        {
            ss << "Profile " << profile_id << " "  << profile_info(profile_id, selector);
        }
    }
    return ss.str();
}

std::string Model::profile_info(std::uint32_t profile_id, nvinfer1::OptProfileSelector selector) const
{
    CHECK_LT(profile_id, m_Engine->getNbOptimizationProfiles());
    std::stringstream ss;

    for (int binding_id = 0; binding_id < m_Engine->getNbBindings(); binding_id++)
    {
        if (m_Engine->bindingIsInput(binding_id))
        {
            auto dims = m_Engine->getProfileDimensions(binding_id, profile_id, selector);
            ss << ProfileSelectorString(selector) << " - input_binding " << binding_id << ": name=" << m_Engine->getBindingName(binding_id)
               << "; ";
            ss << TensorRT::Model::dims_info(dims);
            ss << std::endl;
        }
    }
    return ss.str();
}

std::string Model::bindings_info() const
{
    std::stringstream ss;
    for (int i = 0; i < m_Engine->getNbBindings(); i++)
    {
        ss << binding_info(i) << std::endl;
    }
    return ss.str();
}

std::string Model::binding_info(std::uint32_t binding_id) const
{
    std::stringstream ss;
    ss << "binding " << binding_id << ": name=" << m_Engine->getBindingName(binding_id) << "; ";
    ss << dims_info(m_Engine->getBindingDimensions(binding_id));
    ss << "; isInput=" << (m_Engine->bindingIsInput(binding_id) ? "TRUE" : "FALSE");
    return ss.str();
}

std::string Model::dims_info(const nvinfer1::Dims& dims)
{
    std::stringstream ss;
    ss << "ndims=" << dims.nbDims << "; [";
    for (int i = 0; i < dims.nbDims; i++)
        ss << " " << dims.d[i];
    ss << " ]";
    return ss.str();
}

std::size_t Model::binding_element_count(std::uint32_t binding_id) const
{
    auto dims = m_Engine->getBindingDimensions(binding_id);
    std::size_t count = 1;
    for(int i=0; i<dims.nbDims; i++)
    {
        count *= dims.d[i];
    }
    return count;
}

std::size_t Model::binding_size_in_bytes(std::uint32_t binding_id) const
{
    auto dtype = m_Engine->getBindingDataType(binding_id);
    return data_type_size(dtype) * binding_element_count(binding_id);
}