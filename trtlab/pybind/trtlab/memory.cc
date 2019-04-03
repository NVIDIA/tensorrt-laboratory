
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

#include "trtlab/core/memory/allocator.h"
#include "trtlab/core/memory/descriptor.h"
#include "trtlab/core/memory/malloc.h"
#include "trtlab/cuda/memory/cuda_device.h"
#include "trtlab/cuda/memory/cuda_pinned_host.h"
#include "trtlab/core/types.h"

#include "dlpack.h"
#include "numpy.h"

using namespace trtlab;
using namespace trtlab::python;

void init_ext_memory(py::module &m)
{
    py::class_<types::dtype>(m, "dtype")
        .def("__repr__",
            [](const types::dtype& self) { return self.Description(); });

    m.attr("int8") = types::int8;
    m.attr("int16") = types::int16;
    m.attr("int32") = types::int32;
    m.attr("int64") = types::int64;
    m.attr("uint8") = types::uint8;
    m.attr("uint16") = types::uint16;
    m.attr("uint32") = types::uint32;
    m.attr("uint64") = types::uint64;
    m.attr("fp16") = types::fp16;
    m.attr("fp32") = types::fp32;
    m.attr("fp64") = types::fp64;


    py::class_<CoreMemory, std::shared_ptr<CoreMemory>>(m, "CoreMemory")
        .def_property_readonly("shape", &CoreMemory::Shape)
        .def_property_readonly("strides", &CoreMemory::Strides)
        .def_property_readonly("dtype", &CoreMemory::DataType)
        .def("view_reshape", (void (CoreMemory::*)(const std::vector<mem_size_t>&)) &CoreMemory::Reshape)
        .def("view_reshape", (void (CoreMemory::*)(const std::vector<mem_size_t>&, const types::dtype&)) &CoreMemory::Reshape)
        .def("to_dlpack", [](py::object self) { return DLPack::Export(self); })
        .def("__repr__",
             [](const CoreMemory& mem) { return "<trtlab.Memory: " + mem.Description() + ">"; });

    py::class_<HostMemory, std::shared_ptr<HostMemory>, CoreMemory>(m, "HostMemory")
        .def("to_numpy", [](py::object self) { return NumPy::Export(self); })
        .def("__repr__", [](const HostMemory& mem) {
            return "<trtlab.HostMemory: " + mem.Description() + ">";
        });

    py::class_<DeviceMemory, std::shared_ptr<DeviceMemory>, CoreMemory>(m, "DeviceMemory")
        .def("__repr__", [](const DeviceMemory& mem) {
            return "<trtlab.DeviceMemory: " + mem.Description() + ">";
        });

    m.def("from_dlpack", [](py::capsule obj) {
        auto core = DLPack::Import(obj);
        return core;
    });

    m.def("malloc", [](int64_t size) {
        std::shared_ptr<HostMemory> mem = std::make_shared<Allocator<Malloc>>(size);
        return mem;
    });

    m.def("cuda_malloc_host", [](int64_t size) {
        std::shared_ptr<HostMemory> mem = std::make_shared<Allocator<CudaPinnedHostMemory>>(size);
        return mem;
    });

    m.def("cuda_malloc", [](int64_t size) {
        std::shared_ptr<DeviceMemory> mem = std::make_shared<Allocator<CudaDeviceMemory>>(size);
        return mem;
    });

    m.def("dlpack_from_malloc",
          [](int64_t size) { return DLPack::Export(std::make_shared<Allocator<Malloc>>(size)); });

    m.def("dlpack_from_cuda_malloc", [](int64_t size) {
        return DLPack::Export(std::move(std::make_unique<Allocator<CudaDeviceMemory>>(size)));
    });
}