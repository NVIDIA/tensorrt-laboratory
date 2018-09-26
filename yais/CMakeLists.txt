# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

add_library(yais
  src/Affinity.cc
  src/DeviceInfo.cc
  src/Executor.cc
  src/Metrics.cc
  src/Memory.cc
  src/MemoryStack.cc
  src/Server.cc
  src/ThreadPool.cc
)

add_library(yais-tensorrt
  src/TensorRT/TensorRT.cc
  src/TensorRT/Runtime.cc
  src/TensorRT/Model.cc
  src/TensorRT/Buffers.cc
  src/TensorRT/Bindings.cc
  src/TensorRT/ExecutionContext.cc
  src/TensorRT/ResourceManager.cc
)

target_include_directories(yais PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${CUDA_INCLUDE_DIRS}
  PRIVATE src
)

target_include_directories(yais-tensorrt PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${CUDA_INCLUDE_DIRS}
  ${TensorRT_INCLUDE_DIRS}
  PRIVATE src/TensorRT
)

# If we have compiler requirements for this library, list them
# here
# target_compile_features(lib
#     PUBLIC cxx_auto_type
#     PRIVATE cxx_variadic_templates)

# Depend on a library that we defined in the top-level file
target_link_libraries(yais 
    ${CUDA_LIBRARIES}
    nvidia-ml
    ${_GRPC_GRPCPP_UNSECURE}
    ${_PROTOBUF_LIBPROTOBUF}
    gflags
    glog
    prometheus-cpp::prometheus-cpp
)

target_link_libraries(yais-tensorrt
    yais
    ${TensorRT_LIBRARY}
)

# 'make install' to the correct locations (provided by GNUInstallDirs).
install(TARGETS yais EXPORT yaisConfig
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})  # This is for Windows
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# This makes the project importable from the install directory
# Put config file in per-project dir (name MUST match), can also
# just go into 'cmake'.
install(EXPORT yaisConfig DESTINATION share/yais/cmake)

# This makes the project importable from the build directory
export(TARGETS yais FILE yaisConfig.cmake)

# Every library has unit tests, of course
# add_executable(testlib
#     test/testlib.cpp)
# target_link_libraries(testlib
#     yais)
# add_test(testlib testlib)
