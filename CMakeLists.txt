# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

option (BUILD_DEPENDENCIES "Whether or not a superbuild should be invoked" ON)
option (BUILD_MEMORY "Whether or not build trtlab/memory" ON)
option (BUILD_CORE "Whether or not build trtlab/core" ON)
option (BUILD_CUDA "Whether or not build trtlab/cuda" ON)
option (BUILD_NVRPC "Whether or not build trtlab/nvrpc" ON)
option (BUILD_TENSORRT "Whether or not build trtlab/tensorrt" ON)
option (BUILD_PYTHON "Whether or not build trtlab/pybind" OFF)
option (BUILD_EXAMPLES "Whether or not to build trtlab examples", OFF)

if (BUILD_DEPENDENCIES)
  project (trtlab_dependencies NONE)
  include (cmake/dependencies.cmake)
  return() # stop processing this file further
else()
  project (trtlab)
endif()

# CMake path
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/cmake")

# trtlab 
add_subdirectory(trtlab)

if (BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
