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

pybind11_add_module(trtlab 
  infer.cc
  utils.cc
)

target_link_libraries(trtlab
  PUBLIC
    core
    cuda
    nvrpc
    nvrpc-client
    tensorrt
    nv-inference-protos
)

target_include_directories(trtlab
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

##find_program(PYTHON "python)
##
##if (PYTHON)
##    set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")
##    set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
##    set(DEPS        "${CMAKE_CURRENT_SOURCE_DIR}/trtlab/__init__.py")
##    set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/build")
##
##    configure_file(${SETUP_PY_IN} ${SETUP_PY})
##
##    add_custom_command(OUTPUT ${OUTPUT}
##                       COMMAND ${PYTHON} ${SETUP_PY} build
##                       DEPENDS ${DEPS} _trtlab_cpp)
##
##    add_custom_target(target ALL DEPENDS ${OUTPUT})
##
##    install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} bdist_wheel)")
##endif()
