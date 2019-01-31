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

find_package(Flatbuffers)

# This module defines
#  FLATBUFFERS_INCLUDE_DIR, directory containing headers
#  FLATBUFFERS_LIBS, directory containing flatbuffers libraries
#  FLATBUFFERS_STATIC_LIB, path to libflatbuffers.a
#  FLATBUFFERS_FOUND, whether flatbuffers has been found


add_library(example-fbs 
  example.grpc.fb.cc
)

target_link_libraries(example-fbs PUBLIC
  flatbuffers
)

target_include_directories(example-fbs PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
)


add_executable(fb-server.x
    server.cc)

target_link_libraries(fb-server.x
    nvrpc
    example-fbs
    gflags
)

add_executable(fb-client.x
    client.cc)

target_link_libraries(fb-client.x
    nvrpc
    example-fbs
    gflags
)
