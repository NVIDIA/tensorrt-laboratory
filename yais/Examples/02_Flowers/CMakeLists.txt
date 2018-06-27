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

INCLUDE(GRPCGenerateCPP)
PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS 
    protos/flowers.proto
    protos/dataset.proto)

PROTOBUF_GENERATE_GRPC_CPP(PROTO_GRPC_SRCS PROTO_GRPC_HDRS 
    protos/flowers.proto
    protos/dataset.proto)

#add_custom_target(generate-protos DEPENDS ${PROTO_GPRC_SRCS} ${PROTO_GPRC_HDRS})

include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${TensorRT_INCLUDE_DIRS})

add_executable(flowers-server.x
    src/flowers-server.cc
    ${PROTO_SRCS}
    ${PROTO_GRPC_SRCS})

#add_dependencies(flowers-server.x generate-protos)

target_link_libraries(flowers-server.x
    yais
)

add_executable(flowers-client.x
    src/client.cpp
    ${PROTO_SRCS}
    ${PROTO_GRPC_SRCS})

target_link_libraries(flowers-client.x
    yais
)

add_executable(flowers-async-client.x
    src/async-client.cc
    ${PROTO_SRCS}
    ${PROTO_GRPC_SRCS})

target_link_libraries(flowers-async-client.x
    yais
)

add_executable(siege.x
    src/siege.cc
    ${PROTO_SRCS}
    ${PROTO_GRPC_SRCS})

target_link_libraries(siege.x
    yais
)