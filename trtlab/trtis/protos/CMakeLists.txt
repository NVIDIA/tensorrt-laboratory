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

set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf CONFIG REQUIRED)
message(STATUS "Using protobuf ${protobuf_VERSION}")
set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)

INCLUDE(GRPCGenerateCPPLikeBazel)
INCLUDE(ProtobufGenerateCPPLikeBazel)

PROTOBUF_GENERATE_CPP_LIKE_BAZEL(
  PROTO_SRCS PROTO_HDRS
    api.proto
    grpc_service.proto
    model_config.proto
    request_status.proto
    server_status.proto
)

PROTOBUF_GENERATE_GRPC_CPP_LIKE_BAZEL(
  PROTO_GRPC_SRCS PROTO_GRPC_HDRS
    grpc_service.proto
)

add_library(trtis-protos
    ${PROTO_SRCS}
    ${PROTO_GRPC_SRCS}
)

target_link_libraries(trtis-protos
  PUBLIC
    ${_PROTOBUF_LIBPROTOBUF}
)

target_include_directories(trtis-protos
  PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
)
