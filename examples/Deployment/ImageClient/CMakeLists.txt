set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf CONFIG REQUIRED)
message(STATUS "Using protobuf ${protobuf_VERSION}")
set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)

INCLUDE(GRPCGenerateCPP)

PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS 
    api.proto
)

PROTOBUF_GENERATE_GRPC_CPP(PROTO_GRPC_SRCS PROTO_GRPC_HDRS 
    api.proto
)

add_library(deploy-image-client-protos
    ${PROTO_SRCS}
    ${PROTO_GRPC_SRCS}
)

target_link_libraries(deploy-image-client-protos
  PUBLIC
  ${_PROTOBUF_LIBPROTOBUF}
)

target_include_directories(deploy-image-client-protos PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
)

pybind11_add_module(deploy_image_client
  client.cc
)

target_link_libraries(deploy_image_client
PUBLIC
  nvrpc-client
  deploy-image-client-protos
)
