#project(nvrpc VERSION 0.1 LANGUAGES CXX)

# Build options
#set(CMAKE_CXX_STANDARD 17)
#set(default_build_type "Release")
#set(CMAKE_DEBUG_POSTFIX "-d")
#set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

# Find Protobuf installation
# Looks for protobuf-config.cmake file installed by Protobuf's cmake installation.
set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf CONFIG REQUIRED)
message(STATUS "Using protobuf ${protobuf_VERSION}")
set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)

# Find gRPC installation
# Looks for gRPCConfig.cmake file installed by gRPC's cmake installation.
find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")

set(_GRPC_GRPCPP_UNSECURE gRPC::grpc++_unsecure)
set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)

add_library(nvrpc
  src/server.cc
  src/executor.cc
)

add_library(${PROJECT_NAME}::nvrpc ALIAS nvrpc)

target_link_libraries(nvrpc
  PRIVATE
    ${PROJECT_NAME}::core
)

target_include_directories(nvrpc
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

set_target_properties(nvrpc PROPERTIES OUTPUT_NAME nvrpc)

install(
  TARGETS nvrpc
  EXPORT ${PROJECT_NAME}-targets
  RUNTIME DESTINATION  ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION  ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION  ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
  DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

#if(ENABLE_TESTING)
#  add_subdirectory(tests)
#endif()

#if(benchmark_FOUND)
#  add_subdirectory(benchmarks)
#endif()

