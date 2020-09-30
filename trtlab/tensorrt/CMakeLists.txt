# find tensorrt
find_package(TensorRT REQUIRED 4.0.1)
message(STATUS "TensorRT IncludeDir ${TensorRT_INCLUDE_DIRS}")
message(STATUS "TensorRT Version ${TensorRT_VERSION_STRING}")

add_library(tensorrt
  src/allocator.cc
# src/bindings.cc
# src/buffers.cc
  src/execution_context.cc
# src/inference_manager.cc
# src/infer_bench.cc
  src/model.cc
  src/runtime.cc
  src/utils.cc
  src/workspace.cc
)

add_library(${PROJECT_NAME}::tensorrt ALIAS tensorrt)

target_link_libraries(tensorrt
    ${PROJECT_NAME}::core
    ${PROJECT_NAME}::cuda
    ${TensorRT_LIBRARY}
)

target_include_directories(tensorrt
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${TensorRT_INCLUDE_DIRS}
)

set_target_properties(tensorrt PROPERTIES OUTPUT_NAME ${PROJECT_NAME}_tensorrt)

install(
  TARGETS tensorrt
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

if(ENABLE_TESTING)
  add_subdirectory(tests)
endif()

#if(benchmark_FOUND)
#  add_subdirectory(benchmarks)
#endif()

