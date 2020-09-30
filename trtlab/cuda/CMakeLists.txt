# Find CUDA
find_package(CUDA REQUIRED)
message(STATUS "CUDA IncludeDir ${CUDA_INCLUDE_DIRS}")

set(header_path ${CMAKE_CURRENT_SOURCE_DIR}/include/trtlab/cuda)

add_library(cuda
# src/copy.cc
# src/cuda_device.cc
# src/cuda_managed.cc
# src/cuda_pinned_host.cc
  src/device_info.cc
  src/device_guard.cc
# src/device_memory.cc
  src/cuda_allocators.cc
)

add_library(${PROJECT_NAME}::cuda ALIAS cuda)

target_link_libraries(cuda
    ${PROJECT_NAME}::core
    ${CUDA_LIBRARIES}
    nvidia-ml
)

target_include_directories(cuda
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${CUDA_INCLUDE_DIRS}
  PRIVATE
    ${header_path}
)

set_target_properties(cuda PROPERTIES OUTPUT_NAME ${PROJECT_NAME}-cuda)

install(
  TARGETS cuda
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

  add_subdirectory(tests)

# add_subdirectory(benchmarks)
