# Find CUDA
find_package(CUDA REQUIRED)
message(STATUS "CUDA IncludeDir ${CUDA_INCLUDE_DIRS}")

add_library(nvcxx-cuda
  src/device_info.cc
  src/memory.cc
)

add_library(${PROJECT_NAME}::nvcxx-cuda ALIAS nvcxx-cuda)

target_link_libraries(nvcxx-cuda
  PRIVATE
    ${PROJECT_NAME}::nvcxx
    ${CUDA_LIBRARIES}
    nvidia-ml
)

target_include_directories(nvcxx-cuda
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${CUDA_INCLUDE_DIRS}
)

set_target_properties(nvcxx-cuda PROPERTIES OUTPUT_NAME ${PROJECT_NAME})

install(
  TARGETS nvcxx-cuda
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

if(benchmark_FOUND)
  add_subdirectory(benchmarks)
endif()
