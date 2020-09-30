include(GNUInstallDirs)
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
find_package(Threads)
find_package(cpuaff)
find_package(glog 0.3.5 REQUIRED)
find_package(dlpack)
find_package(Boost REQUIRED COMPONENTS fiber_numa fiber context)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# when linking against static glog one also needs to link against gflags
# which version of gflags is depended on build type
find_package(gflags REQUIRED)
if( CMAKE_BUILD_TYPE STREQUAL "Debug" )
  find_package(gflags COMPONENTS nothreads_shared)
  set(trtlab_gflags "gflags_nothreads_shared")
  message(STATUS "gflags: ${trtlab_gflags}")
  message(STATUS "build_type: ${CMAKE_BUILD_TYPE}")
else()
  find_package(gflags COMPONENTS nothreads_static)
  set(trtlab_gflags "gflags_nothreads_static")
  message(STATUS "gflags: ${trtlab_gflags}")
  message(STATUS "build_type: ${CMAKE_BUILD_TYPE}")
endif()

set(header_path ${CMAKE_CURRENT_SOURCE_DIR}/include/trtlab/core)

add_library(core
# src/types.cc
  src/affinity.cc
  src/utils.cc
  src/cyclic_buffer.cc
  src/cyclic_windowed_buffer.cc
)

add_library(${PROJECT_NAME}::core ALIAS core)

message(STATUS "jemalloc: ${JEMALLOC_STATIC_LIBRARIES}")

target_link_libraries(core
  PUBLIC
    trtlab_memory

    dlpack::dlpack
    Threads::Threads
    $<$<AND:$<BOOL:UNIX>,$<NOT:$<BOOL:APPLE>>>:rt>
    glog::glog
    ${trtlab_gflags}
    Boost::fiber
    Boost::fiber_numa
    Boost::context
    ${JEMALLOC_STATIC_LIBRARIES}
    dl
)

target_include_directories(core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE
    ${header_path}
)

set_target_properties(core PROPERTIES OUTPUT_NAME ${PROJECT_NAME}_core)

install(
  TARGETS core
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
  add_subdirectory(tests)
#endif()

#if(benchmark_FOUND)
  add_subdirectory(benchmarks)
#endif()

