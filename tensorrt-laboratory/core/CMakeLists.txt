find_package(glog 0.3.5 REQUIRED)
find_package(dlpack REQUIRED)
find_library(TCMALLOC_LIB NAMES libtcmalloc_minimal.a tcmalloc_minimal)

add_library(core
  src/types.cc
  src/affinity.cc
  src/memory/copy.cc
  src/memory/memory.cc
  src/memory/host_memory.cc
  src/memory/malloc.cc
  src/memory/system_v.cc
  src/utils.cc
)

add_library(${PROJECT_NAME}::core ALIAS core)

target_link_libraries(core
  PRIVATE
    dlpack::dlpack
    Threads::Threads
    $<$<AND:$<BOOL:UNIX>,$<NOT:$<BOOL:APPLE>>>:rt>
    glog
)

target_include_directories(core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

set_target_properties(core PROPERTIES OUTPUT_NAME ${PROJECT_NAME}-core)

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

if(ENABLE_TESTING)
  add_subdirectory(tests)
endif()

if(benchmark_FOUND)
  add_subdirectory(benchmarks)
endif()

