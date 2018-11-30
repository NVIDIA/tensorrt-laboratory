find_package(glog 0.3.5 REQUIRED)

add_library(nvcxx
  src/affinity.cc
  src/memory.cc
  src/thread_pool.cc
  src/utils.cc
)

add_library(${PROJECT_NAME}::nvcxx ALIAS nvcxx)

target_link_libraries(nvcxx
  PRIVATE
    Threads::Threads
    $<$<AND:$<BOOL:UNIX>,$<NOT:$<BOOL:APPLE>>>:rt>
    glog    
)

target_include_directories(nvcxx
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

set_target_properties(nvcxx PROPERTIES OUTPUT_NAME ${PROJECT_NAME})

install(
  TARGETS nvcxx
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

