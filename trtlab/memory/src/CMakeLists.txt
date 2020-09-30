# MODIFICATION MESSAGE

# Modification Notes:

# Copyright (C) 2015-2016 Jonathan Müller <jonathanmueller.dev@gmail.com>
# This file is subject to the license terms in the LICENSE file
# found in the top-level directory of this distribution


#find_package(Threads)
find_package(dlpack)
find_package(glog REQUIRED)
find_package(gflags REQUIRED)
if( CMAKE_BUILD_TYPE STREQUAL "Debug" )
  find_package(gflags COMPONENTS nothreads_shared)
  set(trtlab_gflags "gflags_nothreads_shared")
  message(STATUS "gflags: ${trtlab_gflags}")
else()
  find_package(gflags COMPONENTS nothreads_static)
  set(trtlab_gflags "gflags_nothreads_static")
  message(STATUS "gflags: ${trtlab_gflags}")
endif()

set(header_path ${TRTLAB_MEMORY_SOURCE_DIR}/include/trtlab/memory)

set(detail_header
        ${header_path}/detail/assert.h
#       ${header_path}/detail/container_node_sizes.hpp
#       ${header_path}/detail/debug_helpers.hpp
#       ${header_path}/detail/ebo_storage.hpp
#       ${header_path}/detail/free_list.hpp
#       ${header_path}/detail/free_list_array.hpp
#       ${header_path}/detail/lowlevel_allocator.hpp
#       ${header_path}/detail/memory_stack.hpp
#       ${header_path}/detail/small_free_list.hpp
        ${header_path}/detail/utility.h
)

set(header
        ${header_path}/align.h
#       ${header_path}/aligned_allocator.hpp
        ${header_path}/allocator_storage.h
        ${header_path}/allocator_traits.h
        ${header_path}/block_allocators.h
        ${header_path}/block_arena.h
        ${header_path}/block_manager.h
        ${header_path}/block_stack.h
        ${header_path}/config.h
#       ${header_path}/container.hpp
        ${header_path}/debugging.h
#       ${header_path}/default_allocator.hpp
        ${header_path}/deleter.h
        ${header_path}/descriptor.h
        ${header_path}/error.h
        ${header_path}/huge_page_allocator.h
        ${header_path}/literals.h
        ${header_path}/memory_block.h
        ${header_path}/memory_pool.h
#       ${header_path}/fallback_allocator.hpp
#       ${header_path}/malloc_allocator.hpp
#       ${header_path}/heap_allocator.hpp
#       ${header_path}/iteration_allocator.hpp
#       ${header_path}/joint_allocator.hpp
#       ${header_path}/memory_arena.hpp
#       ${header_path}/memory_pool.hpp
#       ${header_path}/memory_pool_collection.hpp
#       ${header_path}/memory_pool_type.hpp
#       ${header_path}/memory_resource_adapter.hpp
#       ${header_path}/memory_stack.hpp
#       ${header_path}/namespace_alias.hpp
#       ${header_path}/new_allocator.hpp
#       ${header_path}/segregator.hpp
#       ${header_path}/smart_ptr.hpp
#       ${header_path}/static_allocator.hpp
#       ${header_path}/std_allocator.hpp
#       ${header_path}/temporary_allocator.hpp
#       ${header_path}/threading.hpp
        ${header_path}/tracking.h
        ${header_path}/trackers.h
#       ${header_path}/virtual_memory.hpp
        ${header_path}/utils.h
        ${CMAKE_CURRENT_BINARY_DIR}/container_node_sizes_impl.h
)

set(src
        align.cc
        block_stack.cc
        detail/page_info.c
#       detail/debug_helpers.cpp
#       detail/assert.cpp
        detail/block_list.cc
        detail/free_list.cc
#       detail/free_list_array.cpp
#       detail/free_list_utils.hpp
#       detail/ilog2.hpp
#       detail/small_free_list.cpp
#       debugging.cpp
        descriptor.cc
        error.cc
#       heap_allocator.cpp
#       iteration_allocator.cpp
#       malloc_allocator.cpp
#       memory_pool.cpp
#       memory_pool_collection.cpp
#       memory_stack.cpp
        memory_type.cc
#       new_allocator.cpp
#       static_allocator.cpp
#       temporary_allocator.cpp
        trackers.cc
        utils.cc
#       virtual_memory.cpp)
)

# configure config file
configure_file("config.h.in" "${CMAKE_CURRENT_BINARY_DIR}/config_impl.h")

# generate container_node_sizes.hpp
if(TRTLAB_MEMORY_BUILD_TOOLS AND (NOT CMAKE_CROSSCOMPILING))
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/container_node_sizes_impl.h
            COMMAND memory_node_size_debugger --code --alignof "alignof(T)" ${CMAKE_CURRENT_BINARY_DIR}/container_node_sizes_impl.h
            DEPENDS memory_node_size_debugger
            VERBATIM)
else()
    message(WARNING "cannot generate container_node_sizes_impl.h, node size information will be unavailable")
    file(WRITE  ${CMAKE_CURRENT_BINARY_DIR}/container_node_sizes_impl.h "#define TRTLAB_MEMORY_NO_NODE_SIZE")
endif()

# trtlab_memory library
add_library(trtlab_memory ${detail_header} ${header} ${src})

target_include_directories(trtlab_memory PUBLIC $<BUILD_INTERFACE:${TRTLAB_MEMORY_SOURCE_DIR}/include/> # for client in subdirectory
                                                $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}> # for generated files in build mode
                                                $<INSTALL_INTERFACE:${TRTLAB_MEMORY_INC_INSTALL_DIR}> # for client in install mode
                                         PRIVATE ${header_path})
target_compile_definitions(trtlab_memory PUBLIC
                           TRTLAB_MEMORY=1
                           TRTLAB_MEMORY_VERSION_MAJOR=${TRTLAB_MEMORY_VERSION_MAJOR}
                           TRTLAB_MEMORY_VERSION_MINOR=${TRTLAB_MEMORY_VERSION_MINOR}
                           TRTLAB_MEMORY_VERSION_PATCH=${TRTLAB_MEMORY_VERSION_PATCH})

target_link_libraries(trtlab_memory
  PUBLIC
#   Threads::Threads
    glog::glog
    ${trtlab_gflags}
    dlpack::dlpack
)

#set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-pg")

set_target_properties(trtlab_memory PROPERTIES
                      OUTPUT_NAME "trtlab_memory-${TRTLAB_MEMORY_VERSION}"
                      POSITION_INDEPENDENT_CODE ON)

install(TARGETS trtlab_memory EXPORT trtlab_memoryTargets
        RUNTIME       DESTINATION ${TRTLAB_MEMORY_RUNTIME_INSTALL_DIR}
        LIBRARY       DESTINATION ${TRTLAB_MEMORY_LIBRARY_INSTALL_DIR}
        ARCHIVE       DESTINATION ${TRTLAB_MEMORY_ARCHIVE_INSTALL_DIR}
        FRAMEWORK     DESTINATION ${TRTLAB_MEMORY_FRAMEWORK_INSTALL_DIR})

# Write/install version file
include(CMakePackageConfigHelpers)
set(version_file "${CMAKE_CURRENT_BINARY_DIR}/cmake/trtlab_memory-config-version.cmake")
write_basic_package_version_file(${version_file}
                                 VERSION ${TRTLAB_MEMORY_VERSION}
                                 COMPATIBILITY AnyNewerVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/config_impl.hpp DESTINATION ${TRTLAB_MEMORY_INC_INSTALL_DIR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/container_node_sizes_impl.hpp DESTINATION ${TRTLAB_MEMORY_INC_INSTALL_DIR}/foonathan/memory/detail)
install(FILES ${header}                                   DESTINATION ${TRTLAB_MEMORY_INC_INSTALL_DIR}/foonathan/memory)
install(FILES ${detail_header}                            DESTINATION ${TRTLAB_MEMORY_INC_INSTALL_DIR}/foonathan/memory/detail)
install(FILES ${version_file}                             DESTINATION ${TRTLAB_MEMORY_CMAKE_CONFIG_INSTALL_DIR})