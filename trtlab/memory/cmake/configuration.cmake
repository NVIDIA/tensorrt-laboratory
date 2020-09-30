# MODIFICATION MESSAGE

# Copyright (C) 2015-2016 Jonathan Müller <jonathanmueller.dev@gmail.com>
# This file is subject to the license terms in the LICENSE file
# found in the top-level directory of this distribution.

# defines configuration options
# note: only include it in memory's top-level CMakeLists.txt, after compatibility.cmake

# what to build
# examples/tests if toplevel directory (i.e. direct build, not as subdirectory) and hosted
# tools if hosted

option(TRTLAB_MEMORY_BUILD_TOOLS "whether or not to build the tools" ON)
option(TRTLAB_MEMORY_BUILD_TESTS "whether or not to build the tests" ON)
option(TRTLAB_MEMORY_BUILD_BENCHMARKS "whether or not to build the tools" ON)
option(TRTLAB_MEMORY_BUILD_EXAMPLES "whether or not to build the examples" OFF)

#SET(CPUAFF_ROOT "/usr/local" CACHE STRING "Location of cpuaff header files")
#add_library(cpuaff INTERFACE)
#target_include_directories(cpuaff INTERFACE "${CPUAFF_ROOT}/include")

set(TRTLAB_MEMORY_DEBUG_ASSERT OFF CACHE BOOL "" FORCE)
set(TRTLAB_MEMORY_DEBUG_FILL OFF CACHE BOOL "" FORCE)
set(TRTLAB_MEMORY_DEBUG_FENCE 0 CACHE STRING "" FORCE)
set(TRTLAB_MEMORY_DEBUG_LEAK_CHECK OFF CACHE BOOL "" FORCE)
set(TRTLAB_MEMORY_DEBUG_POINTER_CHECK OFF CACHE BOOL "" FORCE)
set(TRTLAB_MEMORY_DEBUG_DOUBLE_DEALLOC_CHECK OFF CACHE BOOL "" FORCE)

# most of the debugging aspects of foonathan/memory have been disabled
# debug options, pre-set by build type
# if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
#     # disable force to allow external CMakeLists.txt to override the defaults
#     set(TRTLAB_MEMORY_DEBUG_ASSERT ON CACHE BOOL "")
#     set(TRTLAB_MEMORY_DEBUG_FILL ON CACHE BOOL "")
#     set(TRTLAB_MEMORY_DEBUG_FENCE 8 CACHE STRING "")
#     set(TRTLAB_MEMORY_DEBUG_LEAK_CHECK ON CACHE BOOL "")
#     set(TRTLAB_MEMORY_DEBUG_POINTER_CHECK ON CACHE BOOL "")
#     set(TRTLAB_MEMORY_DEBUG_DOUBLE_DEALLOC_CHECK ON CACHE BOOL "")
# elseif("${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
#     set(TRTLAB_MEMORY_DEBUG_ASSERT OFF CACHE BOOL "" FORCE)
#     set(TRTLAB_MEMORY_DEBUG_FILL ON CACHE BOOL "" FORCE)
#     set(TRTLAB_MEMORY_DEBUG_FENCE 0 CACHE STRING "" FORCE)
#     set(TRTLAB_MEMORY_DEBUG_LEAK_CHECK ON CACHE BOOL "" FORCE)
#     set(TRTLAB_MEMORY_DEBUG_POINTER_CHECK ON CACHE BOOL "" FORCE)
#     set(TRTLAB_MEMORY_DEBUG_DOUBLE_DEALLOC_CHECK OFF CACHE BOOL "" FORCE)
# elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
#     set(TRTLAB_MEMORY_DEBUG_ASSERT OFF CACHE BOOL "" FORCE)
#     set(TRTLAB_MEMORY_DEBUG_FILL OFF CACHE BOOL "" FORCE)
#     set(TRTLAB_MEMORY_DEBUG_FENCE 0 CACHE STRING "" FORCE)
#     set(TRTLAB_MEMORY_DEBUG_LEAK_CHECK OFF CACHE BOOL "" FORCE)
#     set(TRTLAB_MEMORY_DEBUG_POINTER_CHECK OFF CACHE BOOL "" FORCE)
#     set(TRTLAB_MEMORY_DEBUG_DOUBLE_DEALLOC_CHECK OFF CACHE BOOL "" FORCE)
# else()
#     option(TRTLAB_MEMORY_DEBUG_ASSERT "whether or not internal assertions (like the macro assert) are enabled" OFF)
#     option(TRTLAB_MEMORY_DEBUG_FILL   "whether or not the (de-)allocated memory will be pre-filled" OFF)
#     set(TRTLAB_MEMORY_DEBUG_FENCE 0 CACHE STRING "the amount of memory used as fence to help catching overflow errors" )
#     option(TRTLAB_MEMORY_DEBUG_LEAK_CHECK "whether or not leak checking is active" OFF)
#     option(TRTLAB_MEMORY_DEBUG_POINTER_CHECK "whether or not pointer checking on deallocation is active" OFF)
#     option(TRTLAB_MEMORY_DEBUG_DOUBLE_DEALLOC_CHECK "whether or not the (sometimes expensive) check for double deallocation is active" OFF)
# endif()

# other options
option(TRTLAB_MEMORY_EXTERN_TEMPLATE
    "whether or not common template instantiations are already provided by the library" ON)
option(TRTLAB_MEMORY_CHECK_ALLOCATION_SIZE
        "whether or not the size of the allocation will be checked" ON)
option(TRTLAB_MEMORY_THREAD_SAFE_REFERENCE
    "whether or not allocator_reference is thread safe by default" ON)

set(TRTLAB_MEMORY_DEFAULT_ALLOCATOR heap_allocator CACHE STRING
    "the default implementation allocator for higher-level ones")
set(TRTLAB_MEMORY_MEMORY_RESOURCE_HEADER "<foonathan/pmr.hpp>" CACHE STRING
    "the header of the memory_resource class used")
set(TRTLAB_MEMORY_MEMORY_RESOURCE foonathan_comp::memory_resource CACHE STRING
    "the memory_resource class used")
set(TRTLAB_MEMORY_TEMPORARY_STACK_MODE 2 CACHE STRING
     "set to 0 to disable the per-thread stack completely, to 1 to disable the nitfy counter and to 2 to enable everything")
