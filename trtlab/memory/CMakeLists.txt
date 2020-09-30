# MODIFICATION MESSAGE

# Copyright (C) 2015-2016 Jonathan Müller <jonathanmueller.dev@gmail.com>
# This file is subject to the license terms in the LICENSE file
# found in the top-level directory of this distribution.

# root CMakeLists.txt, specifies option and interface library

cmake_minimum_required(VERSION 3.1)

project (TRTLAB_MEMORY)

set(TRTLAB_MEMORY_VERSION_MAJOR 0 CACHE STRING "major version of memory" FORCE)
set(TRTLAB_MEMORY_VERSION_MINOR 1 CACHE STRING "minor version of memory" FORCE)
set(TRTLAB_MEMORY_VERSION_PATCH 0 CACHE STRING "patch version of memory" FORCE)
set(TRTLAB_MEMORY_VERSION "${TRTLAB_MEMORY_VERSION_MAJOR}.${TRTLAB_MEMORY_VERSION_MINOR}.${TRTLAB_MEMORY_VERSION_PATCH}"
                             CACHE STRING "version of memory" FORCE)

# set a debug postfix
set(CMAKE_DEBUG_POSTFIX "-dbg")

if(UNIX)
    include(GNUInstallDirs)

    set(TRTLAB_MEMORY_INC_INSTALL_DIR "${CMAKE_INSTALL_INCLUDEDIR}/trtlab_memory") 
    set(TRTLAB_MEMORY_RUNTIME_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}") 
    set(TRTLAB_MEMORY_LIBRARY_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}")
    set(TRTLAB_MEMORY_ARCHIVE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}")
    set(TRTLAB_MEMORY_FRAMEWORK_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}")

    set(TRTLAB_MEMORY_CMAKE_CONFIG_INSTALL_DIR "${CMAKE_INSTALL_DATADIR}/trtlab_memory/cmake")
    set(TRTLAB_MEMORY_ADDITIONAL_FILES_INSTALL_DIR "${CMAKE_INSTALL_DATADIR}/trtlab_memory")

    set(TRTLAB_MEMORY_RUNTIME_INSTALL_DIR "bin") # for the nodesize_dbg, just ignore version and the like
    set(TRTLAB_MEMORY_INC_INSTALL_DIR "include/trtlab_memory") # header filestree

elseif(WIN32)
    set(TRTLAB_MEMORY_INC_INSTALL_DIR "include/trtlab_memory")
    set(TRTLAB_MEMORY_RUNTIME_INSTALL_DIR   "bin") 
    set(TRTLAB_MEMORY_LIBRARY_INSTALL_DIR   "bin")
    set(TRTLAB_MEMORY_ARCHIVE_INSTALL_DIR   "lib")
    set(TRTLAB_MEMORY_FRAMEWORK_INSTALL_DIR "bin")

    set(TRTLAB_MEMORY_CMAKE_CONFIG_INSTALL_DIR "cmake")
    set(TRTLAB_MEMORY_ADDITIONAL_FILES_INSTALL_DIR "./")
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
else()
	message(FATAL_ERROR "Could not set install folders for this platform!")
endif()

#include(cmake/compatibility.cmake)
include(cmake/configuration.cmake)

set(CMAKE_CXX_STANDARD 17)

# subdirectories
add_subdirectory(src)

if(TRTLAB_MEMORY_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(TRTLAB_MEMORY_BUILD_BENCHMARKS)
    enable_testing()
    add_subdirectory(benchmarks)
endif()

if(TRTLAB_MEMORY_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

if(TRTLAB_MEMORY_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# install readme and license
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE" "${CMAKE_CURRENT_SOURCE_DIR}/README.md" DESTINATION ${TRTLAB_MEMORY_ADDITIONAL_FILES_INSTALL_DIR})

#install(EXPORT trtlab_memoryTargets DESTINATION ${TRTLAB_MEMORY_CMAKE_CONFIG_INSTALL_DIR} FILE trtlab_memory-config.cmake)
