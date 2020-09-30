# MODIFICATION MESSAGE

# Copyright (C) 2015-2016 Jonathan Müller <jonathanmueller.dev@gmail.com>
# This file is subject to the license terms in the LICENSE file
# found in the top-level directory of this distribution.

# builds test

find_package(GTest)
enable_testing()

set(tests
#   test_simple.cc
    test_memory.cc
    test_main.cc
)

add_executable(test_memory ${tests})
target_link_libraries(test_memory trtlab_memory GTest::gtest)
target_include_directories(test_memory PRIVATE
                           ${CMAKE_CURRENT_BINARY_DIR}
                           ${TRTLAB_MEMORY_SOURCE_DIR}/include/trtlab/memory)

add_test(NAME memory COMMAND test_memory)
