# Copyright (C) 2015-2016 Jonathan Müller <jonathanmueller.dev@gmail.com>
# This file is subject to the license terms in the LICENSE file
# found in the top-level directory of this distribution.

# builds tools

add_executable(memory_node_size_debugger test_types.hpp node_size_debugger.hpp node_size_debugger.cpp)
#_use_comp(memory_node_size_debugger)
# comp_target_features(memory_node_size_debugger PUBLIC CPP11)
if (MSVC)
    target_compile_options(memory_node_size_debugger PRIVATE "/bigobj")
endif()
target_compile_definitions(memory_node_size_debugger PUBLIC
                           VERSION="${TRTLAB_MEMORY_VERSION_MAJOR}.${TRTLAB_MEMORY_VERSION_MINOR}")
set_target_properties(memory_node_size_debugger PROPERTIES OUTPUT_NAME nodesize_dbg)

install(TARGETS memory_node_size_debugger EXPORT memoryTargets RUNTIME DESTINATION ${TRTLAB_MEMORY_RUNTIME_INSTALL_DIR})
