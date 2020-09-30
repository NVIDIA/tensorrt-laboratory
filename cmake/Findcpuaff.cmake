# This module defines the following variables:
#
# ::
#
#   CPUAFF_INCLUDE_DIRS
#   CPUAFF_FOUND
#
# ::
#
# Hints
# ^^^^^
# A user may set ``CPUAFF_ROOT`` to an installation root to tell this module where to look.
#
set(CPUAFF_FOUND FALSE)
set(_CPUAFF_SEARCHES)

if(CPUAFF_ROOT)
  set(_CPUAFF_SEARCH_ROOT PATHS ${CPUAFF_ROOT} NO_DEFAULT_PATH)
  list(APPEND _CPUAFF_SEARCHES _CPUAFF_SEARCH_ROOT)
else()
  list(APPEND _CPUAFF_SEARCHES "/usr")
  list(APPEND _CPUAFF_SEARCHES "/usr/local")
endif()

# Include dir
foreach(search ${_CPUAFF_SEARCHES})
  find_path(
    CPUAFF_INCLUDE_DIR 
    NAMES cpuaff/cpuaff.hpp
    PATHS ${CPUAFF_ROOT}
    PATH_SUFFIXES include)
  message(STATUS "cpuaff: ${CPUAFF_INCLUDE_DIR}")
endforeach()

mark_as_advanced(CPUAFF_INCLUDE_DIR)

if(CPUAFF_INCLUDE_DIR AND EXISTS "${CPUAFF_INCLUDE_DIR}/cpuaff/cpuaff.hpp")
    set(CPUAFF_FOUND True)

    add_library(cpuaff INTERFACE)
    target_include_directories(cpuaff INTERFACE ${CPUAFF_INCLUDE_DIR})
endif()
