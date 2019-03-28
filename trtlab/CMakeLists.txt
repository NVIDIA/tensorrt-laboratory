cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(trtlab)
include(GNUInstallDirs)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(BUILD_CUDA "Build with NVIDIA CUDA Support" ON)
option(BUILD_TENSORRT "Build with NVIDIA TensorRT Support" ON)
option(BUILD_NVRPC "Build with NVIDIA RPC Support" ON)
option(ENABLE_TESTING "Build tests" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS Off)

#set(CMAKE_DEBUG_POSTFIX "-d")
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(default_build_type "Release")

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
find_package(Threads)

if(ENABLE_TESTING)
  message(STATUS "Building Tests")
  find_package(GTest) #1.8.1 CONFIG REQUIRED)
  find_package(benchmark)
  enable_testing()
endif()

# suppress warnings

add_compile_options(
  $<$<AND:$<STREQUAL:$<COMPILE_LANGUAGE>,CXX>,$<CXX_COMPILER_ID:AppleClang>>:-Wno-deprecated-declarations>
)

add_subdirectory(core)

if(BUILD_CUDA)
  add_subdirectory(cuda)
endif()

if(BUILD_NVRPC)
  add_subdirectory(nvrpc)
endif()

if(BUILD_TENSORRT)
   add_subdirectory(tensorrt)
endif()

add_subdirectory(pybind)

add_subdirectory(trtis)

# install

# include(CMakePackageConfigHelpers)
#
# install(
#   EXPORT ${PROJECT_NAME}-targets
#   NAMESPACE ${PROJECT_NAME}::
#   FILE ${PROJECT_NAME}-targets.cmake
#   DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
# )
#
# configure_package_config_file(
#   "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}-config.cmake.in"
#   ${PROJECT_NAME}-config.cmake
#   INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
#   NO_CHECK_REQUIRED_COMPONENTS_MACRO
#   PATH_VARS CMAKE_INSTALL_INCLUDEDIR
# )
#
# install(
# # FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
#   DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
# )
