include (ExternalProject)

set (DEPENDENCIES)
set (EXTRA_CMAKE_ARGS)

# trtlab external dependencies
list (APPEND DEPENDENCIES boost gflags glog benchmark googletest cpuaff)

# customize the folder for external projects
# download, source and builds for dependencies 
# will be in <build-dir>/Dependencies
set_property (DIRECTORY PROPERTY EP_BASE Dependencies)

# all dependencies will be installed here
# typical directories: bin, include and lib
set (INSTALL_ROOT ${CMAKE_CURRENT_BINARY_DIR}/local)

# set cmake search paths to pick up installed .cmake files
list(INSERT CMAKE_MODULE_PATH 0 "${INSTALL_ROOT}/lib/cmake")
list(INSERT CMAKE_PREFIX_PATH 0 "${INSTALL_ROOT}/lib/cmake")

# cmake config args forwarded to trtlab
list(APPEND EXTRA_CMAKE_ARGS
  -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}
  -DCMAKE_MODULE_PATH=${CMAKE_MODULE_PATH}
# -DBoost_VERBOSE=ON
  -DBoost_USE_STATIC_LIBS=ON
  -DFIND_GTEST=OFF
  -DFIND_BENCHMARK=OFF
  -DCPUAFF_ROOT=${INSTALL_ROOT}
)

# short-cut to dependencies build path
set (BUILD_ROOT ${CMAKE_CURRENT_BINARY_DIR}/Dependencies/Build)

# Boost
# =====
# - Use static linking to avoid issues with system-wide installations of Boost.
# - Use numa=on to ensure the numa component of fiber gets built
set(BOOST_COMPONENTS "context,fiber,filesystem")
ExternalProject_Add (boost
  URL https://dl.bintray.com/boostorg/release/1.72.0/source/boost_1_72_0.tar.gz
  URL_HASH SHA256=c66e88d5786f2ca4dbebb14e06b566fb642a1a6947ad8cc9091f9f445134143f
  CONFIGURE_COMMAND ./bootstrap.sh --prefix=${INSTALL_ROOT} --with-libraries=${BOOST_COMPONENTS} numa=on
  BUILD_COMMAND ./b2 link=static cxxflags=-fPIC cflags=-fPIC cxxflags="-std=c++14" numa=on 
                     --build-dir=${BUILD_ROOT}/boost --stagedir=${BUILD_ROOT}/boost
  BUILD_IN_SOURCE 1
  INSTALL_COMMAND ./b2 install numa=on
)

# gflags
# ======
# config, build and install to INSTALL_ROOT
ExternalProject_Add(gflags
  GIT_REPOSITORY "https://github.com/gflags/gflags.git"
  GIT_TAG "v2.2.2"
  CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
             -DCMAKE_INSTALL_PREFIX=${INSTALL_ROOT}
             -DBUILD_SHARED_LIBS=ON
             -DBUILD_STATIC_LIBS=ON
             -DBUILD_PACKAGING=OFF
             -DBUILD_TESTING=OFF
             -BUILD_CONFIG_TESTS=OFF
             -DINSTALL_HEADERS=ON
             -DBUILD_gflags_LIB=OFF
             -DBUILD_gflags_nothreads_LIB=ON
             -DGFLAGS_NAMESPACE=google
# INSTALL_COMMAND   ""
)

# glog
# ====
# - link against shared 
# - todo: compile with -DWITH_GFLAGS=OFF and remove gflags dependency
ExternalProject_Add(glog
  DEPENDS gflags
  GIT_REPOSITORY "https://github.com/google/glog"
  GIT_TAG "v0.4.0"
  CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
             -DCMAKE_INSTALL_PREFIX=${INSTALL_ROOT}
             -DBUILD_TESTING=OFF
# INSTALL_COMMAND   ""
)

# google benchmark
# ================
ExternalProject_Add(benchmark
  DEPENDS 
  GIT_REPOSITORY    https://github.com/google/benchmark.git
  GIT_TAG           "v1.5.0"
  BINARY_DIR        "${CMAKE_CURRENT_BINARY_DIR}/Dependencies/Build/benchmark"
  CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
             -DCMAKE_INSTALL_PREFIX=${INSTALL_ROOT}
             -DBENCHMARK_ENABLE_TESTING=OFF
  INSTALL_COMMAND   ""
)

# google test
# ===========
ExternalProject_Add(googletest
  DEPENDS glog gflags
  GIT_REPOSITORY    https://github.com/google/googletest.git
  GIT_TAG           "release-1.10.0"
  BINARY_DIR        "${CMAKE_CURRENT_BINARY_DIR}/Dependencies/Build/googletest"
  CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
             -DCMAKE_INSTALL_PREFIX=${INSTALL_ROOT}
  INSTALL_COMMAND   ""
)

# cpuaff
# ======
ExternalProject_Add(cpuaff
  URL http://dcdillon.github.io/cpuaff/releases/cpuaff-1.0.6.tar.gz
  CONFIGURE_COMMAND ./configure --prefix=${INSTALL_ROOT}
  BUILD_COMMAND     make include
  INSTALL_COMMAND   make install include
  BUILD_IN_SOURCE 1
)


# trtlab
# ======
ExternalProject_Add (trtlab_memory
  DEPENDS ${DEPENDENCIES}
  SOURCE_DIR ${PROJECT_SOURCE_DIR}
  CMAKE_ARGS -DBUILD_DEPENDENCIES=OFF ${EXTRA_CMAKE_ARGS}
  INSTALL_COMMAND ""
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})