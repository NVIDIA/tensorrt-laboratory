include (ExternalProject)

set (DEPENDENCIES)
set (EXTRA_CMAKE_ARGS)

# trtlab external dependencies
list (APPEND DEPENDENCIES boost dlpack gflags glog benchmark googletest cpuaff jemalloc)
list (APPEND DEPENDENCIES grpc-repo protobuf c-ares grpc cub cnpy)

# note on ubuntu 18.04, you need
# apt install libz-dev libssl-dev

# customize the folder for external projects
# download, source and builds for dependencies 
# will be in <build-dir>/Dependencies
set_property (DIRECTORY PROPERTY EP_BASE Dependencies)

# all dependencies will be installed here
# typical directories: bin, include and lib
set (BUILD_ROOT ${CMAKE_CURRENT_BINARY_DIR}/Dependencies/Build)
set (SOURCE_ROOT ${CMAKE_CURRENT_BINARY_DIR}/Dependencies/Source)
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
  -DCPUAFF_ROOT=${INSTALL_ROOT}
  -DJEMALLOC_STATIC_LIBRARIES=${INSTALL_ROOT}/lib/libjemalloc_pic.a
  -DCUB_INCLUDE_DIR=${SOURCE_ROOT}/cub
  -DINSTALL_ROOT=${INSTALL_ROOT}
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

# DLPack
# ======
ExternalProject_Add(dlpack
  GIT_REPOSITORY "https://github.com/dmlc/dlpack.git"
  GIT_TAG "master"
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${INSTALL_ROOT}
)


# gflags
# ======
# config, build and install to INSTALL_ROOT
ExternalProject_Add(gflags
  GIT_REPOSITORY "https://github.com/gflags/gflags.git"
  GIT_TAG "v2.2.2"
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${INSTALL_ROOT}
             -DBUILD_SHARED_LIBS=ON
             -DBUILD_STATIC_LIBS=ON
             -DBUILD_PACKAGING=OFF
             -DBUILD_TESTING=OFF
             -DBUILD_CONFIG_TESTS=OFF
             -DINSTALL_HEADERS=ON
             -DBUILD_gflags_LIB=OFF
             -DBUILD_gflags_nothreads_LIB=ON
             -DGFLAGS_NAMESPACE=google
)

# glog
# ====
# - link against shared 
# - todo: compile with -DWITH_GFLAGS=OFF and remove gflags dependency
ExternalProject_Add(glog
  DEPENDS gflags
  GIT_REPOSITORY "https://github.com/google/glog"
  GIT_TAG "v0.4.0"
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${INSTALL_ROOT}
             -DCMAKE_BUILD_TYPE=Release
             -DBUILD_TESTING=OFF
)

# google benchmark
# ================
ExternalProject_Add(benchmark
  DEPENDS 
  GIT_REPOSITORY    https://github.com/google/benchmark.git
  GIT_TAG           "v1.5.0"
  BINARY_DIR        "${CMAKE_CURRENT_BINARY_DIR}/Dependencies/Build/benchmark"
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${INSTALL_ROOT}
             -DCMAKE_BUILD_TYPE=Release
             -DBENCHMARK_ENABLE_TESTING=OFF
)

# google test
# ===========
ExternalProject_Add(googletest
  DEPENDS glog gflags
  GIT_REPOSITORY    https://github.com/google/googletest.git
  GIT_TAG           "release-1.10.0"
  BINARY_DIR        "${CMAKE_CURRENT_BINARY_DIR}/Dependencies/Build/googletest"
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${INSTALL_ROOT}
             -DCMAKE_BUILD_TYPE=Release
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

# nvidia cub
# ==========
ExternalProject_Add(cub
  GIT_REPOSITORY    https://github.com/NVlabs/cub.git 
  GIT_TAG           "1.8.0"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)

# jemalloc
# ========
ExternalProject_Add(jemalloc
  URL https://github.com/jemalloc/jemalloc/releases/download/5.2.1/jemalloc-5.2.1.tar.bz2
  CONFIGURE_COMMAND ./configure --prefix=${INSTALL_ROOT}
  BUILD_COMMAND     make include
  INSTALL_COMMAND   make install include
  BUILD_IN_SOURCE 1
)

# cnpy - c++ library for reading and writing .npy/.npz files
# ==========================================================
ExternalProject_Add(cnpy
  GIT_REPOSITORY "https://github.com/rogersce/cnpy.git"
  GIT_TAG "master"
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${INSTALL_ROOT}
             -DCMAKE_BUILD_TYPE=Release
             -DBUILD_TESTING=OFF
             -DCMAKE_POSITION_INDEPENDENT_CODEL=ON
)

# grpc-repo
# =========
ExternalProject_Add(grpc-repo
  GIT_REPOSITORY "https://github.com/grpc/grpc.git"
  GIT_TAG "v1.32.0"
  GIT_SUBMODULES "third_party/cares/cares" "third_party/protobuf" "third_party/abseil-cpp" "third_party/re2"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)

#
# Build protobuf project from grpc-repo
#
ExternalProject_Add(absl
  SOURCE_DIR "${SOURCE_ROOT}/grpc-repo/third_party/abseil-cpp"
  DOWNLOAD_COMMAND ""
  CMAKE_CACHE_ARGS
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=TRUE
        -DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_ROOT}
  DEPENDS grpc-repo
)

ExternalProject_Add(re2
  SOURCE_DIR "${SOURCE_ROOT}/grpc-repo/third_party/re2"
  DOWNLOAD_COMMAND ""
  CMAKE_CACHE_ARGS
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=TRUE
        -DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_ROOT}
  DEPENDS grpc-repo
)

ExternalProject_Add(protobuf
  SOURCE_DIR "${SOURCE_ROOT}/grpc-repo/third_party/protobuf/cmake"
  DOWNLOAD_COMMAND ""
  CMAKE_ARGS
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -Dprotobuf_BUILD_TESTS:BOOL=OFF
    -Dprotobuf_WITH_ZLIB:BOOL=OFF
    -Dprotobuf_MSVC_STATIC_RUNTIME:BOOL=OFF
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_ROOT}
  DEPENDS grpc-repo
)

# Location where protobuf-config.cmake will be installed varies by
# platform
if (WIN32)
  set(_FINDPACKAGE_PROTOBUF_CONFIG_DIR "${CMAKE_CURRENT_BINARY_DIR}/protobuf/cmake")
else()
  set(_FINDPACKAGE_PROTOBUF_CONFIG_DIR "${INSTALL_ROOT}/lib/cmake")
endif()

#
# Build c-area project from grpc-repo
#
ExternalProject_Add(c-ares
  SOURCE_DIR "${SOURCE_ROOT}/grpc-repo/third_party/cares/cares"
  DOWNLOAD_COMMAND ""
  CMAKE_ARGS
    -DCARES_SHARED:BOOL=OFF
    -DCARES_STATIC:BOOL=ON
    -DCARES_STATIC_PIC:BOOL=ON
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_ROOT}
  DEPENDS grpc-repo
)


#
# Build GRPC
#
ExternalProject_Add(grpc
  SOURCE_DIR "${SOURCE_ROOT}/grpc-repo"
  DOWNLOAD_COMMAND ""
  CMAKE_ARGS
    -DgRPC_INSTALL:BOOL=ON
    -DgRPC_BUILD_TESTS:BOOL=OFF
    -DgRPC_PROTOBUF_PROVIDER:STRING=package
    -DgRPC_PROTOBUF_PACKAGE_TYPE:STRING=CONFIG
    -DProtobuf_DIR:PATH=${INSTALL_ROOT}/lib/cmake
    -DgRPC_ZLIB_PROVIDER:STRING=package
    -DgRPC_CARES_PROVIDER:STRING=package
    -Dc-ares_DIR:PATH=${INSTALL_ROOT}/lib/cmake
    -DgRPC_SSL_PROVIDER:STRING=package
    -DgRPC_GFLAGS_PROVIDER=package
    -DgRPC_BENCHMARK_PROVIDER=package
    -DgRPC_RE2_PROVIDER:STRING=package
    -Dre2_DIR:STRING=${INSTALL_ROOT}/lib/cmake
    -DgRPC_ABSL_PROVIDER:STRING=package
    -Dabsl_DIR:STRING=${INSTALL_ROOT}/lib/cmake
    ${_CMAKE_ARGS_OPENSSL_ROOT_DIR}
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_ROOT}
  DEPENDS grpc-repo c-ares protobuf re2 absl gflags benchmark
)



# trtlab
# ======
ExternalProject_Add (trtlab
  DEPENDS ${DEPENDENCIES}
  SOURCE_DIR ${PROJECT_SOURCE_DIR}
  CMAKE_ARGS -DBUILD_DEPENDENCIES=OFF ${EXTRA_CMAKE_ARGS}
  INSTALL_COMMAND ""
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})