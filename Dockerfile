# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

RUN apt update && apt install -y --no-install-recommends build-essential autoconf libtool git \
        curl wget pkg-config sudo ca-certificates vim-tiny automake libssl-dev bc python3-pip \
 && apt remove -y cmake \
 && apt remove -y libgflags-dev libgflags2v5 \
 && apt -y autoremove \
 && rm -rf /var/lib/apt/lists/* 

env LC_ALL=C.UTF-8
env LANG=C.UTF-8

RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install --upgrade setuptools \
 && python3 -m pip install cmake click jinja2

# install latest cmake 
# WORKDIR /tmp
# RUN version=3.11 \
#  && build=1 \
#  && wget https://cmake.org/files/v$version/cmake-$version.$build.tar.gz \
#  && tar xzf cmake-$version.$build.tar.gz \
#  && cd cmake-$version.$build/ \
#  && ./bootstrap \
#  && make -j \
#  && make install \
#  && cd /tmp && rm -rf cmake-$version.$build

# install gflags
# -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON -DBUILD_gflags_LIB=ON .. \
RUN git clone -b v2.2.1 https://github.com/gflags/gflags.git \
 && cd gflags \
 && mkdir build && cd build \
 && cmake -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON -DBUILD_gflags_LIB=ON .. \ 
 && make -j \
 && make install \
 && cd /tmp && rm -rf gflags

# install glog
RUN git clone -b v0.3.5 https://github.com/google/glog.git \
 && cd glog \
 && cmake -H. -Bbuild -G "Unix Makefiles" -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON \
 && cmake --build build \
 && cmake --build build --target install \
 && cd /tmp && rm -rf glog

# cmake build per: https://github.com/grpc/grpc/blob/master/test/distrib/cpp/run_distrib_test_cmake.sh
# -DCMAKE_INSTALL_PREFIX:PATH=/usr to overwrite the version of protobuf installed in the TensorRT base image
WORKDIR /source
RUN git clone -b v1.11.0 https://github.com/grpc/grpc \
 && cd grpc \
 && git submodule update --init \
 && cd third_party/cares/cares \ 
 && mkdir -p cmake/build \
 && cd cmake/build \
 && cmake -DCMAKE_BUILD_TYPE=Release ../.. \
 && make -j20 install \
 && cd ../../../../.. \
 && rm -rf third_party/cares/cares \
 && cd third_party/zlib && mkdir -p cmake/build && cd cmake/build \
 && cmake -DCMAKE_BUILD_TYPE=Release ../.. \
 && make -j20 install \
 && cd ../../../.. \ 
 && rm -rf third_party/zlib \
 && cd third_party/protobuf && mkdir -p cmake/build && cd cmake/build \
 && cmake -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX:PATH=/usr -DCMAKE_BUILD_TYPE=Release .. \
 && make -j20 install \
 && cd ../../../.. \ 
 && rm -rf third_party/protobuf \
 && cd /source/grpc \
 && mkdir -p cmake/build \
 && cd cmake/build \
 && cmake -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF -DgRPC_PROTOBUF_PROVIDER=package -DgRPC_ZLIB_PROVIDER=package \
          -DgRPC_CARES_PROVIDER=package -DCMAKE_BUILD_TYPE=Release -DgRPC_SSL_PROVIDER=package -DgRPC_GFLAGS_PROVIDER=package ../.. \
 && make -j20 install \
 && cd /source && rm -rf grpc

COPY cmake/*.cmake /usr/local/share/cmake-3.11/Modules/

WORKDIR /tmp
RUN wget https://raw.githubusercontent.com/vishnubob/wait-for-it/master/wait-for-it.sh \
 && chmod +x wait-for-it.sh \
 && mv wait-for-it.sh /usr/local/bin/

RUN git clone -b v1.0.6 https://github.com/dcdillon/cpuaff \
 && cd cpuaff \
 && ls -lF \
 && ./bootstrap.sh \
 && ./configure \
 && make \
 && make install \
 && cd ../ \
 &&  rm -rf cpuaff

RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install --upgrade setuptools \
 && python3 -m pip install click jinja2

env LC_ALL=C.UTF-8
env LANG=C.UTF-8

COPY --from=envoyproxy/envoy:v1.6.0 /usr/local/bin/envoy /usr/local/bin/envoy

#WORKDIR /work
#COPY . .
#RUN mkdir build && cd build \
# && cmake -DCMAKE_BUILD_TYPE=Release ../src ||: \
# && cmake -DCMAKE_BUILD_TYPE=Release ../src \
# && make -j
#
#CMD ["./demo.sh"]

