# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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
        google-perftools \
 && apt remove -y cmake \
 && apt remove -y libgflags-dev libgflags2v5 \
 && apt remove -y libprotobuf-dev \
 && apt -y autoremove \
 && rm -rf /var/lib/apt/lists/* 

env LC_ALL=C.UTF-8
env LANG=C.UTF-8

RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install --upgrade setuptools \
 && python3 -m pip install cmake==3.11.0

# install gflags
# -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON -DBUILD_gflags_LIB=ON .. \
RUN git clone -b v2.2.2 https://github.com/gflags/gflags.git \
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

# grpc 1.17.x is blocked by: https://github.com/google/flatbuffers/pull/5100
# cmake build per: https://github.com/grpc/grpc/blob/master/test/distrib/cpp/run_distrib_test_cmake.sh
# -DCMAKE_INSTALL_PREFIX:PATH=/usr to overwrite the version of protobuf installed in the TensorRT base image
WORKDIR /source
RUN git clone -b v1.16.1 https://github.com/grpc/grpc \
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
 && cmake -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX:PATH=/usr -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE -DBUILD_SHARED_LIBRARIES=ON .. \
 && make -j20 install \
 && cd ../../../.. \ 
 && rm -rf third_party/protobuf \
 && cd /source/grpc \
 && mkdir -p cmake/build \
 && cd cmake/build \
 && cmake -DBUILD_SHARED_LIBRARIES=ON -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF -DgRPC_PROTOBUF_PROVIDER=package -DgRPC_ZLIB_PROVIDER=package \
          -DgRPC_CARES_PROVIDER=package -DCMAKE_BUILD_TYPE=Release -DgRPC_SSL_PROVIDER=package -DgRPC_GFLAGS_PROVIDER=package ../.. \
 && make -j20 install \
 && cd /source && rm -rf grpc

RUN git clone -b v1.0.6 https://github.com/dcdillon/cpuaff \
 && cd cpuaff \
 && ls -lF \
 && ./bootstrap.sh \
 && ./configure \
 && make \
 && make install \
 && cd ../ \
 &&  rm -rf cpuaff

RUN git clone -b v1.4.1 https://github.com/google/benchmark.git \
 && cd benchmark \
 && git clone -b release-1.8.0 https://github.com/google/googletest.git \
 && mkdir build && cd build \
 && cmake .. -DCMAKE_BUILD_TYPE=RELEASE \
 && make -j && make install \
 && cd /tmp && rm -rf benchmark

RUN git clone https://github.com/jupp0r/prometheus-cpp.git \
 && cd prometheus-cpp \
 && git checkout -b yais e7709f7e3b71bc5b1ac147971c87f2f0ae9ea358 \
 && git submodule update --init --recursive \
 && mkdir build && cd build \
 && cmake -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE .. \
 && make -j \
 && make install \
 && cd .. && rm -rf prometheus-cpp

RUN git clone https://github.com/cameron314/concurrentqueue.git \
 && cd concurrentqueue \
 && git checkout 8f65a87 \
 && mkdir -p /usr/local/include/moodycamel \
 && cp *.h /usr/local/include/moodycamel/ \
 && cd .. && rm -rf concurrentqueue 

# install flatbuffers
RUN git clone -b v1.10.0 https://github.com/google/flatbuffers.git \
 && cd flatbuffers \
 && mkdir build2 && cd build2 \
 && cmake -DCMAKE_BUILD_TYPE=Release .. \
 && make -j$(nproc) install \
 && cd .. && rm -rf flatbuffers

WORKDIR /tmp
RUN wget https://raw.githubusercontent.com/vishnubob/wait-for-it/master/wait-for-it.sh \
 && chmod +x wait-for-it.sh \
 && mv wait-for-it.sh /usr/local/bin/

# Envoy v1.9
# https://github.com/envoyproxy/envoy/commit/37bfd8ac347955661af695a417492655b21939dc
COPY --from=envoyproxy/envoy:37bfd8ac347955661af695a417492655b21939dc /usr/local/bin/envoy /usr/local/bin/envoy

RUN wget https://dl.influxdata.com/telegraf/releases/telegraf-1.7.1-static_linux_amd64.tar.gz \
 && tar xzf telegraf-1.7.1-static_linux_amd64.tar.gz \
 && mv telegraf/telegraf /usr/local/bin \
 && rm -rf telegraf-1.7.1-static_linux_amd64.tar.gz telegraf

## RUN git clone -b 1.7.61 https://github.com/aws/aws-sdk-cpp.git \
##  && cd aws-sdk-cpp \
##  && mkdir build && cd build \
##  && cmake -DCMAKE_BUILD_TYPE=Release .. \
##  && make -j \
##  && make install \
##  && cd .. && rm -rf aws-sdk-cpp

## RUN apt update && apt install -y --no-install-recommends \
##         pkg-config zip g++ zlib1g-dev unzip python \
##  && rm -rf /var/lib/apt/lists/*

## ENV BAZEL_VERSION="0.21.0"
## 
## RUN wget https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh \
##  && chmod +x bazel-$BAZEL_VERSION-installer-linux-x86_64.sh \
##  && ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh \
##  && rm -f bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

# ===================
# TensorRT Laboratory
# ===================

COPY requirements.txt /tmp/requirements.txt

RUN python3 -m pip install -r /tmp/requirements.txt \
 && rm -f /tmp/requirements.txt


WORKDIR /work
COPY . .
RUN ./build.sh

ENV PYTHONPATH=/work/build/tensorrt-laboratory/python/trtlab
