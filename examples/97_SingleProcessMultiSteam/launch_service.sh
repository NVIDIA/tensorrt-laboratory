#!/bin/bash -e
#
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
#
cleanup() {
  kill $(jobs -p) ||:
}
trap "cleanup" EXIT SIGINT SIGTERM

ENG=${1:-/work/models/ResNet-50-b1-fp32.engine}
NCTX=${2:-1}

if [ ! -e $ENG ]; then
    echo "$ENG not found"
    exit 911
fi

port=50051
/work/build/examples/02_TensorRT_GRPC/inference-grpc.x --port=$port --engine=${ENG} --contexts=$NCTX &
wait-for-it.sh localhost:$port --timeout=0 -- echo "YAIS Service is ready." > /dev/null 2>&1

echo "warmup with client-async.x"
/work/build/examples/02_TensorRT_GRPC/client-async.x --count=1000 --port=$port

echo
echo "Starting a shell keeping the services and load-balancer running..."
echo "Try /work/build/examples/02_TensorRT_GRPC/siege.x --rate=2000 --port=$port"
bash --rcfile <(echo "PS1='$NCTX x $ENG Subshell: '")
