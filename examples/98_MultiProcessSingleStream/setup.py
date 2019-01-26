#!/usr/bin/env python3
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
import os
import subprocess

models = [
    ("ResNet-50-deploy.prototxt", "prob"),
#   ("ResNet-152-deploy.prototxt", "prob"),
]

precisions = [
    ("fp32", ""),
    ("fp16", "--fp16"),
    ("int8", "--int8")
]

def main():
    for model, o in models:
        for name, p in precisions:
            for b in [1]: #, 2, 4, 8]:
                n = "b{}-{}".format(b, name)
                e = model.replace("prototxt", "engine")
                e = e.replace("deploy", n)
                m = os.path.join("/work/models", model)
                if os.path.isfile(e):
                    continue
                subprocess.call("giexec --deploy={} --batch={} --output={} {} --engine={}".format(
                    m, b, o, p, e
                ), shell=True)

if __name__ == "__main__":
    main()
