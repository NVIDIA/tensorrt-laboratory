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

import click

precision_opts = {
  "fp32": "",
  "fp16": "--fp16",
  "int8": "--fp16 --int8",
}

File = click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True)

@click.command()
@click.option("--batch", type=click.IntRange(min=1, max=128), multiple=True)
@click.option("--precision", type=click.Choice(["fp32", "fp16"]), multiple=True, default="fp16")
@click.argument("models", type=File, nargs=-1)
def main(models, batch, precision):
    for model in models:
        #click.echo(model)
        #click.echo(precision)
        for p in precision:
            #click.echo(p)
            for b in batch:
                #click.echo(b)
                n = "b{}-{}".format(b, p)
                m = os.path.basename(model)
                m, ext = os.path.splitext(m)
                e = "{}-{}.{}".format(m,n,"engine")
                if os.path.isfile(e):
                    continue
                subprocess.call("trtexec --onnx={} --batch={} {} --saveEngine={}".format(model, b, precision_opts.get(p), e), shell=True)

if __name__ == "__main__":
    main()

