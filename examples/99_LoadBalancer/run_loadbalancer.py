#!/usr/bin/env python3
#
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
#
import os
import inspect
import shutil
import tempfile

import click
from jinja2 import Environment, FileSystemLoader, Template

def render(template_path, data=None, extensions=None, strict=False):
    data = data or {}
    extensions = extensions or []
    env = Environment(
        loader=FileSystemLoader(os.path.dirname(template_path)),
        extensions=extensions,
        keep_trailing_newline=True,
    )
    if strict:
        from jinja2 import StrictUndefined
        env.undefined = StrictUndefined

    # Add environ global
    env.globals['environ'] = os.environ.get

    return env.get_template(os.path.basename(template_path)).render(data)

script_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
FileType = click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True)

@click.command()
@click.option("-n", default=1)
@click.option("--template", type=FileType, default=os.path.join(script_path, "lb-envoy.j2"))
def main(n, template):
    envoy = shutil.which("envoy")
    if not os.path.isfile(envoy):
        raise RuntimeError("envoy executable not found in currently directory: {}".format(envoy))
    ports = [50051 + p for p in range(n)]
    print("load balancing over ports: ", [str(p) for p in ports])
    with open("/tmp/lb-envoy.yaml", "w") as file:
        file.write(render(template, data={"ports": ports}))
#   os.system("{} -c /tmp/lb-envoy.yaml".format(envoy))

if __name__ == "__main__":
    main()
    
