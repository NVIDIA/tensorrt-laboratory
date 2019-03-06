## Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
##
## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions
## are met:
##  * Redistributions of source code must retain the above copyright
##    notice, this list of conditions and the following disclaimer.
##  * Redistributions in binary form must reproduce the above copyright
##    notice, this list of conditions and the following disclaimer in the
##    documentation and/or other materials provided with the distribution.
##  * Neither the name of NVIDIA CORPORATION nor the names of its
##    contributors may be used to endorse or promote products derived
##    from this software without specific prior written permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
## EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
## IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
## PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
## CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
## EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
## PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
## PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
## OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
## (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
## OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
import os
import uuid

import boto3
import deploy_image_client as client

class ImageClient:

    def __init__(self, *, hostname = "trt.lab"):
        self._cpp_client = client.ImageClient(hostname)
        self._s3_client = self._get_s3_client()

    def classify(self, image_path, model):
        key = self._upload_to_s3(image_path)
        return self._cpp_client.classify(key, model)

    def object_detection(self, image_path, model):
        key = self._upload_to_s3(image_path)
        return self._cpp_client.object_detection(key, model)

    def _get_s3_client(self):
        kwargs = {}
        if os.environ.get("AWS_ENDPOINT_URL"):
            kwargs = {
                endpoint_url: os.environ.get("AWS_ENDPOINT_URL"),
                use_ssl: False,
                verify: False,
            }
        return boto3.client("s3", **kwargs)

    def _check_if_file(self, file_path):
        if not os.path.isfile(file_path):
            raise RuntimeError("{} is not a file".format(file_path))

    def _upload_to_s3(self, image_path):
        self._check_if_file(image_path)
        key = str(uuid.uuid4())
        with open(image_path, "rb") as data:
            self._s3_client.upload_fileobj(data, 'images', key)
        return key
