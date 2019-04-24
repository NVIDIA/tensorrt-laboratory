#
# Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.

import tensorrt as trt
import os

import pycuda.driver as cuda
import pycuda.autoinit
import matplotlib.pyplot as plt

import mxnet as mx
from mxnet.gluon.data.vision import transforms
import numpy as np
from random import shuffle

class ONNXEntropyCalibrator(trt.IInt8EntropyCalibrator):
    def __init__(self, image_dir, batch_size, calibration_batches, cache_file):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator.__init__(self)

        self.cache_file = cache_file

        # Get a list of all the images in the image directory.
        image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
        shuffle(image_files)

        if len(image_files) < calibration_batches * batch_size:
            print("Only found enough images for {} batches instead of {}, continuing anyway...".format(len(image_files) // batch_size, calibration_batches))
            self.image_files = image_files
        else:
            self.image_files = image_files[:calibration_batches * batch_size]

        # Keeps track of current image in image list
        self.current_image = 0
        self.batch_size = batch_size
        self.input_size = [3,224,224]

        # Each element of the calibration data is a float32.
        self.device_input = cuda.mem_alloc(self.batch_size * self.input_size[0] * self.input_size[1] * self.input_size[2]  * trt.float32.itemsize)

        # Create a generator that will give us batches. We can use next() to iterate over the result.
        def load_batches():
            while self.current_image < len(self.image_files):
                data, images_read = self.read_image_batch()
                self.current_image += images_read
                yield data
        self.batches = load_batches()


    def transform_image(self, img):
        transform_fn = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img = transform_fn(mx.nd.array(img)).asnumpy()
        return img


    # This function is used to load calibration images into batches.
    def read_image_batch(self):
        # Depending on batch size and number of images, the final batch might only be partially full.
        images_to_read = min(self.batch_size, len(self.image_files) - self.current_image)

        host_buffer = np.zeros(shape=[self.batch_size]+self.input_size)
        for i in range(images_to_read):
            img = np.array(plt.imread(self.image_files[self.current_image]))
            img = self.transform_image(img)
            host_buffer[i,:,:,:] = img

        return host_buffer, images_to_read

    def get_batch_size(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        try:
            # Get a single batch.
            data = np.ascontiguousarray(next(self.batches), np.float32)
            # Copy to device, then return a list containing pointers to input device buffers.
            cuda.memcpy_htod(self.device_input, data)
            return [int(self.device_input)]
        except StopIteration:
            # When we're out of batches, we return either [] or None.
            # This signals to TensorRT that there is no calibration data remaining.
            return None

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
