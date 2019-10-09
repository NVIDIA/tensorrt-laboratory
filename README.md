# TensorRT Laboratory

The TensorRT Laboratory is a place where you can explore and build high-level inference 
examples that extend the scope of the examples provided with each of the NVIDIA software
products, i.e. CUDA, TensorRT, TensorRT Inference Server, and DeepStream.  We hope that 
the examples and ideas found in the playground will resonate and possibly inspire.

## Quickstart

  * Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
  * [optional] Sign-up for [NVIDIA GPU Cloud](https://ngc.nvidia.com/) and acquire an [API Key](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html#generating-api-key).
  * [optional] Authenticate your Docker client using your NGC API key. Yes, the username is `$oauthtoken`.
  
```
git clone http://github.com/nvidia/tensorrt-laboratory
cd tensorrt-laboratory
make
nvidia-docker run --rm -ti tensorrt-laboratory jupyter lab
# navigate to the notebooks folder and open Quickstart.ipynb 
```

## Overview

What do you want to do...

- Python
  - Perform inference using a local GPU
  - Perform inference using a remote GPU via TRTIS
  - Serve Models with TRTIS
  - Serve Models with a Custom Service

- C++
  - Add TensorRT to an existing application
      - High-Level Runtime - no direct CUDA or TensorRT API calls
      - Low-Level Runtime - direct control over CUDA and TensorRT primitives
        - Start with the high-level interface even if you already have an advanced CUDA application, very rarely will you need to go deep into the API.
  - Create a custom service for serving basic TensorRT models
  - Create a custom service for highly customized TensorRT models
    - Model Chaining: Model A -> Glue -> Model B -> ect 

- Build and Optimize the Serving Pipeline
    - Build Pre-/Post-processing services that interface with TRTIS
    - Avoiding expensive CPU serialization and deserialization
      - Use Flatbuffers to avoid the serialization of input and output data from/to the calling client
      - Use Shared Memory to transfer data between two separate services that live in the same Pod (Kubernetes) or IPC namespace (Docker)
    - Deploy an Inference Pipeline using Kubernetes and other open-source CNCF projects 
- Deploy an edge Video Streaming Service for your Home Automation System 

## DeepDive

### Core Components

There are 4 primary folder in the [tensorrt-laboratory](tensorrt-laboratory/) folder.  Each of these
components builds separately and build on each other.  Both CMake and Bazel builds are supported.

Components:
  - core - general algorithms and templates, no device specific code
  - cuda - CUDA implemntations of `core` functionality specific for GPU
  - nvrpc - gRPC helper library to simplify building async services.  nvRPC is a core component of the
    TensorRT Inference Server.  
  - tensorrt - high-level conviencence wrappers for TensorRT objects and functionality

### Core Concepts

#### CPU, GPU and Memory Affinity

Aligning host and device resources can greatly improve performance on dense GPU systems.  This example details how we ensure that the host threads and memory affinity are aligned with the closest NUMA-node (typically CPU socket) to each GPU

#### Built-in productivity tools

No one wants or needs to rewrite common components.  As the TensorRT playground is focused on computing inference either locally or remotely, we try to provided many of the underlying core components for executing short lived inference transactions.  List below are the core library components that can be used 

  - Use Smart CUDA / TensorRT Pointers
  - Affinity-aware ThreadPools
  - Resource Pool
  - Basic Memory Allocators
  - Advanced Memory Allocators

#### Explore advanced features of the TensorRT API

Use Unified Memory for TensorRT weight allocations.  This allows the weights of TensorRT 
engines to be paged in and out of host/GPU memory.  This example demonstrates how to 
create a customer IGpuAllocator and use it with an IRuntime to deserialize models and 
capture weights in unified memory.


Reduce memory footprint by using IExeuctionContexts that do not hold internal memory.  If you are hosting a large number of models, then you also need a large number of IExecutionContexts.  By default, each IExecutionContexts hold some internal memory for the scratch space needed for activations.  Using createExecutionContextWithoutDeviceMemory, we can create very lightweight IExecutionContexts; however, before we enqueue work on that context, we first must provide the memory required for the activations.  This API lets us use a smaller memory pool to avoid holding memory allocations for models that are not currently being computed. 


#### Fun with C++ Templates and Lambdas
  - ThreadPool
  - AsyncCompute
  - InferRunner


## Relation to TensorRT Inference Server

This project is designed to be an incubator for ideas that could go into production.  The [TensorRT Inference Server](https://devblogs.nvidia.com/nvidia-serves-deep-learning-inference/) uses some of the core concepts from this project, e.g. nvrpc.

## Copyright and License

This project is released under the [BSD 3-clause license](LICENSE).

## Issues and Contributing

* Please let us know by [filing a new issue](https://github.com/NVIDIA/tensorrt-laboratory/issues/new)
* You can contribute by opening a [pull request](https://help.github.com/articles/using-pull-requests/)

Pull requests with changes of 10 lines or more will require a [Contributor License Agreement](CLA).
