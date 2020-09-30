# TensorRT Laboratory

The TensorRT Laboratory (trtlab) is a general purpose set of tools to build customer inference applications
and services.

[Triton](https://github.com/nvidia/triton) is a professional grade production inference server.

This project is broken into 4 primary components:

  * `memory` is based on [foonathan/memory](https://github.com/foonathan/memory) the `memory` module 
     was designed to write custom allocators for both host and gpu memory.  Several custom allocators are
     included.  

  * `core` contains host/cpu-side tools for common components such as thread pools, resource pool, 
    and userspace threading based on boost fibers.

  * `cuda` extends `memory` with a new memory_type for CUDA device memory.  All custom allocators
    in `memory` can be used with `device_memory`, `device_managed_memory` or `host_pinned_memory`.

  * `nvrpc` is an abstraction layer for building asynchronous microservices.  The current implementation
    is based on grpc.

  * `tensorrt` provides an opinionated runtime built on the TensorRT API.

## Quickstart

The easiest way to manage the external NVIDIA dependencies is to leverage the containers hosted on
[NGC](https://ngc.nvidia.com).  For bare metal installs, use the `Dockerfile` as a template for
which NVIDIA libraries to install.

```
docker build -t trtlab . 
```

For development purposes, the following set of commands first builds the base image, then
maps the source code on the host into a running container.


```
docker build -t trtlab:dev --target base .
docker run --rm -ti --gpus=all -v $PWD:/work --workdir=/work --net=host trtlab:dev bash
```


## Copyright and License

This project is released under the [BSD 3-clause license](LICENSE).

## Issues and Contributing

* Please let us know by [filing a new issue](https://github.com/NVIDIA/tensorrt-laboratory/issues/new)
* You can contribute by opening a [pull request](https://help.github.com/articles/using-pull-requests/)

Pull requests with changes of 10 lines or more will require a [Contributor License Agreement](CLA).
