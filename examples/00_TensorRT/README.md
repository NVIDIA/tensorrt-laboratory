# Inference Example

Basic CLI tool for executing TensorRT engines.

Provide an engine and `inference.x` will run a simplifed inference pipeline using synthetic data.

The program will run a pipelined H2D -> TensorRT -> D2H calculation for `--seconds` (default: 5) with a
0.1 second warmup run. By default, only 1 TensorRT Execution Context is used to perform the evaulation.
You can modify the number of contexts using the `--contexts`. Unless provided, the number of Input/Output
Buffers is set to `(2 * contexts)`.  See below for the list of [options](#options).

The `inference.x` program is fully pipelined and asynchronous.  It performs uses three threads (default)
to: 1) async copy input H2D, 2) launch the async inference evaluation and return output tensor to the host,
and 3) to wait on the resouces used during execution and release them when finished.  This final thread is
where one might build a return message or do something else with the results.

While running `inference.x`, you may find it useful to monitor GPU metrics using:
```
nvidia-smi dmon -i 0 -s put
```

Note: If you see numbers that differ from the output of `giexec`, you may have an IO bottleneck in that 
the transfers are more expensive than the compute.

 * TODO: Update the program to output avg xfer time.
 * TODO: Build .engine files as part of the build

## Quickstart

```
root@dgx:/work/build/examples/00_TensorRT# ./inference.x --engine=/work/models/ResNet-50-b1-int8.engine
I0702 22:16:51.868419 10857 TensorRT.cc:561] -- Initialzing TensorRT Resource Manager --
I0702 22:16:51.868676 10857 TensorRT.cc:562] Maximum Execution Concurrency: 1
I0702 22:16:51.868686 10857 TensorRT.cc:563] Maximum Copy Concurrency: 2
I0702 22:16:53.430330 10857 TensorRT.cc:628] -- Registering Model: 0 --
I0702 22:16:53.430399 10857 TensorRT.cc:629] Input/Output Tensors require 591.9 KiB
I0702 22:16:53.430415 10857 TensorRT.cc:630] Execution Activations require 2.5 MiB
I0702 22:16:53.430428 10857 TensorRT.cc:633] Weights require 30.7 MiB
I0702 22:16:53.437571 10857 TensorRT.cc:652] -- Allocating TensorRT Resources --
I0702 22:16:53.437587 10857 TensorRT.cc:653] Creating 1 TensorRT execution tokens.
I0702 22:16:53.437595 10857 TensorRT.cc:654] Creating a Pool of 2 Host/Device Memory Stacks
I0702 22:16:53.437607 10857 TensorRT.cc:655] Each Host Stack contains 608.0 KiB
I0702 22:16:53.437614 10857 TensorRT.cc:656] Each Device Stack contains 3.2 MiB
I0702 22:16:53.437623 10857 TensorRT.cc:657] Total GPU Memory: 6.5 MiB
I0702 22:16:53.540400 10857 inference.cc:93] -- Inference: Running for ~5 seconds with batch_size 1 --
I0702 22:16:58.543475 10857 inference.cc:131] Inference Results: 4770 batches in 5.00307 seconds; sec/batch: 0.00104886; inf/sec: 953.414
```

## Options
```
    -buffers (Number of Buffers (default: 2x contexts)) type: int32 default: 0
    -contexts (Number of Execution Contexts) type: int32 default: 1
    -cudathreads (Number Cuda Launcher Threads) type: int32 default: 1
    -engine (TensorRT serialized engine) type: string
      default: "/work/models/trt4.engine"
    -respthreads (Number Response Sync Threads) type: int32 default: 1
    -seconds (Number of Execution Contexts) type: int32 default: 5
```


