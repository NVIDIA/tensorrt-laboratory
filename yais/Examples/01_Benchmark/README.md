# Benchmark

Basic CLI benchmarking tool for TensorRT.

Provide an engine and `benchmark.x` will run a quick sanity benchmark.  

The program will run a pipelined H2D -> TensorRT -> D2H calculation for `--seconds` (default: 5) with a
0.1 second warmup run.  By default, only 1 TensorRT Execution Context is used to perform the evaulation.
You can modify the number of contexts using the `--contexts`.  By default this will increase the number of
Input/Output Buffers used as well.  See Options below for more details.

The benchmark program is fully pipelined and asynchronous.  It performs the full pipeline: H2D -> TensorRT -> D2H.
If you see numbers that differ from the output of `giexec`, you may have an IO bottleneck in that the transfers are
more expensive than the compute.

 * TODO: Update the benchmark program to output avg xfer time.
 * TODO: Build .engine files as part of the build

NOTE: for now, you need to bring your own models. I create a folder named `models` in my projects root directory.

## Quickstart

```
root@dgx:/work/build/Examples/01_Benchmark# ./benchmark.x --engine=/work/models/test.engine
I0524 22:05:12.294924   117 TensorRT.cc:72] Initializing Bindings from Engine
I0524 22:05:12.295197   117 TensorRT.cc:115] Binding: data; isInput: true; dtype size: 4; bytes per batch item: 602112
I0524 22:05:12.295212   117 TensorRT.cc:115] Binding: prob; isInput: false; dtype size: 4; bytes per batch item: 4000
I0524 22:05:12.295223   117 TensorRT.cc:207] Configuring TensorRT Resources
I0524 22:05:12.295231   117 TensorRT.cc:208] Creating 2 Host/Device Buffers each of size 6946048 on both host and device
I0524 22:05:12.295238   117 TensorRT.cc:210] Creating 1 TensorRT execution contexts (0 bytes/ctx)
I0524 22:05:12.408265   117 benchmark.cc:62] Benchmark: Running for ~5 seconds with batch_size 8
I0524 22:05:17.426430   117 benchmark.cc:91] Benchmark Results: 616 batches in 5.01817 seconds; sec/batch: 0.00814637; inf/sec: 982.032
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


