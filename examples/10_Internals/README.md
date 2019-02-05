# YAIS Internals

The `internals.x` program is designed to be run on a DGX-Station or DGX-1.  This is mostly to highlight
the use of the Affinity API.  If you want to run on a different CPU architecture, you simply need to
change the following lines to a range that works with your CPU.

```
    // Socket 1 - non-hyperthreads on a DGX-1, or
    // Socket 0 - hyperthreads on a DGX-Station
    auto socket_1 = Affinity::GetAffinity().Intersection(
        Affinity::GetCpusFromString("20-39") // <== Change Me!
    );
```

## Primative Classes

  * `Affinity` 
    * Get and Set CPU affinities for current thread
  * `ThreadPool` 
    * Create generic worker thread pool that accept arbiturary lambda functions.
    * Pinned threads from `ThreadPools` are used to allocate memory to ensure that the CPU allocation
      are allocated and first-touched on the threads on the NUMA node for which it will be used.
      This is important for keeping threads and memory pool separate on NUMA systems.
  * `Memory`
    * `Memory` and the derived classes (`MallocMemory`, `CudaPinnedHostMemory`, `CudaDeviceMemory`) are not
      used directly; however they provided the implmentation details used by the generic `Allocator`.
  * `Allocator<MemoryType>`
    * Generic Templated Class used to create `std::shared_ptr` and `std::unique_ptr` to instances of
      `Allocator<MemoryType>`.
  * `MemoryStack<AllocatorType>`
    * Generic Templated Class to create a memory stack from a given `AllocatorType`.
    * You can only advance the stack pointer, or reset the entire stack.
    * TODO: Create sub-stacks from a given stack.
  * `Pool<ResourceType>`
    * Generic Templated Class that holds objects of `ResourceType`.
    * Resources can be checked-out of the Pool (Pop) as a *special-type* of `std::shared_ptr<ResourceType>`,
      which automatically returned the Resource object the pool when the reference count of the
      `shared_ptr` goes to zero.  This Resources are not lost on exceptions, but also that the Pool can not
      be deleted until all object have been returned to the Pool.

## TensorRT Classes

  * `Model` 
    * Wrapper around `nvinfer1::ICudaEngine`
  * `Buffers` 
    * `MemoryStackWithTracking<CudaPinnedHostMemory>` and `MemoryStackWithTracking<CudaDeviceMemory>` used
      to manage Input/Output Tensor Bindings.
    * Owns a `cudaStream_t` to be used with Async Copies and Kernel Executions on the data held by the Buffers.
    * Convenience H2D and D2H copy functions
  * `ExecutionContext` - Wrapper around `nvinfer1::IExecutionContext`
    * `Enqueue` launches the inference calculuation and adds a `cudaEvent_t` to the stream to be triggered
      when the inference calcuation is finished and the `ExecutionContext` can be released.
  * `Resources`
    * Combines the above set of resources into a single `trtlab::Resources` class capable of being associated
      with a `nvrpc::Context`.


## Examples

### Affinity

  * [Definition: tensorrt/laboratory/core/affinity.h](../../yais/include/tensorrt/laboratory/core/affinity.h)
  * [Implementation: YAIS/Affinity.cc](../../yais/src/Affinity.cc)

In this, we request the all logical CPUs from Socket 0 that are not hyperthreads, then we get either all 
the non-hyperthreads from socket_1 on a DGX-1, or the hyperthreads on socket0 on a DGX-Station using 
`GetCpusFromString`.

```
    // Socket 0 - non-hyperthreads on a DGX-1 or Station
    auto socket_0 = Affinity::GetAffinity().Intersection(
        Affinity::GetCpusBySocket(0).Intersection(
            Affinity::GetCpusByProcessingUnit(0)
    ));

    // Socket 1 - non-hyperthreads on a DGX-1, or
    // Socket 0 - hyperthreads on a DGX-Station
    auto socket_1 = Affinity::GetAffinity().Intersection(
        Affinity::GetCpusFromString("20-39")
    );

    LOG(INFO) << socket_0;
```

Single line output reformatted to per-line-indented output for readability.
```
0515 07:14:48.007148 10919 test_affinity.cc:61] 
    [id: 0, numa: 0, socket: 0, core: 0, processing_unit: 0], 
    [id: 1, numa: 0, socket: 0, core: 1, processing_unit: 0], 
    [id: 2, numa: 0, socket: 0, core: 2, processing_unit: 0],
    ... omitted for brevity ...
    [id: 18, numa: 0, socket: 0, core: 18, processing_unit: 0], 
    [id: 19, numa: 0, socket: 0, core: 19, processing_unit: 0]
```

### ThreadPool

  * [Definition: tensorrt/laboratory/core/thread_pool.h](../../yais/include/tensorrt/laboratory/core/thread_pool.h)
  * [Implementation: YAIS/ThreadPool.cc](../../yais/src/ThreadPool.cc)

The ThreadPool class creates a pool of worker threads that pull work from a queue.  The work queue can
be any set of captured lambda functions or function pointers passed to the `enqueue` function.

```
    // Create a ThreadPool where each thread is pinned to one logical CPU in the CpuSet
    auto workers_0 = std::make_shared<ThreadPool>(socket_0);
    auto workers_1 = std::make_shared<ThreadPool>(socket_1);

    // Create a massive set of threads that can run anywhere our current process is allowed to run
    auto bftp = std::make_unique<ThreadPool>(128, Affinity::GetAffinity());

    // Shutdown the BFTP
    bftp.reset();

    // Enqueue some basic logging
    for(int i=0; i<10; i++) {
        auto result = workers->enqueue([i]{
            LOG(INFO) << i << " " << Affinity::GetAffinity();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        });
    }
```

As these ThreadPools are generic, we can enqueue any type of work to them.  Many thanks to the original
authors Jakob Progsch and Václav Zeman for this incredible useful class.  For details on the original
work and the modificiations made in this project, see [CREDITS.md](../../CREDITS.md) and the source code.


### Memory

One of the reasons why `Affinity` and `ThreadPool` were introduced prior to `Memory` is that memory on
NUMA system can be difficult to do correctly.  For memory segments that will be primarly used by sets
of threads, it is very important to first set the affinity of the threads, then allocate and touch each
page in the memory allocation (first-touch) on the thread that will primarly use the segment.  NERSC 
has a nice [write-up on memory affinity and first touch policies](http://www.nersc.gov/users/computational-systems/cori/application-porting-and-performance/improving-openmp-scaling/).
In this section, we'll show how to properly use the `Memory` and `Allocator` classes in a NUMA friendly
way using `ThreadPool`s.

  * [Definition: tensorrt/laboratory/core/memory.h](../../yais/include/tensorrt/laboratory/core/memory.h)

The `Memory` class and its derived classes, see below, are the core memory classes in YAIS; however,
these classes are not direclty used.  Instead, they provide the implmentation details on how memory of
their respective classes is to be allocated, freed, and page-aligned.  For details, see the comments
in the source code.

Derived `Memory` Classes:
  * `Malloc`
  * `CudaPinnedHostMemory`
  * `CudaDeviceMemory`
  * `CudaManagedMemory`

### Allocator<MemoryType>

  * [Definition: tensorrt/laboratory/core/memory.h](../../yais/include/tensorrt/laboratory/core/memory.h)

The templated `Allocator<MemoryType>` class performs memory allocations and freeing operations.  This
class does not have a public constructor, instead, you are required to use either the `make_shared`
or `make_unique` static methods.  In doing so, the method to free the allocation is captured by the
deconstructor which is triggered by the default deleter of `shared_ptr` and `unique_ptr`.

An allocated memory segments is of type `Allocator<MemoryType>` which inherits from `MemoryType`.
The base `Memory` class provides three functions, `GetPointer()`, `GetSize()`, and `WriteZeros()`.

```
    std::shared_ptr<CudaPinnedHostMemory> pinned_0, pinned_1;

    auto future_0 = workers_0->enqueue([&pinned_0]{
        pinned_0 = Allocator<CudaPinnedHostMemory>::make_shared(1024*1024*1024);
        pinned_0->WriteZeros();
    });

    auto future_1 = workers_1->enqueue([&pinned_1]{
        pinned_1 = Allocator<CudaPinnedHostMemory>::make_shared(1024*1024*1024);
        pinned_1->WriteZeros();
    });

    future_0.get();
    CHECK(pinned_0) << "pinned_0 got deallocated - fail";
    LOG(INFO) << "pinned_0 (ptr, size): (" 
              << pinned_0->GetPointer() << ", "
              << pinned_0->GetSize() << ")";
```

```
I0515 08:36:56.619297 13260 test_affinity.cc:59] pinned_0 (ptr, size): (0x1005e000000, 1073741824)
```

### MemoryStack<AllocatorType>

  * [Definition: tensorrt/laboratory/core/memory_stack.h](../../yais/include/tensorrt/laboratory/core/memory_stack.h)

Generic `MemoryStack` that takes an `AllocatorType`.  The memory stack advances the stack pointer
via `Allocate` and resets the stack pointer via `ResetAllocations`.  `MemoryStackWithTracking`
is a specialized derivation that records the pointer and size of each call to `Allocate`.
`MemoryStackWithTracking` is used in the provided TensorRT classes as a means to push the
input/output tensor bindings onto the stack.

```
    std::shared_ptr<MemoryStackWithTracking<CudaDeviceMemory>> gpu_stack_on_socket0;

    future_0 = workers_0->enqueue([&gpu_stack_on_socket0]{
        CHECK_EQ(cudaSetDevice(0), CUDA_SUCCESS) << "Set Device 0 failed";
        gpu_stack_on_socket0 = std::make_shared<
            MemoryStackWithTracking<CudaDeviceMemory>>(1024*1024*1024);
    });

    future_0.get(); // thread allocating gpu_stack_on_socket0 finished with task
    
    LOG(INFO) << "Push Binding 0 - 10MB - stack_ptr = " 
        << gpu_stack_on_socket0->Allocate(10*1024*1024);
    LOG(INFO) << "Push Binding 1 - 128MB - stack_ptr = " 
        << gpu_stack_on_socket0->Allocate(128*1024*1024);
    gpu_stack_on_socket0->ResetAllocations();
```

```
I0515 09:46:55.159700 14176 test_affinity.cc:78] Push Binding 0 - 10MB - stack_ptr = 0x1009e000000
I0515 09:46:55.159710 14176 test_affinity.cc:80] Push Binding 1 - 128MB - stack_ptr = 0x1009ea00000
```

### Pool<ResourceType>

  * [Definition: tensorrt/laboratory/core/pool.h](../../yais/include/tensorrt/laboratory/core/pool.h)

A `Pool<ResourceType>` is a generic of `Queue<std::shared_ptr<ResourceType>>` with a special `Pop`
method.  The class inherits from `std::enabled_shared_from_this` meaning it must be constructed using
the factory method, which ensures the object is owned by a `std::shared_ptr`.

The `Pop` method of `Pool<ResourceType>` is probably the coolest and most contensious component of this
library.  `Pop` pulls an resource off the queue (`from_queue`); however, it does not return this resource.
Instead, a *new type* of `std::shared_ptr<ResourceType>` is created using the raw pointer from `from_pool`.
The reason they this is a *new type* of `shared_ptr` is because we provide a custom `Deleter` method that
captures by value (increments reference count) of both `from_pool` and a `shared_ptr` to the pool itself.

The custom `Deleter` does not free the resource when its reference count goes to zero; rather, it returns
the original `from_pool` `shared_ptr` to the pool.

By capturing a `shared_ptr` to the pool in the `Deleter`, we ensure the the pool can not be freed while
resources are checkedout.  This also ensures that the `shared_ptr` returned from `Pop` is exception
safe; meaning, the resource will be returned to the pool if an exception is thrown and caught - it won't
leak resources.

Alternatively, `Pop` can be called with an `onReturn` lambda function, which will be executed just prior
to the original object being returned to the Pool. If the `ResourceType` is stateful, this is a good 
chance to clear the state and prepare it for the next use.

```
    struct Buffer
    {
        Buffer(
            std::shared_ptr<CudaPinnedHostMemory> pinned_,
            std::shared_ptr<MemoryStackWithTracking<CudaDeviceMemory>> gpu_stack_,
            std::shared_ptr<ThreadPool> workers_
        ) : pinned(pinned_), gpu_stack(gpu_stack_), workers(workers_) {}

        // a real example probably includes a deviceID and a stream as part of the buffer

        std::shared_ptr<CudaPinnedHostMemory> pinned;
        std::shared_ptr<MemoryStackWithTracking<CudaDeviceMemory>> gpu_stack;
        std::shared_ptr<ThreadPool> workers;
    };

    auto buffers = Pool<Buffer>::Create();

    buffers->EmplacePush(new Buffer(pinned_0, gpu_stack_on_socket0, workers_0));
    buffers->EmplacePush(new Buffer(pinned_1, gpu_stack_on_socket1, workers_1));

    for(int i=0; i<6; i++)
    {
        auto buffer = buffers->Pop();
        buffer->workers->enqueue([buffer]{
            // perform some work - regardless of which buffer you got, you are working
            // on a thread properly assocated with the resources
            // note: buffer is captures by value, incrementing its reference count,
            // meaning you have access to it here and when it goes out of scope, it will
            // be returned to the Pool.
            LOG(INFO) << Affinity::GetAffinity();
        });
    }
```

## TensorRT Examples

  * [Definition: YAIS/YAIS/TensorRT/TensorRT.h](../../yais/include/YAIS/TensorRT/TensorRT.h)
  * [Implemenation: YAIS/TensorRT.cc](../../yais/src/TensorRT.cc)

TensoRT classes build on the primatives above.  For now, see the comments in the header file, as
the header file is pretty well documented.
