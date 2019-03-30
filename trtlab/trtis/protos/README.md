# TRTIS Protos

I have to manually copy all the .protos from the [TensorRT Inference Server](https://github.com/nvidia/tensorrt-inference-server)
repo.

Unfortunately, there is now good way to link again TRTIS as an external library.

Tracking:
https://github.com/NVIDIA/tensorrt-inference-server/issues/110

The import paths are manually modifyed to correspond to the trtlab project.
I don't understand why we cannot use import paths relative to the local
directory.

Similarly, it is really difficult to get CMake and Bazel to build protobufs
in a consistent way between build platforms.

I created two new marcros for this purpose.

- `PROTOBUF_GENERATE_CPP_LIKE_BAZEL`
- `GRPC_GENERATE_CPP_LIKE_BAZEL`

These new macros generate the compiled proto into source files located
in the full path from the project root/workspace.  This makes its so the
.h files will appear in a `trtlab/trtis/protos` directory, which
actually provides a better and more descriptive #include.

```
#include "model_config.pb.h"
```

becomes

```
#include "trtlab/trtis/protos/model_config.pb.h"
```
