"""Build rule generator for locally installed CUDA toolkit and cuDNN SDK."""

# src: https://github.com/google/nvidia_libs_test

def _get_env_var(repository_ctx, name, default):
    if name in repository_ctx.os.environ:
        return repository_ctx.os.environ[name]
    return default

def _impl(repository_ctx):
    cuda_path = _get_env_var(repository_ctx, "CUDA_PATH", "/usr/local/cuda")
    cudnn_path = _get_env_var(repository_ctx, "CUDNN_PATH", cuda_path)

    print("Using CUDA from %s\n" % cuda_path)
    print("Using cuDNN from %s\n" % cudnn_path)

    repository_ctx.symlink(cuda_path, "cuda")
    repository_ctx.symlink(cudnn_path, "cudnn")

    repository_ctx.file("nvcc.sh", """
#! /bin/bash
repo_path=%s
compiler=${CC:+"--compiler-bindir=$CC"}
$repo_path/cuda/bin/nvcc $compiler --compiler-options=-fPIC --include-path=$repo_path $*
""" % repository_ctx.path("."))

    repository_ctx.file("BUILD", """
package(default_visibility = ["//visibility:public"])

sh_binary(
    name = "nvcc",
    srcs = ["nvcc.sh"],
)

# The *_headers cc_library rules below aren't cc_inc_library rules because
# dependent targets would only see the first one.

cc_library(
    name = "cuda_headers",
    hdrs = glob(
        include = ["cuda/include/**/*.h*"],
        exclude = ["cuda/include/cudnn.h"]
    ),
    # Allows including CUDA headers with angle brackets.
    includes = ["cuda/include"],
)

cc_library(
    name = "cuda",
    srcs = ["cuda/lib64/stubs/libcuda.so"],
    linkopts = ["-ldl"],
)

cc_library(
    name = "cuda_runtime",
    srcs = ["cuda/lib64/libcudart_static.a"],
#   deps = [":cuda"],
    linkopts = ["-lrt"],
)

cc_library(
    name = "curand_static",
    srcs = [
        "cuda/lib64/libcurand_static.a",
        "cuda/lib64/libculibos.a",
    ],
)

cc_library(
    name = "cupti_headers",
    hdrs = glob(["cuda/extras/CUPTI/include/**/*.h"]),
    # Allows including CUPTI headers with angle brackets.
    includes = ["cuda/extras/CUPTI/include"],
)

cc_library(
    name = "cupti",
    srcs = glob(["cuda/extras/CUPTI/lib64/libcupti.so*"]),
)

cc_library(
    name = "cudnn",
    srcs = [
        "cudnn/lib64/libcudnn_static.a",
        "cuda/lib64/libcublas_static.a",
        "cuda/lib64/libculibos.a",
    ],
    hdrs = ["cudnn/include/cudnn.h"],
    deps = [
        ":cuda",
        ":cuda_headers"
    ],
)

cc_library(
    name = "cuda_util",
    deps = [":cuda_util_compile"],
)
""")

cuda_configure = repository_rule(
    implementation = _impl,
    environ = ["CUDA_PATH", "CUDNN_PATH"],
)
