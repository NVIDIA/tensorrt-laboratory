"""Build rule generator for locally installed TensorRT."""

# inspired from: https://github.com/google/nvidia_libs_test

def _get_env_var(repository_ctx, name, default):
    if name in repository_ctx.os.environ:
        return repository_ctx.os.environ[name]
    return default

def _impl(repository_ctx):
    hdrs_path = _get_env_var(repository_ctx, "TENSORRT_HDRS_PATH", "/usr/include/x86_64-linux-gnu")
    libs_path = _get_env_var(repository_ctx, "TENSORRT_LIBS_PATH", "/usr/lib/x86_64-linux-gnu")

    print("Using TensorRT Headers from %s\n" % hdrs_path)
    print("Using TensorRT Libs from %s\n" % libs_path)

    repository_ctx.symlink(hdrs_path, "include")
    repository_ctx.symlink(libs_path, "libs")

    repository_ctx.file("BUILD", """
package(default_visibility = ["//visibility:public"])

# The *_headers cc_library rules below aren't cc_inc_library rules because
# dependent targets would only see the first one.

cc_library(
    name = "tensorrt_headers",
    hdrs = glob(
        include = ["include/Nv*.h"],
    ),
    strip_include_prefix = "include",
    # Allows including CUDA headers with angle brackets.
    # includes = ["cuda/include"],
)

cc_library(
    name = "tensorrt_infer",
    srcs = ["libs/libnvinfer.so"],
    linkopts = ["-ldl"],
)

""")

tensorrt_configure = repository_rule(
    implementation = _impl,
    environ = ["TENSORRT_HDRS_PATH", "TENSORRT_LIBS_PATH"],
)
