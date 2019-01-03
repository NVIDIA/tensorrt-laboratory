workspace(name = "com_github_nvidia_tensorrt_playground")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

#load(":repositories.bzl", "tensorrt_playground_cpp_repositories")
#tensorrt_playground_cpp_repositories()

http_archive(
    name = "com_github_grpc_grpc",
    urls = [
        "https://github.com/grpc/grpc/archive/v1.16.1.tar.gz",
    ],
    strip_prefix = "grpc-1.16.1",
)

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

load ("//bazel:cuda_configure.bzl", "cuda_configure")
cuda_configure(name = "local_config_cuda")

load ("//bazel:tensorrt_configure.bzl", "tensorrt_configure")
tensorrt_configure(name = "local_config_tensorrt")
