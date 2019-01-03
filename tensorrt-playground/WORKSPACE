workspace(name = "com_github_nvidia_tensorrt_playground")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

#load(":repositories.bzl", "tensorrt_playground_cpp_repositories")
#tensorrt_playground_cpp_repositories()

http_archive(
    name = "com_github_grpc_grpc",
    urls = [
        "https://github.com/grpc/grpc/archive/3acf8e62079e727ae1925be759fa917f10e982bb.tar.gz",
    ],
    strip_prefix = "grpc-3acf8e62079e727ae1925be759fa917f10e982bb",
)

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

load ("//:cuda_configure.bzl", "cuda_configure")
cuda_configure(name = "local_config_cuda")

load ("//:tensorrt_configure.bzl", "tensorrt_configure")
tensorrt_configure(name = "local_config_tensorrt")
