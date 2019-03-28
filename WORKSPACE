workspace(name = "com_github_nvidia_trtlab")

load(":bazel/repositories.bzl", "repositories")
repositories()

load ("//bazel:cuda_configure.bzl", "cuda_configure")
cuda_configure(name = "local_config_cuda")

load ("//bazel:tensorrt_configure.bzl", "tensorrt_configure")
tensorrt_configure(name = "local_config_tensorrt")

load ("//bazel:python3_configure.bzl", "python3_configure")
python3_configure(name = "local_config_python3")

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
grpc_deps()

