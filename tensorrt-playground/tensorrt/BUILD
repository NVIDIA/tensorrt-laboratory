
cc_library(
    name = "tensorrt",
    srcs = glob([
        "src/**/*.cc",
        "src/**/*.h",
    ]),
    hdrs = glob([
        "include/**/*.h",
    ]),
    deps = [
        "//tensorrt-playground/core",
        "//tensorrt-playground/cuda",
        "@local_config_tensorrt//:tensorrt_headers",
        "@local_config_tensorrt//:tensorrt_infer",
    ],
    linkopts = [
        "-L/usr/local/cuda/lib64/stubs",
    ],
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)
