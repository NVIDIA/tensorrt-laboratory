
cc_library(
    name = "nvrpc",
    srcs = glob([
        "src/**/*.cc",
        "src/**/*.h",
    ]),
    hdrs = glob([
        "include/**/*.h",
    ]),
    deps = [
        "//core",
        "@com_github_grpc_grpc//:grpc++_unsecure",
    ],
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)
