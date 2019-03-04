load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repositories():
    _maybe(
        http_archive,
        name = "com_github_antonovvk_bazel_rules",
        sha256 = "ba75b07d3fd297375a6688e9a16583eb616e7a74b3d5e8791e7a222cf36ab26e",
        strip_prefix = "bazel_rules-98ddd7e4f7c63ea0868f08bcc228463dac2f9f12",
        urls = [
            "https://mirror.bazel.build/github.com/antonovvk/bazel_rules/archive/98ddd7e4f7c63ea0868f08bcc228463dac2f9f12.tar.gz",
            "https://github.com/antonovvk/bazel_rules/archive/98ddd7e4f7c63ea0868f08bcc228463dac2f9f12.tar.gz",
        ],
    )

    _maybe(
        http_archive,
        name = "com_github_gflags_gflags",
        sha256 = "6e16c8bc91b1310a44f3965e616383dbda48f83e8c1eaa2370a215057b00cabe",
        strip_prefix = "gflags-77592648e3f3be87d6c7123eb81cbad75f9aef5a",
        urls = [
            "https://mirror.bazel.build/github.com/gflags/gflags/archive/77592648e3f3be87d6c7123eb81cbad75f9aef5a.tar.gz",
            "https://github.com/gflags/gflags/archive/77592648e3f3be87d6c7123eb81cbad75f9aef5a.tar.gz",
        ],
    )

    _maybe(
        http_archive,
        name = "com_google_glog",
        sha256 = "1ee310e5d0a19b9d584a855000434bb724aa744745d5b8ab1855c85bff8a8e21",
        strip_prefix = "glog-028d37889a1e80e8a07da1b8945ac706259e5fd8",
        urls = [
            "https://mirror.bazel.build/github.com/google/glog/archive/028d37889a1e80e8a07da1b8945ac706259e5fd8.tar.gz",
            "https://github.com/google/glog/archive/028d37889a1e80e8a07da1b8945ac706259e5fd8.tar.gz",
        ],
    )

    _maybe(
        http_archive,
        name = "com_google_googletest",
        sha256 = "c18f281fd6621bb264570b99860a0241939b4a251c9b1af709b811d33bc63af8",
        strip_prefix = "googletest-e3bd4cbeaeef3cee65a68a8bd3c535cb779e9b6d",
        urls = [
            "https://mirror.bazel.build/github.com/google/googletest/archive/e3bd4cbeaeef3cee65a68a8bd3c535cb779e9b6d.tar.gz",
            "https://github.com/google/googletest/archive/e3bd4cbeaeef3cee65a68a8bd3c535cb779e9b6d.tar.gz",
        ],
    )

    _maybe(
        http_archive,
        name = "com_github_grpc_grpc",
        strip_prefix = "grpc-1.16.1",
        urls = [
            "https://github.com/grpc/grpc/archive/v1.16.1.tar.gz",
        ],
    )

def load_trtis():
    http_archive(
        name = "com_github_nvidia_trtis",
        strip_prefix = "tensorrt-inference-server-0.9.0",
        urls = [
            "https://github.com/NVIDIA/tensorrt-inference-server/archive/v0.9.0.tar.gz",
        ],
    )

def load_benchmark():
    http_archive(
        name = "com_github_google_benchmark",
        sha256 = "f8e525db3c42efc9c7f3bc5176a8fa893a9a9920bbd08cef30fb56a51854d60d",
        strip_prefix = "benchmark-1.4.1",
        urls = [
            "https://github.com/google/benchmark/archive/v1.4.1.tar.gz",
        ],
    )

def _maybe(repo_rule, name, **kwargs):
    if name not in native.existing_rules():
        repo_rule(name = name, **kwargs)
