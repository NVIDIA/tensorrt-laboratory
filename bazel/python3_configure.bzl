"""Build rule generator for locally installed Python3."""

# inspired from: https://github.com/google/nvidia_libs_test

def _get_env_var(repository_ctx, name, default):
    if name in repository_ctx.os.environ:
        return repository_ctx.os.environ[name]
    return default

def _impl(repository_ctx):
    hdrs_path = _get_env_var(repository_ctx, "PYTHON3_HDRS_PATH", "/usr/include/python3.5")
    libs_path = _get_env_var(repository_ctx, "PYTHON3_LIBS_PATH", "/usr/lib/x86_64-linux-gnu")

    print("Using Python3 Headers from %s\n" % hdrs_path)
    print("Using Python3 Libs from %s\n" % libs_path)

    repository_ctx.symlink(hdrs_path, "include")
    repository_ctx.symlink(libs_path, "libs")

    repository_ctx.file("BUILD", """
package(default_visibility = ["//visibility:public"])

# The *_headers cc_library rules below aren't cc_inc_library rules because
# dependent targets would only see the first one.

cc_library(
    name = "libpython3",
    srcs = [
#       "libs/libpython3.5m.so.1",
    ],
    hdrs = glob(["include/*.h"]),
    strip_include_prefix = "include",
    visibility = ["//visibility:public"]
)

""")

python3_configure = repository_rule(
    implementation = _impl,
    environ = ["PYTHON3_HDRS_PATH", "PYTHON3_LIBS_PATH"],
)
