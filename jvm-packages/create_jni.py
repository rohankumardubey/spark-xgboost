#!/usr/bin/env python
import errno
import argparse
import glob
import os
import platform
import shutil
import subprocess
import sys
from contextlib import contextmanager
from cudautils import cudaver

# Monkey-patch the API inconsistency between Python2.X and 3.X.
if sys.platform.startswith("linux"):
    sys.platform = "linux"


CONFIG = {
    "USE_OPENMP": "ON",
    "USE_HDFS": "OFF",
    "USE_AZURE": "OFF",
    "USE_S3": "OFF",

    "USE_CUDA": "ON",
    "USE_NCCL": "ON",
    "PLUGIN_RMM": "ON",
    "JVM_BINDINGS": "ON",
    "LOG_CAPI_INVOCATION": "OFF",
    "BUILD_WITH_CUDA_CUB": "ON"
}


@contextmanager
def cd(path):
    path = normpath(path)
    cwd = os.getcwd()
    os.chdir(path)
    print("cd " + path)
    try:
        yield path
    finally:
        os.chdir(cwd)


def maybe_makedirs(path):
    path = normpath(path)
    print("mkdir -p " + path)
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def run(command, **kwargs):
    print(command)
    subprocess.check_call(command, shell=True, **kwargs)


def cp(source, target):
    source = normpath(source)
    target = normpath(target)
    print("cp {0} {1}".format(source, target))
    shutil.copy(source, target)


def normpath(path):
    """Normalize UNIX path to a native path."""
    normalized = os.path.join(*path.split("/"))
    if os.path.isabs(path):
        return os.path.abspath("/") + normalized
    else:
        return normalized


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-capi-invocation', type=str, choices=['ON', 'OFF'], default='OFF')
    parser.add_argument('--use-cuda', type=str, choices=['ON', 'OFF'], default='OFF')
    cli_args = parser.parse_args()

    if sys.platform == "darwin":
        # Enable of your compiler supports OpenMP.
        CONFIG["USE_OPENMP"] = "OFF"
        os.environ["JAVA_HOME"] = subprocess.check_output(
            "/usr/libexec/java_home").strip().decode()

    cuda = cudaver()
    print("building Java wrapper on " + cuda)
    with cd(".."):
        build_dir = 'build-gpu' if cli_args.use_cuda == 'ON' else 'build'
        maybe_makedirs(build_dir)
        with cd(build_dir):
            if sys.platform == "win32":
                # Force x64 build on Windows.
                maybe_generator = ' -A x64'
            else:
                maybe_generator = ""
            if sys.platform == "linux":
                maybe_parallel_build = " -- -j $(nproc)"
            else:
                maybe_parallel_build = ""

            if cli_args.log_capi_invocation == 'ON':
                CONFIG['LOG_CAPI_INVOCATION'] = 'ON'

            if cli_args.use_cuda == 'ON':
                CONFIG['USE_CUDA'] = 'ON'
                CONFIG['USE_NCCL'] = 'ON'

            args = ["-D{0}:BOOL={1}".format(k, v) for k, v in CONFIG.items()]

            # if enviorment set rabit_mock
            if os.getenv("RABIT_MOCK", None) is not None:
                args.append("-DRABIT_MOCK:BOOL=ON")

            # if enviorment set GPU_ARCH_FLAG
            gpu_arch_flag = os.getenv("GPU_ARCH_FLAG", None)
            if gpu_arch_flag is not None:
                args.append("%s" % gpu_arch_flag)

            lib_dir = os.path.join(os.pardir, 'lib')
            if os.path.exists(lib_dir):
                shutil.rmtree(lib_dir)
            run("cmake .. " + " ".join(args) + maybe_generator)
            run("cmake --build . --config Release" + maybe_parallel_build)

        with cd("demo/CLI/regression"):
            run(sys.executable + " mapfeat.py")
            run(sys.executable + " mknfold.py machine.txt 1")

    xgboost4j = 'xgboost4j-gpu' if cli_args.use_cuda == 'ON' else 'xgboost4j'
    xgboost4j_spark = 'xgboost4j-spark-gpu' if cli_args.use_cuda == 'ON' else 'xgboost4j-spark'

    print("copying native library")
    library_name, os_folder = {
        "Windows": ("xgboost4j.dll", "windows"),
        "Darwin": ("libxgboost4j.dylib", "macos"),
        "Linux": ("libxgboost4j.so", "linux"),
        "SunOS": ("libxgboost4j.so", "solaris"),
    }[platform.system()]
    arch_folder = {
        "x86_64": "x86_64",  # on Linux & macOS x86_64
        "amd64": "x86_64",  # on Windows x86_64
        "i86pc": "x86_64",  # on Solaris x86_64
        "sun4v": "sparc",  # on Solaris sparc
        "arm64": "aarch64",  # on macOS & Windows ARM 64-bit
        "aarch64": "aarch64"
    }[platform.machine().lower()]

    # get major version, e.g. get 'x' from 'x.y'
    cuda_major = cuda.split(".")[0]
    maybe_makedirs("xgboost4j/src/main/resources/lib/" + cuda_major)
    cp("../lib/" + library_name, "xgboost4j/src/main/resources/lib/" + cuda_major)

    print("copying pure-Python tracker")
    cp("../dmlc-core/tracker/dmlc_tracker/tracker.py",
       "{}/src/main/resources".format(xgboost4j))

    print("copying train/test files")
    maybe_makedirs("{}/src/test/resources".format(xgboost4j_spark))
    with cd("../demo/CLI/regression"):
        run("{} mapfeat.py".format(sys.executable))
        run("{} mknfold.py machine.txt 1".format(sys.executable))

    for file in glob.glob("../demo/CLI/regression/machine.txt.t*"):
        cp(file, "{}/src/test/resources".format(xgboost4j_spark))
    for file in glob.glob("../demo/data/agaricus.*"):
        cp(file, "{}/src/test/resources".format(xgboost4j_spark))

    maybe_makedirs("{}/src/test/resources".format(xgboost4j))
    for file in glob.glob("../demo/data/agaricus.*"):
        cp(file, "{}/src/test/resources".format(xgboost4j))
