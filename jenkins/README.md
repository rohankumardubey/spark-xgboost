# Build Jar artifact of XGBoost

## Build the docker image

### Prerequisite

1. Docker should be installed.
2. [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) should be installed.

### Build the docker image

In the root path of XGBoost repo, run below command to build the docker image.
```bash
docker build -f jenkins/local/Dockerfile.centos7_build jenkins/local -t xgboost-build:dev3-centos7-cuda11.2
```

## Start the docker then build

### Start the docker

Run below command to start a docker container with GPU.
```bash
nvidia-docker run -it xgboost-build:dev3-centos7-cuda11.2 bash
```

### Download the XGBoost source code

You can download the XGBoost repo in the docker container or you can mount it into the container.
Here I choose to download again in the container.
```bash
git clone --recursive https://gitlab-master.nvidia.com/nvspark/xgboost.git -b nv-release-1.4.0 
```

### Build XGBoost jar with devtoolset

```bash
cd xgboost
export WORKSPACE=`pwd`
export OUT="$WORKSPACE/out"
export SIGN_FILE="false"
export ART_URL="https://repo.maven.apache.org/maven2"
scl enable devtoolset-9 "jenkins/local/build-jvm-multi.sh $SIGN_FILE"
```

### The output

You can find the XGBoost jar in out/ dir, like out/xgboost4j/xgboost4j_3.0-1.30-SNAPSHOT.jar, out/xgboost4j-spark/xgboost4j-spark_3.0-1.3.0-SNAPSHOT.jar.
