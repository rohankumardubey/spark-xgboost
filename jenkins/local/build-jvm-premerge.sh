#!/bin/bash
##
#
# Script to build xgboost jar files.
#
# Source tree is supposed to be ready by Jenkins
# before starting this script.
#
###
set -e
gcc --version

source $WORKSPACE/jenkins/local/common.sh
build_rmm

BUILD_ARG="-Dmaven.repo.local=$WORKSPACE/.m2 -DskipTests -B -s settings.xml -Pmirror-apache-to-art"

if [ "${CUDA_VER}"x != x ];then
   . /opt/tools/to_cudaver.sh $CUDA_VER
fi
echo "CUDA_VER: $CUDA_VER, BUILD_ARG: $BUILD_ARG"

CUDA_UTIL=$WORKSPACE/jvm-packages/cudautils.py
# Use the default cuda version in docker image, get the related classifier
CLASSIFIER=`$CUDA_UTIL g`

cd $WORKSPACE/jvm-packages
rm -rf ../build
mvn $BUILD_ARG clean package -Dcudf.classifier=$CLASSIFIER
cd ..
