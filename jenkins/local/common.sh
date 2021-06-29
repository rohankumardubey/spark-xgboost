#! /bin/bash

function build_rmm() {
  pushd `pwd`
  cd $WORKSPACE/jvm-packages
  RMM_VERSION=$(mvn help:evaluate -Dexpression=cudf.version -q -DforceStdout | grep -o -E '([0-9]+\.[0-9]+)')

  root_dir=$WORKSPACE/rapids-build
  rm -fr $root_dir
  mkdir -p $root_dir
  export RMM_ROOT=$root_dir/installed

  cd $root_dir
  git clone --recurse-submodules https://github.com/rapidsai/rmm.git -b branch-$RMM_VERSION
  mkdir -p $root_dir/rmm/build
  cd $root_dir/rmm/build

  echo "RMM SHA: `git rev-parse HEAD`"
  cmake .. -DCMAKE_INSTALL_PREFIX=$RMM_ROOT -DBUILD_TESTS=OFF
  make -j2 install

  # Install spdlog headers from RMM build
  (cd $root_dir/rmm/build/_deps/spdlog-src && find include/spdlog | cpio -pmdv $RMM_ROOT)

  popd
}
