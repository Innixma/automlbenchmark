#!/usr/bin/env bash

# exit when any command fails
set -e

HERE=$(dirname "$0")
VERSION=${1:-"stable"}
REPO=${2:-"https://github.com/autogluon/autogluon.git"}
PKG=${3:-"autogluon"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="master"
fi

# creating local venv
. ${HERE}/../shared/setup.sh ${HERE} true

# Below fixes seg fault on MacOS due to bug in libomp: https://github.com/autogluon/autogluon/issues/1442
if [[ -x "$(command -v brew)" ]]; then
    wget https://raw.githubusercontent.com/Homebrew/homebrew-core/fb8323f2b170bd4ae97e1bac9bf3e2983af3fdb0/Formula/libomp.rb -P "${HERE}/lib"
    brew install "${HERE}/lib/libomp.rb"
fi

PIP install --upgrade pip
PIP install --upgrade setuptools wheel

PIP install s3fs

if [[ "$VERSION" == "latest_gpu" ]]; then
    PIP install mmcv
    PIP install pulp
    PIP install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchtext==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu113
fi

if [[ "$VERSION" == "stable" ]]; then
    PIP install --no-cache-dir -U "${PKG}"
    PIP install --no-cache-dir -U "${PKG}.tabular[skex]"
elif [[ "$VERSION" =~ ^[0-9] ]]; then
    PIP install --no-cache-dir -U "${PKG}==${VERSION}"
    PIP install --no-cache-dir -U "${PKG}.tabular[skex]==${VERSION}"
else
    VERSION="2024_10_24_amlb"
    REPO="https://github.com/Innixma/autogluon.git"

    TARGET_DIR="${HERE}/lib/${PKG}"
    rm -Rf ${TARGET_DIR}
    git clone --depth 1 --single-branch --branch ${VERSION} --recurse-submodules ${REPO} ${TARGET_DIR}
    cd ${TARGET_DIR}
    PY_EXEC_NO_ARGS="$(cut -d' ' -f1 <<<"$py_exec")"
    PY_EXEC_DIR=$(dirname "$PY_EXEC_NO_ARGS")
    env PATH="$PY_EXEC_DIR:$PATH" bash -c ./full_install.sh
    PIP install -e tabular/[skex]
fi

#PIP install s3fs
#PIP install "boto3==1.26.90"

PY -c "from autogluon.tabular.version import __version__; print(__version__)" >> "${HERE}/.setup/installed"
