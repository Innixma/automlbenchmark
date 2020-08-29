#!/usr/bin/env bash
HERE=$(dirname "$0")

# creating local venv
. $HERE/../shared/setup.sh $HERE
#if [[ -x "$(command -v apt-get)" ]]; then
#    SUDO apt-get install -y libomp-dev
if [[ -x "$(command -v brew)" ]]; then
    brew install libomp
fi

# cat $HERE/requirements.txt | sed '/^$/d' | while read -r i; do PIP install "$i"; done
PIP install -U --pre "mxnet>=1.7.0b20200713, <2.0.0" -f https://sxjscience.github.io/KDD2020/

# git clone https://github.com/awslabs/autogluon.git
git clone -b tabular_fix_weighted_ensemble_metric --single-branch https://github.com/awslabs/autogluon.git
# git clone -b nn_fastai_v2 --single-branch https://github.com/gradientsky/autogluon.git

PIP install -e autogluon

#PIP install --no-cache-dir -r $HERE/requirements.txt
