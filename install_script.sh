#!/bin/bash

set -e

git clone https://github.com/microsoft/varuna.git
git clone https://github.com/NVIDIA/apex

cd apex
git reset --hard 0c2c6eea6556b208d1a8711197efc94899e754e1
git clean -df


cp path/to/apex_new.patch ./

git apply apex_new.patch
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..
rm -rf apex


cd varuna
python setup.py install
cd ..
