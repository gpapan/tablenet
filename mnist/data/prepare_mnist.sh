#!/bin/sh

CAFFE_DIR=${HOME}/rmt/work/deeplabel/caffe_google

# Get MNIST data and build LMDB files if not already there

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

if [ ! -d mnist_train_lmdb ]; then
    rm -rf mnist_{train,test}_lmdb
    rm -f {train,t10k}-*
    ${CAFFE_DIR}/data/mnist/get_mnist.sh
    mv ${CAFFE_DIR}/data/mnist/{train,t10k}-* .
    CONVERT_BIN=${CAFFE_DIR}/.build_release/examples/mnist/convert_mnist_data.bin
    ${CONVERT_BIN} \
        train-images-idx3-ubyte \
        train-labels-idx1-ubyte \
        mnist_train_lmdb \
        --backend=lmdb
    ${CONVERT_BIN} \
        t10k-images-idx3-ubyte \
        t10k-labels-idx1-ubyte \
        mnist_test_lmdb \
        --backend=lmdb
    rm {train,t10k}-*
fi
