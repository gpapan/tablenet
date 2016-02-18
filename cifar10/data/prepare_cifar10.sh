#!/bin/sh

CAFFE_DIR=${HOME}/rmt/work/deeplabel/caffe_google

# Get data and build LEVELDB files if not already there

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
echo $DIR
cd $DIR

if [ ! -d orig ]; then
    mkdir -p orig

    rm -rf cifar10_{train,test}_leveldb

    echo "Downloading cifar-10 ..."
    ${CAFFE_DIR}/data/cifar10/get_cifar10.sh
    mv ${CAFFE_DIR}/data/cifar10/{*.bin,batches.meta.txt,readme.html} .

    echo "Creating leveldb ..."
    CONVERT_BIN=${CAFFE_DIR}/.build_release/examples/cifar10/convert_cifar_data.bin
    ${CONVERT_BIN} ./ ./orig leveldb

    echo "Computing image mean..."
    MEAN_BIN=${CAFFE_DIR}/.build_release/tools/compute_image_mean
    ${MEAN_BIN} -backend=leveldb orig/cifar10_train_leveldb orig/mean.binaryproto

    rm -f {*.bin,batches.meta.txt,readme.html}
fi

if [ ! -d white ]; then
    mkdir -p white

    echo "Downloading whitened cifar-10 ..."
    wget https://www.dropbox.com/s/9m3cx3mmpwohstf/cifar-train-leveldb.tar.gz
    wget https://www.dropbox.com/s/lricqumnwqpv9ac/cifar-test-leveldb.tar.gz

    tar xvfz cifar-train-leveldb.tar.gz
    tar xvfz cifar-test-leveldb.tar.gz

    mv cifar-train-leveldb white/cifar10_train_leveldb
    mv cifar-test-leveldb white/cifar10_test_leveldb

    rm -rf cifar-train-leveldb* cifar-test-leveldb*
fi

echo "Done."
