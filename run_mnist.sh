#!/bin/sh

ROOT_DIR=/usr/local/google/home/gpapan

CAFFE_DIR=${ROOT_DIR}/rmt/work/deeplabel/caffe_google
CAFFE_BIN=${CAFFE_DIR}/.build_release/tools/caffe.bin

EXP=mnist
NUM_LABELS=10

NET_ID=lenet

DEV_ID=0

#####

# Create dirs

CONFIG_DIR=${EXP}/config/${NET_ID}
MODEL_DIR=${EXP}/model/${NET_ID}
mkdir -p ${MODEL_DIR}
LOG_DIR=${EXP}/log/${NET_ID}
mkdir -p ${LOG_DIR}
export GLOG_log_dir=${LOG_DIR}

# Run 

RUN_TRAIN=1
RUN_TEST=0

# Training

if [ ${RUN_TRAIN} -eq 1 ]; then
    echo Training net ${EXP}/${NET_ID}
    TRAIN_SET=train
    for pname in solver; do
	sed "$(eval echo $(cat sub.sed))" \
            ${CONFIG_DIR}/${pname}.prototxt > ${CONFIG_DIR}/${pname}_${TRAIN_SET}.prototxt
    done
    CMD="${CAFFE_BIN} train \
         --solver=${CONFIG_DIR}/solver_${TRAIN_SET}.prototxt \
         --gpu=${DEV_ID}"
    MODEL=${EXP}/model/${NET_ID}/init.caffemodel
    if [ -f ${MODEL} ]; then
	CMD="${CMD} --weights=${MODEL}"
    fi
    echo Running ${CMD} && ${CMD}
fi

if [ ${RUN_TEST} -eq 1 ]; then
    MODEL=${EXP}/model/${NET_ID}/test.caffemodel
    if [ ! -f ${MODEL} ]; then
	MODEL=`ls -t ${EXP}/model/${NET_ID}/train_iter_*.caffemodel | head -n 1`
    fi
    echo Testing net ${EXP}/${NET_ID}
    CMD="${CAFFE_BIN} test \
         --model=${CONFIG_DIR}/test_${TEST_SET}.prototxt \
         --gpu=${DEV_ID} \
         --iterations=${TEST_ITER}"
    echo Running ${CMD} && ${CMD}
fi
