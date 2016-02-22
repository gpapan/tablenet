#!/bin/sh

CAFFE_DIR=${HOME}/rmt/work/deeplabel/caffe_google
CAFFE_BIN=${CAFFE_DIR}/.build_release/tools/caffe.bin

EXP=cifar10
NUM_LABELS=10

# Times refer to 100 train iters of 64-image mini-batches on a Titan-X.

# Uncomment the NET_ID you want to experiment with.
# NET_ID=treenet_logit1x1_lutWxW  ## too many parameters, accuracy oscillates between 60% and 96%

for NET_ID in treenet_logitWxW_lut1x1_noncum_leaf; do

DEV_ID=0

#####

# Create dirs

CONFIG_DIR=${EXP}/config/${NET_ID}
MODEL_DIR=${EXP}/model/${NET_ID}
mkdir -p ${MODEL_DIR}
LOG_DIR=${EXP}/log/${NET_ID}
mkdir -p ${LOG_DIR}
export GLOG_log_dir=${LOG_DIR}
DATA_DIR=${EXP}/data

# Get CIFAR-10 data and build LMDB files if not already there
${DATA_DIR}/prepare_${EXP}.sh

# Run 

RUN_TRAIN=1
RUN_TIME=0
RUN_TEST=0

# Training + Testing

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
    MODEL=${MODEL_DIR}/init.caffemodel
    if [ -f ${MODEL} ]; then
	CMD="${CMD} --weights=${MODEL}"
    fi
    echo Running ${CMD} && ${CMD}
fi

if [ ${RUN_TIME} -eq 1 ]; then
    MODEL=${MODEL_DIR}/test.caffemodel
    if [ ! -f ${MODEL} ]; then
	MODEL=`ls -t ${MODEL_DIR}/train_iter_*.caffemodel | head -n 1`
    fi
    echo Testing net ${EXP}/${NET_ID}
    TEST_ITER=100  # 10,000 images in 100 batches
    CMD="${CAFFE_BIN} time \
         --model=${CONFIG_DIR}/model.prototxt \
         --weights=${MODEL} \
         --gpu=${DEV_ID} \
         --iterations=${TEST_ITER}"
    echo Running ${CMD} && ${CMD}
fi

# Testing only

if [ ${RUN_TEST} -eq 1 ]; then
    MODEL=${MODEL_DIR}/test.caffemodel
    if [ ! -f ${MODEL} ]; then
	MODEL=`ls -t ${MODEL_DIR}/train_iter_*.caffemodel | head -n 1`
    fi
    echo Testing net ${EXP}/${NET_ID}
    TEST_ITER=100  # 10,000 images in 100 batches
    CMD="${CAFFE_BIN} test \
         --model=${CONFIG_DIR}/model.prototxt \
         --weights=${MODEL} \
         --gpu=${DEV_ID} \
         --iterations=${TEST_ITER}"
    echo Running ${CMD} && ${CMD}
fi

done
