#!/bin/sh

CAFFE_DIR=${HOME}/rmt/work/deeplabel/caffe_google
CAFFE_BIN=${CAFFE_DIR}/.build_release/tools/caffe.bin

EXP=mnist
NUM_LABELS=10

# Times refer to 100 train iters of 64-image mini-batches on a Titan-X.

# Uncomment the NET_ID you want to experiment with.

# (1) Baseline convnet with filters of size [C_out C_in W W]. Time: 1.8 sec.
#NET_ID=lenet  ## 0.9895

# (2) Tablenet quantizing Q x W x W input patches. Has 1 x 1 LUTs. Time 5.2 sec.
# NET_ID=tablenet_logitWxW_lut1x1  ## 0.9885

# (3) Tablenet quantizing Q x 1 x 1 input patches and having W x W LUTs. Time 5.8 sec.
# NET_ID=tablenet_logit1x1_lutWxW  ## 0.9874

## Various forms of (soft) Winner-Take-All

# (4) Similar to (2) but enforces semi-hard assignment:
#     Dominant state retains its p_max, others receive (1 - p_max) / (L - 1). Time 5.2 sec.
# NET_ID=tablenet_logitWxW_lut1x1_wta_diffuse_intact  ## 0.9827

# (5) Similar to (2) but enforces semi-hard assignment:
#     Dominant state gets prob_winner, others receive (1 - prob_winner) / (L - 1). Time 5.2 sec.
# NET_ID=tablenet_logitWxW_lut1x1_wta_diffuse_const  ## 0.8067 (prob_winner = 0.5) / 0.9597 (prob_winner = 0.9) / 0.9598 (prob_winner = 0.95)

# (6) Similar to (2) but enforces hard assignment:
#     Dominant state retains its p_max, others receive 0. Time 5.2 sec.
# NET_ID=tablenet_logitWxW_lut1x1_wta_hard_intact  ## 0.9738

# (7) Similar to (2) but enforces hard assignment:
#     Dominant state receives prob_winner, others receive 0. Time 5.2 sec.
# NET_ID=tablenet_logitWxW_lut1x1_wta_hard_const  ## 0.8606 -- oscillates a lot (prob_winner = 0.5)

for NET_ID in tablenet_logitWxW_lut1x1 tablenet_logitWxW_lut1x1_wta_hard_const; do

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

# Get MNIST data and build LMDB files if not already there
${DATA_DIR}/prepare_mnist.sh

# Run 

RUN_TRAIN=1
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
