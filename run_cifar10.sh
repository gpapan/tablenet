#!/bin/sh

CAFFE_DIR=${HOME}/rmt/work/deeplabel/caffe_google
CAFFE_BIN=${CAFFE_DIR}/.build_release/tools/caffe.bin

EXP=cifar10
NUM_LABELS=10

# Times refer to 100 train iters of 128-image mini-batches on a Titan-X.

# Baselines

#NET_ID=convnet
# --> 0.8982 (with slow train schedule @50K iters)
# --> 0.8863 (with aggresive train schedule @10K iters, like methods below)

#NET_ID=convnet_bottleneck (similar to term_depth)
# --> 0.8637

#NET_ID=convnet_bottleneck_term_1 (using, however the original conv1 layer)
# --> 0.7584

#NET_ID=treenet_logitWxW_lut1x1_noncum_leaf
# --> 0.8318 (alpha: 1, is_cumulative: f, balance_tree_init: f, haar_param: f)

#NET_ID=treenet_logitWxW_lut1x1_noncum_depth
# --> 0.7596

# (200+) Tablenet with tree-structured classifier and LUT stages.
#        No sigmoids, just logits.
#
# (201)
#NET_ID=treenet2_logitWxW_lut1x1_term_depth
# num_terms_per_tree=depth, is_cumulative=true, balance_tree_init=true
# Forward pass: 92.514 ms. Backward pass: 616.978 ms.
# -> 0.8323

# (202)
#NET_ID=treenet2_logitWxW_lut1x1_term_1
# num_terms_per_tree=1, is_cumulative=true, balance_tree_init=true
# Forward pass: 61.4793 ms. Backward pass: 300.621 ms.
# -> 0.7118

# (203)
#NET_ID=treenet2_logitWxW_lut1x1_term_1_balance
# num_terms_per_tree=1, is_cumulative=true, balance_tree_init=true
# -> 0.6768 (balance_tree_loss_weight: 1e-1, using the original conv1 layer)
# -> 0.7974 (balance_tree_loss_weight: 1e-7)
# -> 0.8039 (balance_tree_loss_weight: 0)

# (204)
#NET_ID=treenet2_logitWxW_lut1x1_term_depth_balance
# num_terms_per_tree=depth, is_cumulative=true, balance_tree_init=true, balance_tree_loss_weight: 1e-1
# -> 0.8370

# (300+) Tablenet with tree-structured logit, prob, and LUT stages.
#NET_ID=treenet2_logitWxW_prob_lut1x1_depth5_balance
# -> 0.7821 (depth=5, balance_tree_loss_weight: 1e-7, is_cumulative: true)
# ->  (depth=5, balance_tree_loss_weight: 1e-7, is_cumulative: false)

#NET_ID=treenet2_logitWxW_prob_lut1x1_depth6_balance
# ->  (depth=6, balance_tree_loss_weight: 1e-7)


for NET_ID in treenet2_logitWxW_prob_lut1x1_depth5_balance; do

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
