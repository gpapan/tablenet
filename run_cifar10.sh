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

#NET_ID=treenet1_logitWxW_lut1x1_noncum_leaf
# --> 0.8318 (alpha: 1, is_cumulative: f, balance_tree_init: f, haar_param: f, 50K + 20K iters)

#NET_ID=treenet1_logitWxW_lut1x1_noncum_depth
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
# -> 0.8196 (depth=5/6, balance_tree_loss_weight: 1e-7, is_cumulative: true)
# -> 0.8196 (depth=5/6, balance_tree_loss_weight: 0, is_cumulative: true)
# -> 0.8030 (depth=5/6, balance_tree_loss_weight: 0, is_cumulative: false). cf: treenet_logitWxW_lut1x1_noncum_leaf

#NET_ID=treenet2_logitWxW_prob_lut1x1_depth6_balance
# -> 0.7287 (depth=6/7, balance_tree_loss_weight: 1e-7, is_cumulative: true)

#NET_ID=treenet2_logitWxW_prob_lut1x1_depth4_balance
# -> 0.8235 (depth=4/5, balance_tree_loss_weight: 1e-7, is_cumulative: true)

#NET_ID=treenet2_logitWxW_prob_lut1x1_depth4
# -> 0.8209 (depth=4/5, balance_tree_init: f, balance_tree_loss_weight: 0, is_cumulative: true)

#NET_ID=treenet2_logitWxW_prob_lut1x1_depth3_balance
# -> 0.8041 (depth=3/4, balance_tree_loss_weight: 1e-7, is_cumulative: true)

#NET_ID=treenet2_logitWxW_prob_lut1x1_depth2_balance
# -> 0.7807 (depth=2/3, balance_tree_loss_weight: 1e-7, is_cumulative: true)

#NET_ID=treenet2_logitWxW_prob_lut1x1_depth1_balance
# -> 0.7217 (depth=1/2, balance_tree_loss_weight: 1e-7, is_cumulative: true)

# (400+) Tablenet with stochastic tree-structured logit, prob, and LUT stages.

# NET_ID=treenet3_logitWxW_prob_lut1x1_depth5_sample_soft
# depth=5, temperature: 0.1 (train) 0.0 (test), flip_prob: 0, num_samples: 1, hard_paths: false
# A. temperature: 0.1 in classify, 1 in prob layers
# -> 0.8231 temperature: 0.0 (test), num_samples: 1
# -> 0.8240 temperature: 0.1 (test), num_samples: 1
# -> 0.8266 temperature: 0.1 (test), num_samples: 10
# B. temperature: 0.1 in both classify and prob layers -> does not work
# C. temperature: 0.1 in classify, 0.5 in prob layers (gpapan.mtv)
# -> 0.7059

# NET_ID=treenet3_logitWxW_prob_lut1x1_depth5_sample_hard
# depth=5, temperature: 0.1 (train) 0.1 (test), flip_prob: 0, num_samples: 1, hard_paths: true
# A. temperature: 0.1 in classify, 1 in prob layers (gpapan.mtv)
# -> 0.6023
# B. temperature: 1 in classify (weight std: 1), 1 in prob layers (mavra.lax)
# -> killed (0.2777 @ 4500 iters)

# NET_ID=treenet3_logitWxW_prob_lut1x1_depth5_flip_soft
# depth=5, temperature: 0, flip_prob in train only, num_samples: 1, hard_paths: false
# -> 0.7200 flip_prob: 0.05
# -> 0.8232 flip_prob: 0.01 (mavra.lax)
# -> 0.8211 flip_prob: 0 (gpapan.mtv)

# NET_ID=treenet3_logitWxW_prob_lut1x1_depth5_sample_soft
# depth=5, temperature: 0.1 (train) 0.1 (test), flip_prob: 0, hard_paths: false
# -> 0.8272 num_samples (train+test): 2 (mavra.lax)

# (500+) TreeNet with configurable number of bottleneck layer terms.

# NET_ID=treenet_beg_depth3_bottleneck_norelu
# model2.prototxt (bottleneck-style, depth = 3, group = 4) temperature: 0.1 (train) 0.1 (test)
# -> 0.8573


for NET_ID in treenet_beg_depth3_bottleneck_norelu; do

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
    TEST_ITER=10  # 10,000 images in 100 batches
    CMD="${CAFFE_BIN} time \
         --model=${CONFIG_DIR}/model2.prototxt \
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
