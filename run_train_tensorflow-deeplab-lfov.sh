#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

train_script=${HOME}/codes/PycharmProjects/tensorflow-deeplab-lfov/train.py

#dataDir="/home/hlc/Data/aws_SpaceNet/un_gz/voc_format"
dataDir=/home/hlc/Data/VOCdevkit/VOC2012
#data_list=/home/hlc/Data/aws_SpaceNet/un_gz/voc_format/AOI_2_Vegas_Train/trainval_aug_path.txt
#num_classes=2
batchSize=16
learning_rate=1e-5
SNAPSHOT_DIR=snapshots_03


pre_train_model=${HOME}/Data/deeplab/pre-train/tensorflow-deeplab-lfov/model.ckpt-init
#pre_train_model=./snapshots/model.ckpt-3000.data-00000-of-00001

#run script with python3 and tensorflow
#python ${train_script} --data_dir ${dataDir} --data_list ${data_list} --restore_from ${pre_train_model} --num-classes ${num_classes} --batch_size ${batchSize}   --save_num_images 5

python ${train_script} --data_dir ${dataDir} --restore_from ${pre_train_model}  --batch_size ${batchSize}   --save_num_images 5 --save_pred_every 2000 --learning_rate ${learning_rate}   --snapshot_dir ${SNAPSHOT_DIR}
