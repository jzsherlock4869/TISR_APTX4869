#!/bin/bash

opt=$1
gpu=$2
basename=`basename $opt`
trackid=$(echo $basename | awk  '{ string=substr($0,1,2); print string; }')
expid=$(echo $basename | awk  '{ string=substr($0,4,3); print string; }')
echo "Started task, track${trackid}, exp ${expid} on GPU no. ${gpu}"

if [ ! -d "./logs" ]
then
    mkdir ./logs
fi
CUDA_VISIBLE_DEVICES=$gpu nohup python -u train_track1.py -opt $opt > logs/track_${trackid}_${expid}_gpu${gpu}.log 2>&1 &
