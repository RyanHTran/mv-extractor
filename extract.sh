#!/bin/bash
train_dir="/home/ryan/Desktop/sciency/compressed_video_detector/data/unlabeled_moving_persons/"
extraction_dir="/home/ryan/Desktop/sciency/compressed_video_detector/data/unlabeled_moving_persons/new_decode/mv/"
iframe_target_dir="/home/ryan/Desktop/sciency/compressed_video_detector/data/unlabeled_moving_persons/extraction/iframe/"
for video_name in rainy4 rainy5 seoul1 seoul2 seoul3 seoul4
do
    rm -r out/bgr
    mkdir out/bgr
    mkdir out/mv
    mkdir out/iframe
    python decode.py $train_dir$video_name".mp4" --dump
    mv out/mv/ $extraction_dir$video_name
    mv out/iframe/ $iframe_target_dir$video_name
done