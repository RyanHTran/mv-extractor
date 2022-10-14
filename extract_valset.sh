#!/bin/bash
val_dir="/home/ryan/Desktop/sciency/compressed_video_detector/data/gop145_val/"
extraction_dir="/home/ryan/Desktop/sciency/compressed_video_detector/data/gop145_val/new_decode/mv/"
iframe_target_dir="/home/ryan/Desktop/sciency/compressed_video_detector/data/gop145_val/extraction/iframe/"
for video_name in cam1_short cam5_short marathon_short pier_park_1 walking_past
do
    rm -r out/bgr
    mkdir out/bgr
    mkdir out/mv
    mkdir out/iframe
    python decode.py $val_dir$video_name".mp4" --dump
    mv out/mv/ $extraction_dir$video_name
    mv out/iframe/ $iframe_target_dir$video_name
done