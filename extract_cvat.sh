#!/bin/bash
train_dir="/home/ryan/Desktop/sciency/compressed_video_detector/data/cvat_new_mv/"
for video_name in australian_night_life cloud_city_walk littlegirl-musician nyc_street people_on_the_line
do
    rm -r out/bgr
    mkdir out/bgr
    mkdir out/mv
    mkdir out/iframe
    python decode.py $train_dir$video_name".mp4" --dump
    mv out/mv/ $train_dir$video_name
    mv out/iframe/ $train_dir$video_name
done