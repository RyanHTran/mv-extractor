#!/bin/bash
train_dir="/home/ryan.tran/sciency/compressed_video_detector/data/face_test_videos_gop28"
extraction_dir="/home/ryan.tran/sciency/compressed_video_detector/data/face_test_full_gop28/bidirectional/mv"
iframe_target_dir="/home/ryan.tran/sciency/compressed_video_detector/data/face_test_full_gop28/extraction/iframe"
for video_name in cam1_short cam2_1 cam2_2 cam2_3 cam2_4 cam5_short central_park_1 central_park_2 central_park_3 marathon_short pier_park_1 pier_park_2 pier_park_3 pier_park_4 pier_park_5 pier_park_6 pier_park_7 san_antonio_1 san_antonio_2 walking_past
do
    echo $video_name
    rm -r out/bgr
    mkdir out/bgr
    mkdir out/mv
    # mkdir out/iframe
    python decode_gop.py $train_dir/$video_name".mp4" --dump
    mv out/mv/ $extraction_dir/$video_name
    # mv out/iframe/ $iframe_target_dir/$video_name
done