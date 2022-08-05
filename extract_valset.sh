#!/bin/bash
val_dir="/home/ryan/Desktop/sciency/compressed_video_detector/data/scenecut300_val/"
extraction_dir="/home/ryan/Desktop/sciency/compressed_video_detector/data/scenecut300_val/new_decode/mv/"
for video_name in cam1_short cam5_short marathon_short pier_park_1 walking_past
do
    rm out/bgr/*
    rm out/mv/*
    python decode.py $val_dir$video_name".mp4" --dump
    cp out/mv/* $extraction_dir$video_name"/"
done