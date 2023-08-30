# Install

    git clone https://github.com/RyanHTran/mv-extractor.git
    cd mv-extractor
    pip install .

# Quickstart
`decode.py` gives an example of how to extract accumulated motion vectors using the `mvextractor.videocap` module. To test that it is working, run

    python decode.py VIDEO_PATH --verbose 

To use `decode.py` to save motion vectors to disk,

    mkdir -p out/mv
    mkdir -p out/iframe
    python decode.py VIDEO_PATH --dump

This will save the motion vectors and extracted frames to the directory `out`.

# Implementing Residual Accumulation
[FFmpeg version 4.3](https://github.com/FFmpeg/FFmpeg/tree/release/4.3) was cloned into `src/mvextractor/`, and the `include`s in `src/mvextractor/video_cap.hpp` have been modified to point to this custom version. So, `pip install .` should build this extension with the custom FFmpeg. The most promising places to look are `src/mvextractor/FFmpeg/libavcodec/h264_cavlc.c` and `src/mvextractor/FFmpeg/libavcodec/h264_cabac.c`. One of these two are used during h264 decoding. There are several functions related to decoding residuals, and it is possible that reconstruction is happening here as well.