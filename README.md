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