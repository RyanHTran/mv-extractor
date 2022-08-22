for i in {1..10}
do
rm out/bgr/*
# time ffmpeg -i cam2_2.mp4 'out/bgr/%04d.bmp'
time python decode.py
# time python decode_frame_only.py
done
