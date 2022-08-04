for i in {1..10}
do
# rm tmp/*
# time ffmpeg -i cam2_2.mp4 'tmp/%04d.bmp'
rm out/frames/*
time python decode.py
done
