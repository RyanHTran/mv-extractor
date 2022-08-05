from h264_frame_reader import H264FrameReader
from coviar_h264 import load
import argparse
import os
import numpy as np
import cv2
from joblib import Parallel, delayed

representations = {'iframe': 0, 'mv': 1, 'residual': 2}

def write_frame(frame_idx, args):
    gop_idx, gop_pos = frame_reader.get_as_gop(frame_idx)
    bgr = load(args.video_url, gop_idx, gop_pos, representations['iframe'], True)
    mv = load(args.video_url, gop_idx, gop_pos, representations['mv'], True)

    cv2.imwrite(os.path.join(f"out", "bgr", f"frame-{frame_idx}.bmp"), bgr)

    if args.verbose:
        print("Frame: ", frame_idx, end=" ")
        print("frame size: {} | ".format(bgr.shape))

    if args.dump:
        save_path = os.path.join('out', 'mv', '{}_{}.npz'.format(gop_idx, gop_pos))
        np.savez_compressed(save_path, mv.astype(np.int16))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract motion vectors from video.')
    parser.add_argument('video_url', type=str, nargs='?', default="cam2_2_raw.mp4", help='File path or url of the video stream')
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=False, help='Show detailled text output')
    parser.add_argument('--dump', action=argparse.BooleanOptionalAction, default=False, help='Dump frames, motion vectors, frame types, and timestamps to output directory')
    args = parser.parse_args()

    frame_reader = H264FrameReader(args.video_url.replace('.mp4', '.txt'))
    print('Starting decode: {}'.format(args.video_url))
    # for frame_idx in range(frame_reader.num_frames):
    #     write_frame(frame_idx, args)
    Parallel(n_jobs=12, verbose=10)(delayed(write_frame)(frame_idx, args) for frame_idx in range(frame_reader.num_frames))