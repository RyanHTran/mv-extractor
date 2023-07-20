import os
import time
from datetime import datetime
import argparse

import numpy as np
import cv2
from tqdm import tqdm

from mvextractor.videocap import VideoCap

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract motion vectors from video.')
    parser.add_argument('video_url', type=str, nargs='?', default="cam1_short.mp4", help='File path or url of the video stream')
    parser.add_argument('--verbose', action='store_true', default=False, help='Show detailled text output')
    parser.add_argument('--dump', action='store_true', default=False, help='Dump frames, motion vectors, frame types, and timestamps to output directory')
    parser.add_argument('--verify', action='store_true', default=False, help='Verify that mv extraction is correct')
    args = parser.parse_args()

    num_frames = int(cv2.VideoCapture(args.video_url).get(cv2.CAP_PROP_FRAME_COUNT))

    cap = VideoCap()
    # open the video file
    ret = cap.open(args.video_url, 'A', -1, -1, 8, -1)

    if not ret:
        raise RuntimeError(f"Could not open {args.video_url}")
    
    if args.verbose:
        print("Sucessfully opened video file")

    step = 0
    gop_idx = -1
    gop_pos = 0
    # continuously read and display video frames and motion vectors
    with tqdm(total=num_frames) as pbar:
        while True:
            if args.verbose:
                print("Frame: ", step, end=" ")

            # read next video frame and corresponding motion vectors
            ret, frame, motion_vectors, frame_type, gop_idx, gop_pos = cap.read_accumulate()

            # if there is an error reading the frame
            if not ret:
                if args.verbose:
                    print("No frame read. Stopping.")
                break

            frame_height, frame_width = frame.shape[0], frame.shape[1]

            if args.dump:
                save_path = os.path.join('out', 'mv', '{}_{}.npz'.format(gop_idx, gop_pos))
                np.savez_compressed(save_path, (motion_vectors).astype(np.int16))
                cv2.imwrite(os.path.join(f"out", "iframe", '{}_{}.jpg'.format(gop_idx, gop_pos)), frame)

            # print results
            if args.verbose:
                print("({}, {})".format(gop_idx, gop_pos), step, end=" ")
                print("frame type: {} | ".format(frame_type), end=" ")

                print("frame size: {} | ".format(np.shape(frame)), end=" ")
                print("motion vectors: {}".format(np.shape(motion_vectors)))

            step += 1
            pbar.update(1)

    cap.release()
