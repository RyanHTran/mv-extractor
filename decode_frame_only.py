import os
import time
from datetime import datetime
import argparse

import numpy as np
import cv2

from mv_extractor import VideoCap
# from mvextractor.videocap import VideoCap

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract motion vectors from video.')
    parser.add_argument('video_url', type=str, nargs='?', default="cam2_2.mp4", help='File path or url of the video stream')
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=False, help='Show detailled text output')
    parser.add_argument('--verify', action=argparse.BooleanOptionalAction, default=False, help='Verify that mv extraction is correct')
    args = parser.parse_args()

    cap = VideoCap()

    # open the video file
    ret = cap.open(args.video_url, 'P')

    if not ret:
        raise RuntimeError(f"Could not open {args.video_url}")
    
    if args.verbose:
        print("Sucessfully opened video file")

    step = 0
    # continuously read and display video frames and motion vectors
    while True:
        if args.verbose:
            print("Frame: ", step, end=" ")

        # read next video frame
        ret, frame, frame_type, gop_idx, gop_pos = cap.read()
        # if there is an error reading the frame
        if not ret:
            if args.verbose:
                print("No frame read. Stopping.")
            break

        # cv2.imwrite('reference/bgr_gop/{}_{}.bmp'.format(gop_idx, gop_pos), frame)

        if args.verify:
            # Check frames
            # load_path = os.path.join('reference', 'bgr', 'frame-{}.bmp'.format(step))
            load_path = os.path.join('reference', 'bgr_gop', '{}_{}.bmp'.format(gop_idx, gop_pos))
            reference_bgr = cv2.imread(load_path)
            if not (frame == reference_bgr).all():
                print('Decoded bgr do not match expected output at frame {}'.format(step))
                print('Expected: {}'.format(reference_bgr[frame != reference_bgr]))
                print('Got: {}'.format(frame[frame != reference_bgr]))
                break

        # print results
        if args.verbose:
            print("({}, {})".format(gop_idx, gop_pos), step, end=" ")
            print("frame type: {} | ".format(frame_type), end=" ")
            print("frame size: {} | ".format(np.shape(frame)))

        cv2.imwrite(os.path.join(f"out", "bgr", f"frame-{step}.bmp"), frame)

        step += 1

    cap.release()
