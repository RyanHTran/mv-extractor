import os
import time
from datetime import datetime
import argparse

import numpy as np
import cv2

from mv_extractor import VideoCap


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract motion vectors from video.')
    parser.add_argument('video_url', type=str, nargs='?', default="cam2_2.mp4", help='File path or url of the video stream')
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=False, help='Show detailled text output')
    args = parser.parse_args()

    cap = VideoCap()

    # open the video file
    ret = cap.open(args.video_url)

    if not ret:
        raise RuntimeError(f"Could not open {args.video_url}")
    
    if args.verbose:
        print("Sucessfully opened video file")

    step = 0

    # continuously read and display video frames and motion vectors
    while True:
        if args.verbose:
            print("Frame: ", step, end=" ")

        # read next video frame and corresponding motion vectors
        ret, frame, motion_vectors, frame_type, timestamp = cap.read()

        # displacement = motion_vectors[:, 3:5] - motion_vectors[:, 1:3]
        # import pdb; pdb.set_trace()
        # motion_vectors = motion_vectors[]
        # if there is an error reading the frame
        if not ret:
            if args.verbose:
                print("No frame read. Stopping.")
            break

        # print results
        if args.verbose:
            print("timestamp: {} | ".format(timestamp), end=" ")
            print("frame type: {} | ".format(frame_type), end=" ")

            print("frame size: {} | ".format(np.shape(frame)), end=" ")
            print("motion vectors: {}".format(np.shape(motion_vectors)))

        cv2.imwrite(os.path.join(f"out", "frames", f"frame-{step}.bmp"), frame)

        step += 1

    cap.release()
