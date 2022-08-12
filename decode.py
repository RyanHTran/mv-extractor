import os
import time
from datetime import datetime
import argparse

import numpy as np
import cv2

from mv_extractor import VideoCap

def get_clipped_corners(coords, xmax, ymax):
    ''' Clips coords between 0 and respective max value
    Args:
        coords: (N, 4) where last dim is [x1, y1, x2, y2]
    '''
    x1, y1, x2, y2 = np.split(coords, 4, axis=1)
    np.clip(x1, 0, xmax, out=x1)
    np.clip(x2, 0, xmax, out=x2)
    np.clip(y1, 0, ymax, out=y1)
    np.clip(y2, 0, ymax, out=y2)

    return np.concatenate((x1, y1, x2, y2), axis=1)

def isnt_clipped(corners, width, height):
    '''
    Args:
        corners: (N, 4) where last dim is [x1, y1, x2, y2]
        width: (N,)
        height: (N,)
    Returns:
        ndarray (N,) where corner has not been clipped
    '''
    return np.logical_and((corners[:, 2] - corners[:, 0]) == width,
        (corners[:, 3] - corners[:, 1]) == height)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract motion vectors from video.')
    parser.add_argument('video_url', type=str, nargs='?', default="cam2_2.mp4", help='File path or url of the video stream')
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=False, help='Show detailled text output')
    parser.add_argument('--dump', action=argparse.BooleanOptionalAction, default=False, help='Dump frames, motion vectors, frame types, and timestamps to output directory')
    parser.add_argument('--verify', action=argparse.BooleanOptionalAction, default=False, help='Verify that mv extraction is correct')
    args = parser.parse_args()

    cap = VideoCap()

    # open the video file
    ret = cap.open(args.video_url)

    if not ret:
        raise RuntimeError(f"Could not open {args.video_url}")
    
    if args.verbose:
        print("Sucessfully opened video file")

    step = 0
    gop_idx = -1
    gop_pos = 0
    # continuously read and display video frames and motion vectors
    while True:
        if args.verbose:
            print("Frame: ", step, end=" ")

        # read next video frame and corresponding motion vectors
        ret, frame, motion_vectors, frame_type, timestamp = cap.read_accumulate()
        # if there is an error reading the frame
        if not ret:
            if args.verbose:
                print("No frame read. Stopping.")
            break

        frame_height, frame_width = frame.shape[0], frame.shape[1]

        if args.dump or args.verify:
            if frame_type == 'I':
                gop_idx += 1
                gop_pos = 0
            else:
                gop_pos += 1

        if args.verify:
            load_path = os.path.join('reference', 'mv', '{}_{}.npz'.format(gop_idx, gop_pos))
            reference_mv = np.load(load_path)['arr_0']
            if not (motion_vectors == reference_mv).all():
                print('Decoded motion vectors do not match expected output at frame {}: {}_{}.npz'.format(step, gop_idx, gop_pos))
                print('Expected: {}'.format(reference_mv[motion_vectors != reference_mv]))
                print('Got: {}'.format(motion_vectors[motion_vectors != reference_mv]))
                break

        if args.dump:
            save_path = os.path.join('out', 'mv', '{}_{}.npz'.format(gop_idx, gop_pos))
            np.savez_compressed(save_path, (motion_vectors).astype(np.int16))

        # print results
        if args.verbose:
            print("timestamp: {} | ".format(timestamp), end=" ")
            print("frame type: {} | ".format(frame_type), end=" ")

            print("frame size: {} | ".format(np.shape(frame)), end=" ")
            print("motion vectors: {}".format(np.shape(motion_vectors)))

        cv2.imwrite(os.path.join(f"out", "bgr", f"frame-{step}.bmp"), frame)

        step += 1

    cap.release()
