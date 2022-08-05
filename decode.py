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
    args = parser.parse_args()

    cap = VideoCap()

    # open the video file
    ret = cap.open(args.video_url)

    if not ret:
        raise RuntimeError(f"Could not open {args.video_url}")
    
    if args.verbose:
        print("Sucessfully opened video file")

    COORDS = None
    step = 0
    gop_idx = -1
    gop_pos = 0
    # continuously read and display video frames and motion vectors
    while True:
        if args.verbose:
            print("Frame: ", step, end=" ")

        # read next video frame and corresponding motion vectors
        ret, frame, motion_vectors, frame_type, timestamp = cap.read()
        # if there is an error reading the frame
        if not ret:
            if args.verbose:
                print("No frame read. Stopping.")
            break

        frame_height, frame_width = frame.shape[0], frame.shape[1]

        if step == 0 or frame_type == 'I':
            if COORDS is None:
                COORDS = np.ones((frame_height, frame_width, 2))
                COORDS[:, :, 0] = np.cumsum(COORDS[:, :, 0], axis=1) - 1
                COORDS[:, :, 1] = np.cumsum(COORDS[:, :, 1], axis=0) - 1
            # Reset accumulated motion vectors
            prev_mv_accumulate = COORDS.copy()
            curr_mv_accumulate = COORDS.copy()
        if frame_type != 'I':
            # Motion vector accumulation
            idx = np.any(motion_vectors[:, 5:7] != motion_vectors[:, 3:5], axis=1)
            nonzero_mv = motion_vectors[idx]
            # assert bool(np.all(nonzero_mv[:,0] == -1))
            window_size = nonzero_mv[:, 1:3] // 2
            src_ctr, dst_ctr = nonzero_mv[:, 3:5], nonzero_mv[:, 5:7]

            src_corner = np.concatenate((src_ctr - window_size, src_ctr + window_size), axis=1)
            src_corner = get_clipped_corners(src_corner, frame_width, frame_height)

            dst_corner = np.concatenate((dst_ctr - window_size, dst_ctr + window_size), axis=1)
            dst_corner = get_clipped_corners(dst_corner, frame_width, frame_height)

            unclipped_src = isnt_clipped(src_corner, nonzero_mv[:, 1], nonzero_mv[:, 2])
            unclipped_dst = isnt_clipped(dst_corner, nonzero_mv[:, 1], nonzero_mv[:, 2])
            unclipped_idx = np.logical_and(unclipped_src, unclipped_dst)
            src_corner = src_corner[unclipped_idx]
            dst_corner = dst_corner[unclipped_idx]

            for i in range(src_corner.shape[0]):
                # x1, y1, x2, y2
                src, dst = src_corner[i], dst_corner[i]
                curr_mv_accumulate[dst[1]:dst[3], dst[0]:dst[2]] = \
                    prev_mv_accumulate[src[1]:src[3], src[0]:src[2]]

        if args.dump:
            if frame_type == 'I':
                gop_idx += 1
                gop_pos = 0
            else:
                gop_pos += 1

            save_path = os.path.join('out', 'mv', '{}_{}.npz'.format(gop_idx, gop_pos))
            np.savez_compressed(save_path, (COORDS - curr_mv_accumulate).astype(np.int16))
            
        np.copyto(prev_mv_accumulate, curr_mv_accumulate)

        # print results
        if args.verbose:
            print("timestamp: {} | ".format(timestamp), end=" ")
            print("frame type: {} | ".format(frame_type), end=" ")

            print("frame size: {} | ".format(np.shape(frame)), end=" ")
            print("motion vectors: {}".format(np.shape(motion_vectors)))

        cv2.imwrite(os.path.join(f"out", "bgr", f"frame-{step}.bmp"), frame)

        step += 1

    cap.release()
