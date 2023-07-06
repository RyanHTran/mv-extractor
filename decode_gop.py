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
    parser.add_argument('video_url', type=str, nargs='?', default="cam2_2.mp4", help='File path or url of the video stream')
    parser.add_argument('--verbose', action='store_true', default=False, help='Show detailled text output')
    parser.add_argument('--dump', action='store_true', default=False, help='Dump frames, motion vectors, frame types, and timestamps to output directory')
    parser.add_argument('--verify', action='store_true', default=False, help='Verify that mv extraction is correct')
    args = parser.parse_args()

    num_frames = int(cv2.VideoCapture(args.video_url).get(cv2.CAP_PROP_FRAME_COUNT))

    cap = VideoCap()
    # open the video file
    ret = cap.open(args.video_url, 'A', -1, -1, 8, 15)

    if not ret:
        raise RuntimeError(f"Could not open {args.video_url}")
    
    if args.verbose:
        print("Sucessfully opened video file")

    step = 0
    # continuously read and display video frames and motion vectors
    with tqdm(total=num_frames) as pbar:
        while True:
            if args.verbose:
                print("Frame: ", step, end=" ")

            # read next video frame and corresponding motion vectors
            # ret, frames, frame_type, gop_idx = cap.read_gop()
            ret, frames, forward_mvs, backward_mvs, frame_type, gop_idx = cap.read_accumulate_gop()

            # if there is an error reading the frame
            if not ret:
                if args.verbose:
                    print("No frame read. Stopping.")
                break

            gop_pos = 1
            for i in range(len(frames)):
                frame = frames[i]
                motion_vectors = forward_mvs[i]
                backward_motion_vectors = backward_mvs[i]

                frame_height, frame_width = frame.shape[0], frame.shape[1]
                
                if args.verify:
                    # Check motion vectors
                    load_path = os.path.join('reference_cam2_2', 'mv', '{}_{}.npz'.format(gop_idx, gop_pos))
                    reference_mv = np.load(load_path)['arr_0']
                    if not (motion_vectors == reference_mv).all():
                        print('Decoded motion vectors do not match expected output at frame {}: {}_{}.npz'.format(step, gop_idx, gop_pos))
                        print('Expected: {}'.format(reference_mv[motion_vectors != reference_mv]))
                        print('Got: {}'.format(motion_vectors[motion_vectors != reference_mv]))
                        raise ValueError()
                    # Check frames
                    load_path = os.path.join('reference_cam2_2', 'bgr', '{}_{}.bmp'.format(gop_idx, gop_pos))
                    reference_bgr = cv2.imread(load_path)
                    if not (frame == reference_bgr).all():
                        print('Decoded bgr do not match expected output at frame {}_{}'.format(gop_idx, gop_pos))
                        print('Expected: {}'.format(reference_bgr[frame != reference_bgr]))
                        print('Got: {}'.format(frame[frame != reference_bgr]))
                        raise ValueError()

                if args.dump:
                    stacked_motion_vectors = np.concatenate((motion_vectors, backward_motion_vectors), axis=2)
                    save_path = os.path.join('out', 'mv', '{}_{}.npz'.format(gop_idx, gop_pos))
                    np.savez_compressed(save_path, (stacked_motion_vectors).astype(np.int16))
                    cv2.imwrite(os.path.join(f"out", "iframe", '{}_{}.jpg'.format(gop_idx, gop_pos)), frame)

                # print results
                if args.verbose:
                    print("({}, {})".format(gop_idx, gop_pos), step, end=" ")
                    print("frame type: {} | ".format(frame_type), end=" ")

                    print("frame size: {} | ".format(np.shape(frame)), end=" ")
                    print("motion vectors: {}".format(np.shape(motion_vectors)))

                step += 1
                pbar.update(1)
                gop_pos += 1

    cap.release()
