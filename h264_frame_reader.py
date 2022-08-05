import numpy as np

class H264FrameReader:
    def __init__(self, file_path) -> None:
        '''
        Args:
            file_path: File path to txt file listing frame types in order
        '''
        self.frame_table = np.array(self._get_iframe_indices(file_path))

    def _get_iframe_indices(self, file_path):
        '''Returns a list containing the frame number of each iframe
        '''
        iframe_idx = []

        with open(file_path, 'r') as f:
            frame_idx = 0
            for line in f:
                frame_type = line[-2]
                if frame_type == 'I':
                    iframe_idx.append(frame_idx)
                frame_idx += 1
        
        assert iframe_idx[0] == 0
        self.num_frames = frame_idx

        return iframe_idx

    def get_as_gop(self, frame_idx):
        '''Convert frame index into GOP index and offset
        '''
        gop_idx = np.cumsum(frame_idx >= self.frame_table).max() - 1
        gop_pos = frame_idx - self.frame_table[gop_idx]

        return gop_idx, gop_pos

    def get_as_absolute(self, gop_idx, gop_pos):
        '''Convert GOP index and offset into frame index
        '''
        return self.frame_table[gop_idx] + gop_pos

if __name__ == '__main__':
    reader = H264FrameReader('/home/ryan/Desktop/sciency/compressed_video_detector/data/scenecut300_val/cam1_short.txt')
    print(reader.get_as_absolute(1, 0))
