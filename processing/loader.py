import os
import cv2 as cv
import numpy as np
import json
import torch
from .process import Process
from .video import Video



class VideoLoader(Process):
    """
    load the individual frames of the video 
    args: 
        root_directory: directory containing subdirectories for each individual video
                        in the subdirectories are the frames stored as image files. 
                        the names of the files are of the format: 1.png, 2.jpg etc. 
                        the subdirectories should also be named according to 1, 2, etc. 
    """

    def __init__(self, root_directory: str, **kwargs):
        super().__init__(**kwargs)

        self.video_index = 0
        self.root_directory = root_directory
        self.paths, self.video_names = self.enumerate_paths()

    def enumerate_paths(self):
        """
        for each subdir in root_directory enumerate all frames and return all the paths 
        as well as the names of the subdir
        the paths are returned in 2 dim array: 
            arr[subdir][frames of this subdir]
        """
        subdirs = [d for d in os.listdir(self.root_directory) if not (d.startswith('.'))]
        #subdirs = sorted(d for d in os.listdir(self.root_directory) if d.isdigit(), key=lambda x: int(x))
        paths = []
        for dir in subdirs:
            files = sorted(os.listdir(os.path.join(self.root_directory, dir)), key=lambda x: int(x[:-4]))
            video_paths = []
            for file in files:
                video_paths.append(os.path.join(self.root_directory, dir, file))
            paths.append(video_paths)
        return paths, subdirs

    def is_empty(self):
        """
        while there are unprocessed directories (videos) left do not consider the
        processing step as empty 
        """
        return self.video_index >= len(self.paths)

    def _process(self, video: Video):
        """
        take the current directory (referenced through self.video_index) and read in 
        all its frames, create video object and return this video object.
        ATTENTION: The resulting frames are in BGR-format !!!!!
        """

        frames = []
        for path in self.paths[self.video_index]:
            frames.append(np.asarray(cv.imread(path)))
        video = Video(video_id=self.video_names[self.video_index],
                      frames=frames,
                      producer_name=self.name)
        self.video_index += 1
        return video


class VideoLoaderGPU(VideoLoader):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _process(self, video: Video):
        """
        take the current directory (referenced through self.video_index) and read in 
        all its frames, create video object and return this video object.
        ATTENTION: The resulting frames are in BGR-format !!!!!
        """

        frames = []
        for path in self.paths[self.video_index]:
            frames.append(torch.from_numpy((cv.imread(path))).to(device='cuda', dtype=torch.float32))
        video = Video(video_id=self.video_names[self.video_index],
                      frames=frames,
                      producer_name=self.name)
        self.video_index += 1
        return video



class VideoStorer(Process):
    """
    store the frames of a processed video 
    args: 
        root_directory: directory in which the new subdirs are created. in the subdirs
        are the processed frames. Each created subdir corresponds to a subdir which was 
        found through the loading (see VideoLoader (above))
    """

    def __init__(self, frames: str, root_directory: str, **kwargs):
        super().__init__(frames=frames, **kwargs)
        self.root_directory = root_directory
        os.makedirs(root_directory, exist_ok=True)

    def _process(self, video: Video):
        """
        stores the processed frames (with the same name as read in) in the 
        specifiv subdirs
        ATTENTION: expects the frames to be in GBR format !!!!!
        """
        video_id = video.video_id
        os.makedirs(os.path.join(self.root_directory, str(video_id)), exist_ok=True)
        for i, frame in enumerate(self.frames):
            img = frame
            cv.imwrite(os.path.join(self.root_directory, str(video_id), str(i) + ".png"), img)


class VideoStorerGPU(VideoStorer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def _process(self, video: Video):

        video_id = video.video_id
        os.makedirs(os.path.join(self.root_directory, str(video_id)), exist_ok=True)
        for i, frame in enumerate(self.frames):
            # Ensure the frame is on the CPU and convert to a NumPy array
            img = frame.cpu().numpy().transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
            img = img.astype('uint8')  # Ensure the image is in uint8 format
            cv.imwrite(os.path.join(self.root_directory, str(video_id), str(i) + ".png"), img)
