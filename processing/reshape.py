import numpy as np 
import cv2 as cv

from .process import Process
from .video import Video
from typing import Tuple

class ReShaper(Process):
    """
    reshape the frames
    args: 
        target_shape: self-explanatory
    """

    def __init__(self, frames: str, target_shape: Tuple[int, int], **kwargs):
        super().__init__(frames=frames, **kwargs)
        self.target_shape = target_shape

    def _process(self, video: Video):
        reshaped = []
        for frame in self.frames:
            reshaped.append(cv.resize(frame, (self.target_shape[0], self.target_shape[1]),
                                       interpolation=cv.INTER_AREA))
        return reshaped
