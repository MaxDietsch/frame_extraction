import numpy as np 
import cv2 as cv
import torch
import torch.nn.functional as F

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

class ReShaperGPU(ReShaper):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _process(self, video: Video):
        reshaped = []
        for frame in self.frames:
            frame_tensor = frame.unsqueeze(0)
            resized_tensor = F.interpolate(frame_tensor, size=self.target_shape, mode='area')
        
            # Remove the batch dimension
            resized_tensor = resized_tensor.squeeze(0)        
            reshaped.append(resized_tensor)

        return reshaped
