
import cv2 as cv
import torchvision.transforms.functional as TF
from .process import Process
from .video import Video
import torch
from typing import Tuple

class GaussianBlurring(Process):

    def __init__(self, frames: str, kernel_shape: Tuple[int, int], **kwargs):
        super().__init__(frames=frames, **kwargs)
        self.kernel_shape = kernel_shape

    def _process(self, video: Video):
        blurred = []
        for frame in self.frames:
            blurred.append(cv.GaussianBlur(frame, self.kernel_shape, 0))
        return blurred


class GaussianBlurringGPU(GaussianBlurring):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _process(self, video: Video):
        blurred = []
        for frame in self.frames:
            blurred_tensor = TF.gaussian_blur(frame, self.kernel_shape)
        
            blurred.append(blurred_tensor)  

        return blurred

class MedianBlurring(Process):

    def __init__(self, frames: str, ksize, **kwargs):
        super().__init__(frames=frames, **kwargs)
        self.ksize = ksize

    def _process(self, video: Video):
        filtered = []
        for frame in self.frames:
            filtered.append(cv.medianBlur(frame, ksize=self.ksize))
        return filtered
