
import cv2 as cv
from .process import Process
from .video import Video
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


class MedianBlurring(Process):

    def __init__(self, frames: str, ksize, **kwargs):
        super().__init__(frames=frames, **kwargs)
        self.ksize = ksize

    def _process(self, video: Video):
        filtered = []
        for frame in self.frames:
            filtered.append(cv.medianBlur(frame, ksize=self.ksize))
        return filtered
