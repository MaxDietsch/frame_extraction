import cv2 as cv

from .process import Process
from .video import Video

"""
All classes do not have additional arguments
"""


class ToRGB(Process):
    """
    convert BGR to RGB
    """

    def __init__(self, frames: str, **kwargs):
        super().__init__(frames=frames, **kwargs)

    def _process(self, video: Video):
        rgb_frames = []
        for frame in self.frames:
            rgb_frames.append(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        return rgb_frames


class ToBGR(Process):
    """
    convert RGB to BGR
    """

    def __init__(self, frames: str, **kwargs):
        super().__init__(frames=frames, **kwargs)

    def _process(self, video: Video):
        bgr_frames = []
        for frame in self.frames:
            bgr_frames.append(cv.cvtColor(frame, cv.COLOR_RGB2BGR))
        return bgr_frames


class GrayScaler(Process):
    """
    convert BGR image to gray-scale image
    ATTENTION: assumes BGR image
    """

    def __init__(self, frames: str, **kwargs):
        super().__init__(frames=frames, **kwargs)

    def _process(self, video: Video):
        gray = []
        for frame in self.frames:
            gray.append(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
        return gray
