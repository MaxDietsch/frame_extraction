
import cv2 as cv

from .process import Process
from .video import Video


class CannyEdgeDetector(Process):
    """
    apply canny edge detection to frames
    args: 
        thresh1: threshold 1 for canny edge detector
        thresh2: threshold 2 for canny edge detector
    """

    def __init__(self, frames: str, thresh1: int = 100, thresh2: int = 200, **kwargs):
        super().__init__(frames=frames, **kwargs),
        self.thresh1 = thresh1
        self.thresh2 = thresh2

    def _process(self, video: Video):
        canned = []
        for frame in self.frames:
            canned.append(cv.Canny(frame, self.thresh1, self.thresh2))
        return canned



class Dilation(Process):
    """
    apply a dilation to the frames
    args: 
        kernel_size: size of kernel used for dilation
        iterations: how often the dilation should take place
    """

    def __init__(self, frames: str, kernel_size, iterations, **kwargs):
        super().__init__(frames=frames, **kwargs)
        self.kernel = cv.getStructuringElement(cv.MORPH_RECT, ksize=(kernel_size, kernel_size))
        self.iterations = iterations

    def _process(self, video: Video):
        dilated = []
        for frame in self.frames:
            dilated.append(cv.dilate(src=frame, kernel=self.kernel, iterations=self.iterations))
        return dilated


