import numpy as np
import cv2 as cv
from .process import Process
from .video import Video

class HighlightAreaScorer(Process):
    """
    score the highlighted area (from highlight detection)
    score is calculated according to ratio of size(specular highlights) / size(whole image)
    """
    def __init__(self, masks: str, **kwargs):
        super().__init__(masks=masks, **kwargs)

    def _process(self, video: Video):
        scores = []
        for mask in self.masks:
            scores.append(1 - np.sum(mask) / (255 * mask.size))
        return scores


class IsolatedPixelRatioScorer(Process):
    """
    takes as input canny edge filtered image and calculates IPR (Isolated Pixel Ratio)
    """

    def __init__(self, masks: str, **kwargs):
        super().__init__(masks = masks, **kwargs)

    def _process(self, video: Video):
        iprs = []
        kernel = np.array([ [1, 1, 1],
                            [1, 0, 1],
                            [1, 1, 1]], dtype=np.uint8)
        for mask in self.masks: 
            # Perform the convolution using cv2.filter2D
            neighbor_count = cv.filter2D(mask, -1, kernel, borderType=cv.BORDER_CONSTANT)

            # Isolated pixels have zero neighbors
            isolated_pixels = (mask == 1) & (neighbor_count <= 1)
            # Count the number of isolated pixels
            isolated_pixel_count = np.sum(isolated_pixels)
            ipr = isolated_pixel_count / (mask.shape[0] * mask.shape[1])
            iprs.append(ipr)
        return iprs


class CountEdgeScorer(Process):
    """
    takes as input canny edge filtered image and calculates score according to sumed
    edges from the canny edge detected edges
    """

    def __init__(self, masks: str, **kwargs):
        super().__init__(masks = masks, **kwargs)

    def _process(self, video: Video):
        iprs = []
        for mask in self.masks: 
            iprs.append(np.sum(mask > 0))
        return iprs


class TenengradScorer(Process):
    """
    score image edges according to tenengrad_variance score
    the input masks should be the grayscale images
    """
    
    def __init__(self, masks: str, **kwargs):
        super().__init__(masks = masks, **kwargs)

    def _process(self, video: Video):
        iprs = []
        for mask in self.masks: 
            gx = cv.Sobel(mask, cv.CV_64F, 1, 0, ksize=3)
            gy = cv.Sobel(mask, cv.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(gx**2 + gy**2)
            
            iprs.append(np.var(grad_mag))
        return iprs


class EnergyOfLaplacianScorer(Process):
    """
    score image edges according to energy of laplacian score
    the input masks should be the grayscale images
    """
    
    def __init__(self, masks: str, **kwargs):
        super().__init__(masks = masks, **kwargs)

    def _process(self, video: Video):
        iprs = []
        for mask in self.masks:
            laplacian = cv.Laplacian(mask, cv.CV_64F)
            iprs.append(np.sum(laplacian**2))
        return iprs

class ContrastScorer(Process):
    """
    score image according to contrast (std) score
    the input masks should be the grayscale images
    """
    
    def __init__(self, masks: str, **kwargs):
        super().__init__(masks = masks, **kwargs)

    def _process(self, video: Video):
        iprs = []
        for mask in self.masks:
            contrast = np.std(mask)
            iprs.append(contrast)
        return iprs



