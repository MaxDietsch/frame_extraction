import numpy as np
import cv2 as cv

from .process import Process
from .video import Video 

class FrameDistanceEstimator(Process):

    """
    estimates the distance between subsequent frames
    args:
        n_best_matches: number of best matches which should be used for feature matching between corresponding frames
        frames: the frames where distance should be estimated
        masks: corresponding masks to the frames, used to focus feature detection to specific regions (mostly edges, could be None but not efficient)
    """

    def __init__(self, frames: str, masks: str, n_best_matches: int, **kwargs):
        super().__init__(frames=frames, masks=masks, **kwargs)
        self.orb = cv.ORB_create()
        self.matcher = cv.BFMatcher()
        self.n_best_matches = n_best_matches

    def _approximate_movement(self, frame, mask, next_frame, next_mask):
        """
        approximates movement between frame and next_frame
        """
        # get keypoints and descriptore
        kp1, des1 = self.orb.detectAndCompute(frame, mask)
        kp2, des2 = self.orb.detectAndCompute(next_frame, next_mask)

        try:
            # match descriptors
            matches = self.matcher.match(des1, des2)
            
            # estimate the movement between the descriptors 
            displacement_vectors = []
            for i in range(min(self.n_best_matches, len(matches))):
                kp_id1, kp_id2 = matches[i].queryIdx, matches[i].trainIdx
                pt1, pt2 = kp1[kp_id1].pt, kp2[kp_id2].pt
                displacement_vectors.append((np.array(pt1) - np.array(pt2)))

            if len(displacement_vectors) != 0:
                vectors = np.stack(displacement_vectors, axis=0)
                # outlier robust fusion of the displacement vectors
                displacement = np.median(vectors, axis=0)

                return np.linalg.norm(displacement, ord=2)
            else:
                print('No match found. Use high default distance')
                return np.array(100)

        except cv.error as e:
            print(f'Error during orb feature matching (probable no keypoint was detected in at least one of the 2 frames). Use distance 0. Error message: {str(e)} \n This error is not dramatic, but it should not happen too often during the processing of 1 video.')
            return np.array(0)

    def _process(self, video: Video):
        motions = [0]  # first frame has zero displacement
        for i in range(1, len(self.frames)):
            motions.append(self._approximate_movement(self.frames[i-1],
                                                      np.logical_not(self.masks[i-1]).astype('uint8') * 255 if self.masks else None,
                                                      self.frames[i],
                                                      np.logical_not(self.masks[i]).astype('uint8') * 255 if self.masks else None))
        return motions


class Sectionizer(Process):
    """
    Sectionizes the video into stripes having mostly the same content
    args:
        motion: distance between subsequent frames
        threshold: threshold below which the distance is considered small enough that the content of the 2 frames did not change 
    section_indices (which is returned by _process) is an array containing arrays, 
    the inner arrays have the indices of the frames in which the motion between consecutive frames
    is <= threshold 
    """

    def __init__(self, motion: str, threshold: float, **kwargs):
        super().__init__(motion=motion, **kwargs)
        self.threshold = threshold

    def _process(self, video: Video):
        motion = np.array(self.motion)
        # detect where motion is below threshold
        filtered_frame_indices = np.compress(motion <= self.threshold, np.arange(motion.size))
        # calculate split points of video, points where the motion between 2 frames is > threshold
        split_points = np.nonzero(np.diff(filtered_frame_indices) > 1)[0] + 1
        # split the videos into the individual parts
        section_indices = np.split(filtered_frame_indices, split_points)
        return section_indices


