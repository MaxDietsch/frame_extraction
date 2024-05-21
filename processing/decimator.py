import numpy as np

from .process import Process
from .video import Video

class Decimator(Process):

    """
        Each section is assumed to be a new viewing angle. 
        For each viewing angle, the best ranked frames are selected
        are selected.
        Returns indices of the frames which are selected from decimator
        args: 
            ranking: the ranking of the frames
            sections: the sections of frames (could be from Sectionizer)
            n_frames_per_section: number of frames which should be chosen out of one section 
    """

    def __init__(self, ranking: str, sections: str, n_frames_per_section: int, **kwargs):
        super().__init__(ranking=ranking, sections=sections, **kwargs)
        self.n_frames_per_section = n_frames_per_section

    def _process(self, video: Video) -> Video:
        selected_frames = []
        for section in self.sections:
            # sort the frames of one section according to its ranking
            sorted_frames = np.argsort(self.ranking[section])
            # select the best ranked frames
            selected_frame_indices = section[sorted_frames][:min(self.n_frames_per_section, section.size)]
            selected_frames += selected_frame_indices.tolist()
        return selected_frames


class PercentileDecimator(Process):
    """
    only take images which have top values for their sharpness
    return the indices of the corresponding images
    args: 
        percentile: the top percentile which should be used
    """

    def __init__(self, ranking: str, percentile: int, **kwargs):
        super().__init__(ranking = ranking, **kwargs)
        self.percentile = percentile


    def _process(self, video: Video):
        selected_frames = []
        idx = int(self.percentile / 100 * len(self.ranking))
        return self.ranking[ : idx ]


