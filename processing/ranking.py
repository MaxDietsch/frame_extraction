
import numpy as np

from .process import Process
from .video import Video

class FrameScorer(Process):

    """
        Produces a range binning
        Raw scores in [0,1], greater better
        first ranked according the sharpness, after that ranked according to highlight score
        returns indices ranked (ordered) according to the scores
        args:
            sharpness_score: a sharpness_score, could be from IsolatedPixelRatioScorer
            highlight_score: a highlight_score, could be from HighlightAreaScorer
            n_sharpness_bins: number of bins used for sharpness binning
            n_highlight_bins: number of bins used for highlight binning 
    """

    def __init__(self,
                 sharpness_score: str,
                 highlight_score: str,
                 n_sharpness_bins: int,
                 n_highlight_bins: int,
                 **kwargs):
        super().__init__(sharpness_score=sharpness_score,
                         highlight_score=highlight_score,
                         **kwargs)

        self.n_highlight_bins = n_highlight_bins
        self.n_sharpness_bins = n_sharpness_bins

    def _bin_scores(self, score: np.array, n_bins: int):
        """
        the scores are assigned to n_bins number of bins
        """
        # create bins
        bins = np.arange(0, n_bins + 1) * (1 / n_bins)
        # write the scores to the specific bins
        binning = np.digitize(score, bins)
        return binning

    def _process(self, video: Video):
        # calculate the bin affiliation and flip the bins (lower bin better)
        sharpness_binning = self.n_highlight_bins - self._bin_scores(self.sharpness_score, self.n_sharpness_bins)
        highlight_binning = self.n_highlight_bins - self._bin_scores(self.highlight_score, self.n_highlight_bins)
        # do the actual ranking, first sort according to sharpness
        rank = np.lexsort(np.stack([sharpness_binning, 
                                    highlight_binning]))

        return np.argsort(rank)



class InformativeScorer(Process):

    """
    same as above but uses different scores
    """

    def __init__(self,
                 tenengrad_score: str,
                 laplacian_score: str,
                 contrast_score: str,
                 n_tenengrad_bins: int,
                 n_laplacian_bins: int,
                 n_contrast_bins: int, 
                 **kwargs):
        super().__init__(tenengrad_score=tenengrad_score,
                         laplacian_score=laplacian_score,
                         contrast_score=contrast_score,
                         **kwargs)

        self.n_tenengrad_bins = n_tenengrad_bins
        self.n_laplacian_bins = n_laplacian_bins
        self.n_contrast_bins = n_contrast_bins

    def _bin_scores(self, score: np.array, n_bins: int):
        """
        the scores are assigned to n_bins number of bins
        """
        # create bins
        bins = np.arange(0, n_bins + 1) * (1 / n_bins)
        # write the scores to the specific bins
        binning = np.digitize(score, bins)
        return binning

    def _process(self, video: Video):
        # calculate the bin affiliation and flip the bins (lower bin better)
        tenengrad_binning = self.n_tenengrad_bins - self._bin_scores(self.tenengrad_score, self.n_tenengrad_bins)
        laplacian_binning = self.n_laplacian_bins - self._bin_scores(self.laplacian_score, self.n_laplacian_bins)
        contrast_binning = self.n_contrast_bins - self._bin_scores(self.contrast_score, self.n_contrast_bins)
        
        # do the actual ranking, first sort according to sharpness
        rank = np.lexsort(np.stack([tenengrad_binning + laplacian_binning,
                                    laplacian_binning,
                                    contrast_binning]))
        return np.argsort(rank)


class Filter(Process):
    """
    filter images according to some threshold
    """
    
    def __init__(self, tenengrad_score: str, laplacian_score: str, 
                 contrast_score: str, tenengrad_thresh: int, 
                 laplacian_thresh: int, contrast_thresh: int, **kwargs):
        super().__init__(tenengrad_score=tenengrad_score, 
                        laplacian_score = laplacian_score,
                         contrast_score=contrast_score, **kwargs)

        self.tenengrad_thresh = tenengrad_thresh
        self.laplacian_thresh = laplacian_thresh
        self.contrast_thresh = contrast_thresh

    def _process(self, video: Video):
        idx = []
        for i in range(len(self.tenengrad_score)):
            if self.tenengrad_score[i] > self.tenengrad_thresh and self.laplacian_score[i] > self.laplacian_thresh and self.contrast_score[i] > self.contrast_thresh:
                idx.append(i)
        return idx



class Ranker(Process):
    """
    rank frames according to their edge scorer
    return array of indices according to their ranking
    """
    
    def __init__(self, score: str, **kwargs):
        super().__init__(score = score, **kwargs)

    def _process(self, video: Video):
        return np.argsort(self.score)[::-1]

    

