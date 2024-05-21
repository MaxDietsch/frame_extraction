
from .process import Process
from .video import Video

class Selector(Process):
    """
        select the frames out of a given indices (selected indices)
        args:
            selection: indices which frames should be included in the output
    """

    def __init__(self, frames: str, selection, **kwargs):
        super().__init__(frames=frames, selection=selection, **kwargs)

    def _process(self, video: Video):
        return [self.frames[i] for i in self.selection]


