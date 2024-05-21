from .video import Video
from abc import ABCMeta,abstractmethod

class Process(metaclass=ABCMeta):
    """
    General class for a processing step. 
    args: 
        name: the name of the process (used to identify its the outputs of the corresponding process)
        feed: other attributes (often the inputs of the process) which can change from process to process (often this are frames, scores ...)
    """


    def __init__(self, **kwargs):
        self.name = kwargs.pop('name', type(self).__name__)
        self._feed = kwargs


    def is_empty(self):
        """
        say whether the process if finished and there is nothing to process anymore
        """
        return True


    def process(self, video: Video):
        """
        process a video and return the ouput of the video
        """
        
        # set attributes of process object according to its feed
        for key, val in self._feed.items():  # if video is None, _feed is empty
            setattr(self, key, video.get_artifacts(val) if val is not None else None)
        
        # if the video is not none, then we can get the output frames 
        # of the corresponding process object from the video object
        if video is not None:
            video.include_artifact(self._process(video), self.name)

        else:
            video = self._process(None)

        return video


    def __call__(self, video):
        return self.process(video)


    @abstractmethod
    def _process(self, video: Video):
        pass
 
