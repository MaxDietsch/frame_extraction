from .process import Process
from .video import Video


class DeleteFrames(Process):
    """
    delete certain frames from the video 
    args: 
        delete: the name of the process which output frames should be deleted
    """

    def __init__(self, delete: str, **kwargs):
        super().__init__(**kwargs)
        self.delete = delete

    def _process(self, video: Video):
        video.delete_artifact(self.delete)
        return None


