import numpy as np

from typing import Union, List


class Video:
    """
    Defines a video
    args:
        video_id: id of the video
        frames: all the frames belonging to the video (including input frames, all intermediate frames etc...)
        producer_name: name of the process which produced the frames 
        artifacts: a dict with keys producer_name (referring to a process) and the 
            value of a key is an array of all output frames of that producer (process)
    """

    def __init__(self,
                 video_id: int,
                 frames: List[np.array],
                 producer_name: str):

        self.video_id = video_id
        self.artifacts = dict([(producer_name, frames)])

    def include_artifact(self, artifact: Union[float, np.array, List[np.array]], producer_name: str):
        """
        add the process (referenced by its name) and its output frames to the artifacts list
        """
        if producer_name in self.artifacts:
            raise ValueError(f"Inconsistent pipeline. The producer {producer_name} occurrs more than once.")
        self.artifacts[producer_name] = artifact

    def has_artifact(self, identifier: str):
        """
        return if results of an process (referenced through identifier) are in the artifact
        """
        return identifier in self.artifacts

    def delete_artifact(self, delete):
        if delete in self.artifacts:
            del self.artifacts[delete]

    def get_artifacts(self, *identifier):
        """
        return the output of the process referred to by identifier (name)
        """
        if len(identifier) == 1:
            return self.artifacts[identifier[0]]
        return [self.artifacts[id] for id in identifier]
