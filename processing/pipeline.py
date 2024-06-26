from .process import Process
from typing import List
import time

class Pipeline:
    """
    Defines a processing pipeline
    args: processes: List[Processes] defines the pipeline through a sequence of 
            processing steps
    """

    def __init__(self, processes: List[Process]):
        self.processes = processes

    def process(self):
        """
        apply each process step to a video sequentially
        """
        while not self.processes[0].is_empty(): # do it for every directory found
            start = time.time()

            print(f'Starting processing cylce: ')
            video = None
            for process in self.processes:
                print(f'    Launching {process.name}')
                video = process(video)

            end = time.time()
            print(f'The processing cycle took {end - start} seconds.')
