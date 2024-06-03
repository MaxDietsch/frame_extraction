# stuff need to be done, if processing package is not in the PYTHONPATH
import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#import the processing package
from processing import *



data_root = '/home/stud/dietsch/master-thesis/images/train'
store_path = '/home/stud/dietsch/master-thesis/images/processed'

pipeline = Pipeline([
    # load frames and reshape them
    VideoLoaderGPU(root_directory = data_root, name = 'frame_loader'),
    ReShaperGPU(frames = 'frame_loader', target_shape = (640, 640), name = 'frame_reshaper1'),
    GaussianBlurringGPU(frames = 'frame_loader', kernel_shape = (3, 3), name = 'blurred'),
    DeleteFrames(delete = 'frame_loader', name = 'deleter1'),
    FeatureScorer(masks = 'blurred', name = 'feature_score'),
    DeleteFrames(delete = 'blurred', name = 'deleter2'),
    Ranker(score = 'feature_score', name = 'ranker'),
    PercentileDecimator(ranking = 'ranker', percentile = 50, name = 'decimator'),
    Selector(frames = 'frame_reshaper1', selection='decimator', name = 'selected_frames'),
    DeleteFrames(delete = 'frame_reshaper1', name = 'deleter3'),


    ToRGBGPU(frames = 'selected_frames', name = 'rgb_converter'),
    HighlightDetectorGPU(frames = 'rgb_converter', T1 = 240, T2_abs = 180, T2_rel = 1, T3 = 170, Nmin = 20, kernel_size = 15, inpaint = False, name = 'highlight_masks'),
    HighlightAreaScorer(masks = 'highlight_masks', name = 'highlight_score'),
    DeleteFrames(delete = 'rgb_converter', name = 'deleter4'),
    DeleteFrames(delete = 'highlight_masks', name = 'deleter5'),

    Ranker(score = 'highlight_score', name = 'highlight_ranker'),
    PercentileDecimator(ranking = 'highlight_ranker', percentile = 50, name = 'highlight_decimator'),
    Selector(frames = 'selected_frames', selection='highlight_decimator', name = 'selected_highlight_frames'),
    VideoStorerGPU(frames = 'selected_highlight_frames', root_directory=store_path, name = 'frame_storer'),
    ])


pipeline.process()
