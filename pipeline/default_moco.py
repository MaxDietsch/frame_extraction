# stuff need to be done, if processing package is not in the PYTHONPATH
import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#import the processing package
from processing import *



#data_root = '/Users/maxdietsch/Desktop/HyperKvasirVideo/extracted-frames/pathological-findings/polyps'
#store_path = '/Users/maxdietsch/Desktop/HyperKvasirVideo/processed-frames/pathological-findings/polyps'

data_root = '/home/stud/dietsch/master-thesis/extracted_frames/polyps'
store_path = '/home/stud/dietsch/master-thesis/extracted_frames/polyps_processed2'



pipeline = Pipeline([
    # load frames and reshape them
    VideoLoader(root_directory = data_root, name = 'frame_loader'),

    # calc sharpness score
    ReShaper(frames = 'frame_loader', target_shape = (640, 640), name = 'frame_reshaper1'),
    GaussianBlurring(frames = 'frame_loader', kernel_shape = (3, 3), name = 'blurred'),
    DeleteFrames(delete = 'frame_loader', name = 'deleter1'),
    FeatureScorer(masks = 'blurred', name = 'feature_score'),
    DeleteFrames(delete = 'blurred', name = 'deleter2'),
    
    #Ranker(score = 'feature_score', name = 'ranker'),
    #PercentileDecimator(ranking = 'ranker', percentile = 70, name = 'decimator'),
    #Selector(frames = 'frame_reshaper1', selection='decimator', name = 'selected_frames'),
    #DeleteFrames(delete = 'frame_reshaper1', name = 'deleter3'),

    
    # calc highlight score
    ToRGB(frames = 'frame_reshaper1', name = 'rgb_converter'),
    HighlightDetector(frames = 'rgb_converter', T1 = 240, T2_abs = 180, T2_rel = 1, T3 = 170, Nmin = 20, kernel_size = 15, inpaint = False, name = 'highlight_masks'),
    HighlightAreaScorer(masks = 'highlight_masks', name = 'highlight_score'),
    DeleteFrames(delete = 'rgb_converter', name = 'deleter4'),
    DeleteFrames(delete = 'highlight_masks', name = 'deleter5'),

    #Ranker(score = 'highlight_score', name = 'highlight_ranker'),
    #PercentileDecimator(ranking = 'highlight_ranker', percentile = 70, name = 'highlight_decimator'),
    #Selector(frames = 'selected_frames', selection='highlight_decimator', name = 'selected_highlight_frames'),
    
    # rank the frames 
    FrameScorer(sharpness_score = 'feature_score', highlight_score = 'highlight_score', n_sharpness_bins = 20, n_highlight_bins = 20, name = 'frame_scorer'),

    # do the sectioning based on motion detection
    GrayScaler(frames = 'frame_reshaper1', name = 'gray_frames'),
    FrameDistanceEstimator(frames = 'gray_frames', masks = None, n_best_matches = 10, name = 'frame_distances'),
    Sectionizer(motion = 'frame_distances', threshold = 5, name = 'frame_sections'),

    # decimate images    
    Decimator(ranking = 'frame_scorer', sections = 'frame_sections', n_frames_per_section = 10, name = 'decimator'),
    
    # select images
    Selector(frames = 'frame_reshaper1', selection = 'decimator', name = 'selected_frames'),

    # store images
    VideoStorer(frames = 'selected_frames', root_directory=store_path, name = 'frame_storer'),



    ])


pipeline.process()
