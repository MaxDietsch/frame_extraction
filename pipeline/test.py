# stuff need to be done, if processing package is not in the PYTHONPATH
import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#import the processing package
from processing import *



data_root = '/Users/maxdietsch/Desktop/test'
store_path = '/Users/maxdietsch/Desktop/results'




pipeline = Pipeline([
    # load frames and reshape them
    VideoLoader(root_directory = data_root, name = 'frame_loader'),

    # delete unneccessary frames
    #DeleteFrames(delete = 'frame_loader', name = 'deleter'),

    
    #GaussianBlurring(frames = 'frame_loader', kernel_shape = (5, 5), name = 'blurred'),
    #GrayScaler(frames = 'blurred', name = 'gray_frames'),
    #TenengradScorer(masks = 'gray_frames', name = 'tenengrad_score'),
    #EnergyOfLaplacianScorer(masks = 'gray_frames', name = 'laplacian_score'),
    #ContrastScorer(masks = 'gray_frames', name = 'contrast_score'),
    #InformativeScorer(tenengrad_score='tenengrad_score', laplacian_score='laplacian_score',
    #                  contrast_score='contrast_score', n_tenengrad_bins=30, n_laplacian_bins=30,
    #                  n_contrast_bins=30, name = 'ranker'),

    #PercentileDecimator(ranking = 'ranker', percentile = 1000, name = 'decimator'),
    #Selector(frames = 'frame_reshaper', selection = 'decimator', name = 'selected_frames'),
    #VideoStorer(frames = 'selected_frames', root_directory = store_path, name = 'frame_storer'),
    
    # not bad
    #GrayScaler(frames = 'frame_loader', name = 'gray_frames'),
    #GaussianBlurring(frames = 'gray_frames', kernel_shape = (15, 15), name = 'blurred'),
    #TenengradScorer(masks = 'blurred', name = 'tenengrad_score'),
    #Ranker(score = 'tenengrad_score', name = 'ranker'),
    #PercentileDecimator(ranking = 'ranker', percentile = 1000, name = 'decimator'),
    #Selector(frames = 'frame_reshaper', selection='decimator', name = 'selected_frames'),
    #VideoStorer(frames = 'selected_frames', root_directory=store_path, name = 'frame_storer'),


    # quite good results (eventually without smoothing to have otehr ones)

    #GaussianBlurring(frames = 'frame_loader', kernel_shape = (3, 3), name = 'blurred'),
    #CannyEdgeDetector(frames = 'blurred', thresh1 = 200, thresh2 = 300, name = 'edge_masks'),
    #CountEdgeScorer(masks = 'edge_masks', name = 'edge_scorer'),
    #Ranker(score = 'edge_scorer', name = 'ranker'),
    #PercentileDecimator(ranking = 'ranker', percentile = 1000, name = 'decimator'),
    #Selector(frames = 'frame_reshaper', selection = 'decimator', name = 'selected_frames'),
    #VideoStorer(frames = 'selected_frames', root_directory = store_path, name = 'frame_storer'),
    

    # good overall pipeline:
    ReShaper(frames = 'frame_loader', target_shape = (640, 640), name = 'frame_reshaper1'),
    #GaussianBlurring(frames = 'frame_loader', kernel_shape = (3, 3), name = 'blurred'),
    #DeleteFrames(delete = 'frame_loader', name = 'deleter1'),
    #FeatureScorer(masks = 'blurred', name = 'feature_score'),
    #DeleteFrames(delete = 'blurred', name = 'deleter2'),
    #Ranker(score = 'feature_score', name = 'ranker'),


    ToRGB(frames = 'frame_reshaper1', name = 'rgb_converter'),
    HighlightDetector(frames = 'rgb_converter', T1 = 240, T2_abs = 165, T2_rel = 1, T3 = 170, Nmin = 10, kernel_size = 15, inpaint = False, name = 'highlight_masks'),

    VideoStorer(frames = 'frame_reshaper1', root_directory=store_path, name = 'frame_storer'),

    ])


pipeline.process()
