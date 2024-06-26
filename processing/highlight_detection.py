import numpy as np
import cv2 as cv
from .process import Process
from .video import Video


class HighlightDetector(Process):

    """
        follows the paper: Automatic Segmentation and Inpainting of Specular Highlights from Endoscopic Images
        ATTENTION: Assumes RGB images as its frames !!!!!
        args: 
            T1: as in the paper (threshold for module 1, [0, 255])
            T2_abs: as in the paper (threshold for module 2 when applying module 1, [0, 255])
            T2_rel: as in the paper (threshold for final classification in module 2, [0, 255])
            kernel_size: kernel size used for median filter in module 2
            T3: as in the paper (threshold for postprocessing, [0, 255])
            Nmin: as in the paper (for postprocessing, minimal pixels needed to define a contour area as false positive)
            inpaint: whether to inpaint the detected highlights with the proposed method of the paper 

    """

    def __init__(self, frames: str, T1: int, T2_abs: int, T2_rel: float, T3: float, Nmin: int, kernel_size: int, inpaint: bool, **kwargs):
        super().__init__(frames=frames, **kwargs)
        self.T1 = T1 / 255
        self.T2_abs = T2_abs / 255
        self.T2_rel = T2_rel
        self.T3 = T3 / 255
        self.Nmin = Nmin
        self.kernel_size = kernel_size
        self._process_function = self._highlight_detection
        if inpaint:
            self._process_function = self._inpaint_process


    def _normalize(self, frame):
        """
        get gray-scale intnesity values of the frame
        """
        frame = frame.astype('float32') / 255
        frame = 0.2989 * frame[:, :, 0] \
                + 0.587 * frame[:, : ,1] \
                + 0.114 * frame[:, :, 2]
        return frame


    def _calculate_percentile_ratios(self, frame, normalized_frame, channel, percentile: float):
        """
        calculate r.. ratios values for module 1 for a given channel (mainly G or B) and a certain percentile
        """
        norm_flattened = np.reshape(normalized_frame, (-1, ))  # single channel image
        frame_flattened = np.reshape(frame[:, :, channel], (-1, ))
        norm_percentile = np.sort(norm_flattened)[int(percentile * norm_flattened.size)]
        frame_percentile = np.sort(frame_flattened)[int(percentile * frame_flattened.size)]
        return frame_percentile / norm_percentile


    def _module_1(self, frame):
        """
        do calculations for the first module (and the beginning of the second module)
        returns frames for module1 highlight detection, module2 highlight detection and the grayscale intensity frames 
        """
        norm_frame = self._normalize(frame)
        # calc r_ge
        green_percentile_ratio = self._calculate_percentile_ratios(frame, norm_frame, 1, 0.95)
        # calc r_be
        blue_percentile_ratio = self._calculate_percentile_ratios(frame, norm_frame, 2, 0.95)
        # calc highlights for first module 
        highlights_T1 = np.any([frame[:, :, 1] > (green_percentile_ratio * self.T1),
                                frame[:, :, 2] > (blue_percentile_ratio * self.T1 ),
                                norm_frame > (self.T1 * np.ones_like(frame[:, :, 1]))], axis=0)
        
        # calculate highlights for second module (not needded in first module )
        highlights_T2 = np.any([frame[:, :, 1] > (green_percentile_ratio * self.T2_abs),
                                 frame[:, :, 2] > (blue_percentile_ratio * self.T2_abs),
                                 norm_frame > self.T2_abs * np.ones_like(frame[:, :, 1])], axis=0)

        return highlights_T1, highlights_T2, norm_frame


    def _module_2_inpainting(self, org_frame, highlight_mask):
        """
        second phase of module2 (inpainting specular highlights with centroid of surrounding colours)
        highlight_mask should be output of first phase of second module 
        returns the inpainted frame
        """

        frame = np.copy(org_frame)
        # get specular highlight region's surrounding
        kernel_1 = cv.getStructuringElement(cv.MORPH_RECT, ksize=(2, 2))
        kernel_2 = cv.getStructuringElement(cv.MORPH_RECT, ksize=(4, 4))
        highlight_mask = highlight_mask.astype('uint8')
        dilation_1 = cv.dilate(src=highlight_mask, kernel=kernel_1, iterations=1)
        dilation_2 = cv.dilate(src=highlight_mask, kernel=kernel_2, iterations=1)

        surrounding_regions = np.logical_and(dilation_2, np.logical_not(dilation_1)).astype('uint8')
        # find the contours of the regions
        contours, _ = cv.findContours(surrounding_regions, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        # calculate colour centroids
        spatial_centroids = []
        color_centroids = []
        for contour in contours:
            contour = np.squeeze(contour, axis=1)  # n_points x 2
            # position of contour 
            spatial_centroids.append(np.mean(contour, axis=0).astype(int))
            # color centroid of contour 
            color_centroids.append(np.mean(frame[(contour[:, 0], contour[:, 1])], axis=0).astype('uint8'))

        spatial_centroids = np.expand_dims(np.stack(spatial_centroids), axis=0)  # 1 x n_centroids x 2
        color_centroids = np.stack(color_centroids)

        # Assign every highlight candidate to its contour
        nonzero_elements = np.nonzero(highlight_mask > 0)
        highlight_positions = np.transpose(np.stack(nonzero_elements), axes=(1, 0))  # n_highlights x 2
        highlight_positions_exp = np.expand_dims(highlight_positions, axis=1)  # n_highlights x 1 x 2

        distances = np.sum(np.square(spatial_centroids - highlight_positions_exp), axis=-1)  # n_highlights x n_centroids
        centroid_assignment = np.argmin(distances, axis=-1)

        # retrieve colors and conduct inpainting
        highlight_colors = color_centroids[centroid_assignment]
        frame[(highlight_positions[:, 0], highlight_positions[:, 1])] = highlight_colors

        return frame

    def _module_2_highlight_retrieval(self, frame, inpainted_frame):
        """
        final phase for module 2 
        compares input frame and inpainted_frame of second phase of module 2
        returns the highlights
        """

        # median filtering on inpainted image 
        blurred_frame = cv.medianBlur(inpainted_frame, ksize=self.kernel_size)

        channel_means = np.mean(frame, axis=(0, 1))
        channel_stds = np.std(frame, axis=(0, 1))
        # referred to as tau in the paper 
        contrast_coefficient = np.reciprocal((channel_means + channel_stds) / channel_means)
        # calculate epsilon_tilde 
        color_ratios = np.max(frame / (blurred_frame + 1e-6) * np.reshape(contrast_coefficient, newshape=(1, 1, 3)),
                              axis=-1)

        # highlight retrieval:
        highlights = color_ratios > self.T2_rel

        return highlights

    def _highlight_detection(self, frame):
        """
        apply the highlight detection algorithm of the paper. 
        returns the resulting masks 
        """
        # apply module 1
        highlights_t1, highlights_t2, norm_frame = self._module_1(frame)

        # second phase for module 2
        inpainted_frame = self._module_2_inpainting(frame, highlights_t2)

        # final phase for module 2
        highlights_t2 = self._module_2_highlight_retrieval(frame, inpainted_frame)
        
        # postprocessing 
        highlights = np.logical_or(highlights_t1, highlights_t2).astype('uint8')
        
        # applying the gradient criterium
        # Step 1: Compute gradients
        grad_x = cv.Sobel(norm_frame, cv.CV_64F, 1, 0, ksize=3)
        grad_y = cv.Sobel(norm_frame, cv.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        contours, _ = cv.findContours(highlights, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        # Step 3 & 4: Compute mean gradient magnitude along the contour regions and apply the condition
        filtered_contours = []
        for contour in contours:
            # Get the pixel coordinates of the contour
            contour_pixels = np.squeeze(contour, axis = 1) # Shape: (N, 2)
            # Extract gradient magnitudes at the contour points
            grad_values = gradient_magnitude[contour_pixels[:, 1], contour_pixels[:, 0]]
            # Compute mean gradient magnitude
            mean_grad = np.mean(grad_values)
            # Apply the condition
            if mean_grad > self.T3 and len(contour) > self.Nmin:
                filtered_contours.append(contour)
    
        # only the filtered contours should remain white
        mask = np.zeros_like(norm_frame)
        cv.drawContours(mask, filtered_contours, -1, color=255, thickness=cv.FILLED)
        highlights = mask

        # remove salt-and-pepper noise from the mask
        highlights = cv.medianBlur(highlights, ksize=3)

        # remove black highlights
        highlights = np.logical_and(highlights, norm_frame > 0.1).astype('uint8')
        # dilate the highlights
        kernel = cv.getStructuringElement(cv.MORPH_RECT, ksize=(4, 4))
        highlights = cv.dilate(highlights, kernel=kernel, iterations=1)

        # masked pixels indicated with value zero:
        return (highlights).astype('uint8') * 255


    def _inpaint(self, frame, org_frame, highlight_mask):
        # blurr the preliminary interpolated frame
        frame = cv.GaussianBlur(frame, ksize=(11, 11), sigmaX=8)
        # decaying mask (cascade of dilations)
        weight_masks = [highlight_mask]
        acc_dilation = highlight_mask
        kernel = cv.getStructuringElement(cv.MORPH_RECT, ksize=(4, 4))
        for i in range(5):
            dilated = cv.dilate(src=acc_dilation, kernel=kernel, iterations=1)
            weight_masks.append(np.logical_and(dilated, np.logical_not(acc_dilation)).astype('uint8'))
            acc_dilation = dilated
        masks = np.stack(weight_masks, axis=-1)
        weights = np.array([[[1, 0.9, 0.7, 0.4, 0.1, 0.05]]])
        weights = np.sum(masks * weights, axis=-1, keepdims=True)

        inpainted_frame = ((1 - weights) * org_frame + weights * frame).astype('uint8')
        return inpainted_frame

    def _inpaint_process(self, frame):
        highlights = (np.logical_not(self._highlight_detection(frame) / 255)).astype('uint8')
        inpainted_frame_prep = self._module_2_inpainting(frame, highlights)
        fused_frame = self._inpaint(inpainted_frame_prep, frame, highlights)
        return fused_frame

    def _process(self, video: Video):
        res_frames = []
        for frame in self.frames:
            res_frames.append(self._process_function(frame))
        return res_frames

