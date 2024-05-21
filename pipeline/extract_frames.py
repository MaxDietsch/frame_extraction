import cv2
import os

def extract_frames(video_path, output_dir, num_frames=1000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_capture = cv2.VideoCapture(video_path)
    extracted_frames = 0

    while extracted_frames < num_frames and video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        
        frame_filename = os.path.join(output_dir, f"{extracted_frames}.png")
        cv2.imwrite(frame_filename, frame)
        extracted_frames += 1

    video_capture.release()

# Example usage
video_file_path = '../../frame_extraction/videos/polyps_1.avi'
output_directory = '../../frame_extraction/test/polyps'
extract_frames(video_file_path, output_directory)

