import cv2
import os

def get_demo():
    frame_folder = "demo"
    output_video = "demo.mp4"
    fps = 2  # frames per second

    # Get all frame filenames sorted
    frames = sorted([f for f in os.listdir(frame_folder) if f.endswith(".png")])

    # Read the first frame to get size
    first_frame = cv2.imread(os.path.join(frame_folder, frames[0]))
    height, width, layers = first_frame.shape

    # Define the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Add frames to the video
    for frame_name in frames:
        frame = cv2.imread(os.path.join(frame_folder, frame_name))
        video.write(frame)

    video.release()
    print(f"Video saved as {output_video}")
