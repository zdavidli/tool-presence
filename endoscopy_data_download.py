# coding: utf-8

# # Data Extraction from YouTube
#
# In this notebook we download a monocular endoscopic surgery video.
# The video source is: https://www.youtube.com/watch?v=6niL7Poc_qQ.
# We separate the video into individual frames and save them to `data/surgical_video/` and create a PyTorch dataloader to load frames of the video.

from pytube import YouTube
import os
from skimage import io, transform
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm

video_url = 'https://www.youtube.com/watch?v=6niL7Poc_qQ'
data_root = 'data'
video_dir = 'surgical_video_src'
output_dir = 'surgical_video_frames'

if not os.path.exists(os.path.join(data_root, video_dir)):
    os.makedirs(os.path.join(data_root, video_dir))

if not os.path.exists(os.path.join(data_root, output_dir)):
    os.makedirs(os.path.join(data_root, output_dir))


filename = YouTube(video_url).streams.first().download(os.path.join(data_root, video_dir))


vid = imageio.get_reader(filename,  'ffmpeg')


fig = plt.figure()
plt.imshow(vid.get_data(500))


for frame in tqdm(range(340, 1916)):
    #Manual start and end points of video

    image = vid.get_data(frame)
    imageio.imwrite("{0}/frame_{1:04d}.png".format(os.path.join(data_root, output_dir), int(frame)-340), image)


