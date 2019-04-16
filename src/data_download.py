from pytube import YouTube
import os
import imageio

video_url = 'https://www.youtube.com/watch?v=6niL7Poc_qQ'
data_root = '../data/'
video_dir = 'surgical_video_src'
output_dir = 'surgical_video_frames'

if not os.path.exists(os.path.join(data_root, video_dir)):
    os.makedirs(os.path.join(data_root, video_dir))

if not os.path.exists(os.path.join(data_root, output_dir)):
    os.makedirs(os.path.join(data_root, output_dir))

filename = YouTube(video_url).streams.first().download(
    os.path.join(data_root, video_dir))
vid = imageio.get_reader(filename,  'ffmpeg')

for frame in range(340, 1916):
    # Manual start and end points of video

    image = vid.get_data(frame)
    imageio.imwrite(
        "{0}/frame_{1:04d}.png".format(output_dir, int(frame)-340), image)
