clone under /opt/nvdia/deepstream/deepstream-6.3/sources/deepstream_python_apps/apps/
To run:
  $ python3 deepstream_imagedata-multistream.py <uri1> [uri2] ... [uriN] <FOLDER NAME TO SAVE FRAMES>
e.g.
  $ python3 deepstream_imagedata-multistream.py file:///home/ubuntu/video1.mp4 file:///home/ubuntu/video2.mp4 frames
  $ python3 deepstream_imagedata-multistream.py rtsp://127.0.0.1/video1 rtsp://127.0.0.1/video2 frames
