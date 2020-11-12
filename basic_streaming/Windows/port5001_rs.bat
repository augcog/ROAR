@ECHO OFF
gst-launch-1.0 udpsrc port=5001 ! application/x-rtp,encoding-name=H264,payload=96 ! rtph264depay ! h264parse ! avdec_h264 ! autovideosink