# jetson_tensorflowLite_objectTracker

Welcome to the code repository of object tracking using CSI Camera on Jetson nano and RPlidar. This repository 
shows the python backend of Jetson nano where the video frames are processed using tensorflow lite SSD inference
model, receives the depth readings of all the bounding boxes and tracks the objects using | pyimagesearch.tensorTracker.py | 
module. 

- SSD Inference model trained on Coco dataset
- Quantized Tensorflow Lite For mobile use 
- Average CPU Load 1.72 
- Takes up 1.4G Memory
- Preliminary object tracking via ID assignment and error handling




