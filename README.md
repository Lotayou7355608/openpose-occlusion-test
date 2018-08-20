# Funny Experiment on Openpose

This repository showcases the robustness of Openpose 2D human pose estimator against limb occlusions. Adding 20*16 rectangle mask at center of both thighs only reduce the detection rate by 0.105.

__Description__

Run the script `pose_estimator_test.py` and sees everything for yourself.
The required pose-estimator model file can be downloaded at: https://yadi.sk/d/blgmGpDi3PjXvK 

__Prerequsites__

tqdm
Keras
numpy
skimage
h5py