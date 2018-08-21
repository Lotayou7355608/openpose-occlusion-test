# Funny Experiment on Openpose

This repository showcases the robustness of Openpose 2D human pose estimator against limb occlusions. Adding 20*16 rectangle mask at center of both thighs only reduce the detection rate by 0.105.

### Description

Run the script `pose_estimator_test.py` and sees everything for yourself.
<<<<<<< HEAD
The required pose-estimator model file can be downloaded at [here](https://yadi.sk/d/blgmGpDi3PjXvK)
=======

The required pose-estimator model file can be downloaded at: https://yadi.sk/d/blgmGpDi3PjXvK 

__Prerequsites__
>>>>>>> 98d9a4b293f2068106f68ddc392e4bac478accc9

### Prerequsites
```
tqdm
<<<<<<< HEAD
keras (or tf.keras)
=======

Keras

>>>>>>> 98d9a4b293f2068106f68ddc392e4bac478accc9
numpy

skimage
<<<<<<< HEAD
h5py
```

### Some Images
Here is an example of successful attack, the pose estimator failed to detect left knee and left foot.
![good_image](sample/good_image_00058.jpg)
![good_pose](sample/good_pose_00058.jpg)
![bad_image](sample/bad_image_00058.jpg)
![bad_pose](sample/bad_pose_00058.jpg)

Here is an example of failed attack, the pose estimator failed to detect left foot in the original image, but after the attack, it managed to recover the missing foot, how weird.
![good_image](sample/good_image_00000.jpg)
![good_pose](sample/good_pose_00000.jpg)
![bad_image](sample/bad_image_00000.jpg)
![bad_pose](sample/bad_pose_00000.jpg)



### Misdetection Results

| Amputation Type  | Misdetection Rate |
| :---: | :---: |
| Both thighs | 0.105  |
| Both shins | NaN |

### TODO
- [x] Add some images
- [ ] Add results for other datasets and amputation types
=======

h5py

Please note that Keras is merged into tensorflow now, so it's now `from tensorflow.keras` instead of `from keras`
>>>>>>> 98d9a4b293f2068106f68ddc392e4bac478accc9
