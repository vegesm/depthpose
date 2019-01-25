# Depthpose: Absolute Human Pose Estimation with Depth Prediction Network

This repository contains the trained model for the paper. Parts of the code are based on the MegaDepth [repository](https://github.com/lixx2938/MegaDepth).

## Prerequisites
The code was written in Python 2, using the following packages:
- nipgutils
- torch 0.4.1
- numpy 1.15.4
- OpenCV 3

## Setup
You will need the MuPoTS-3D dataset downloaded from [here](http://gvv.mpi-inf.mpg.de/projects/SingleShotMultiPerson/). Also, 2D pose detections are needed, which are available from [here](https://drive.google.com/open?id=1lfepP2IlNXNW6UZEhQaL4XB7KJOaJ-b5). The archive must be extracted in to the root MuPoTS-3D folder.

The pretrained model can be downloaded from [here](https://drive.google.com/open?id=14FpxGlawafRaegeqg1aA6CZUfhKnZWa-). You must extract the contents in the base directory of Dephtpose.

Note if you want to run the model on your own data, you'll have to install [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose). For training the model, commit e59215a219a7328a6b9eb0af14b41c35564d399d was used.

## Running the model

After the data and model weights were downloaded, you can run the baseline algorithm like this:

```bash
python scripts/baseline.py --model-path models/baseline  --mupots <path to mupots dataset>
```

You can optionally add the `--relative` argument to calculate relative errors. 

To evaluate the full model:

```bash
python scripts/full.py --model-path models/end2end/model_weights.pkl  --mupots <path to mupots dataset>
```

The `--relative` switch works here as well.

