#### Create a conda env : Since we will work with SplaTam, use its conda env

##### (Recommended)
[SplaTam](https://github.com/spla-tam/SplaTAM) has been tested on python 3.10, CUDA>=11.6. The simplest way to install all dependences is to use [anaconda](https://www.anaconda.com/) and [pip](https://pypi.org/project/pip/) in the following steps: 

```bash
conda create -n splatam python=3.10
conda activate splatam
conda install -c "nvidia/label/cuda-11.6.0" cuda-toolkit
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install -r requirements.txt
```

#### To run SuperPoint on Frieberg desktop data
Note:
1. Data is already part of this repo
2. Code for network and weights are adapted from [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork) by Magicleap
```
python3 my_superpoint.py datasets/frieberg_desktop/  superpoint_v1.pth --cuda
```  
3. This doesnt track detections in consecutive frame to form associations. But you can also find code to track in the original implementation [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork)


```
OpenCV version:  4.9.0
PyTorch version:  1.12.1
usage: my_superpoint.py [-h] [--H H] [--W W] [--nms_dist NMS_DIST] [--conf_thresh CONF_THRESH]
                        [--nn_thresh NN_THRESH] [--cuda] [--display_scale DISPLAY_SCALE] [--no_display]
                        [--waitkey WAITKEY]
                        dataset_path weights_path

This script runs SuperPoint on my data

positional arguments:
  dataset_path          Path to the data directory
  weights_path          Path to the pretrained SuperPointNet weights

options:
  -h, --help            show this help message and exit
  --H H                 Input image height (default used by magicleap = 120)
  --W W                 Input image width (default used by magicleap = 160)
  --nms_dist NMS_DIST   Non Maximum Suppression (NMS) distance (default: 4).
  --conf_thresh CONF_THRESH
                        Detector confidence threshold (default: 0.015).
  --nn_thresh NN_THRESH
                        Descriptor matching threshold (default: 0.7).
  --cuda                Use cuda GPU to speed up network processing speed (default: False)
  --display_scale DISPLAY_SCALE
                        Factor to scale output visualization (default: 2).
  --no_display          Do not display images to screen. Useful if running remotely (default: False).
  --waitkey WAITKEY     OpenCV waitkey time in ms (default: 1).
```
