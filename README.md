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
