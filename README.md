# HFGI: High-Fidelity GAN Inversion for Image Attribute Editing

Manipulate some attributes (smile, age) of the most significant faces in News images. Only for inference.
Based on the original [HFGI](https://github.com/Tengfei-Wang/HFGI).


## Environment
The environment can be simply set up by Anaconda (only tested for inference):
```
# HFGI
conda create -n myHFGI python=3.6
conda activate myHFGI
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit==10.1.243 -c pytorch
conda install matplotlib
#conda install -c 3dhubs gcc-5  # failed

# Others
pip install opencv-python==4.2.0.34
pip install dlib==19.22.0
conda install scikit-learn
conda install tqdm
```

## Quick Start
### Pretrained Models
Please download the pre-trained model and put it in  `./checkpoint`.
| Model | Description
| :--- | :----------
|[Face Editing](https://drive.google.com/file/d/19y6pxOiJWB0NoG3fAZO9Eab66zkN9XIL/view?usp=sharing)  | Trained on FFHQ.


### Inference
<!-- Modify `inference_fakenews.sh` according to the follwing instructions, and run it.

| Args | Description
| :--- | :----------
| --images_dir | the path of images.
| --n_sample | number of images that you want to infer.
| --edit_attribute | We provide options of 'inversion', 'age', 'smile', 'eyes', 'lip' and 'beard' in the script.
| --edit_degree | control the degree of editing (works for 'age' and 'smile'). -->

TO BE CONTINUED...


## Citation
``` 
@inproceedings{wang2021HFGI,
  title={High-Fidelity GAN Inversion for Image Attribute Editing},
  author={Wang, Tengfei and Zhang, Yong and Fan, Yanbo and Wang, Jue and Chen, Qifeng},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```
